"""Orchestrate the whole TreeAnalysis pipeline from a per-animal config.

Reads a ``config/<ANIMAL>.json`` (see ``config/GP04.json``) and runs every
implemented step end-to-end, writing all intermediates and figures into a fresh
dated results folder so each run is reproducible and self-contained:

    results/<YYYY-MM-DD>_treeanalysis_<animal>/
        cell_nodes.csv          calibrated whole-cell SWC
        cell_graph.csv          annotated tree graph
        cell_bp.csv             whole-cell branch points
        <session>_mosaic.csv    stitched spines (stage um)
        <session>_mosaic_bp.csv mosaic skeleton branch points
        <session>_transform.json registration result
        <session>_spines_in_cell.csv  spines mapped to cell frame
        <session>_spines_assigned.csv structural coordinates per spine
        qc/...                  overlays + residual plots

Stages whose inputs are missing in the config (e.g. an empty ``swc`` path) are
skipped with a clear message, so the orchestrator is usable while the config is
still being filled in. Paths are resolved relative to the config file's parent
unless absolute.

CLI:
    python run_tree_analysis.py config/GP04.json
    python run_tree_analysis.py config/GP04.json --outdir results --inlier-um 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from apply_transform import apply_transform
from assign_to_branch import assign_spines, assignment_summary
from build_tree_graph import build_tree
from calibrate_swc import calibrate
from correct_swc_z import ZSchedule
from extract_branchpoints import extract_branchpoints
from io_pvscan_xml import PVScanMeta, fov_pixel_to_stage_um, parse_pvscan
from io_spines import parse_detected_spines
from io_swc import parse_swc
from manual_transform import apply_manual_to_mosaic, load_manual_registration, to_pipeline_report
from register_mosaic_to_cell import Similarity3D, _load_pairs, register
from stitch_session_fovs import _nearest_frame_index, fov_spines_to_stage


def _strip_comments(obj):
    """Recursively drop ``_comment`` keys so configs can be self-documenting."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if k != "_comment"}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return _strip_comments(cfg)


def _resolve(base: Path, p: str) -> Optional[Path]:
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def _exists(p: Optional[Path]) -> bool:
    return p is not None and p.exists()


def _resolve_swc_list(base: Path, value) -> List[Path]:
    """Resolve a ``swc`` config value into a sorted list of .swc files.

    Accepts a single path, a list of paths, or a folder (globbed for
    ``*.swc``). Non-existent entries are dropped. An empty value yields ``[]``.
    """
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    out: List[Path] = []
    for it in items:
        p = _resolve(base, it)
        if p is None:
            continue
        if p.is_dir():
            out.extend(sorted(p.glob("*.swc")))
        elif p.exists():
            out.append(p)
    return out


def _load_combined_swc(paths: List[Path]) -> pd.DataFrame:
    """Parse one or more SWC files into a single forest with unique node ids.

    Node and parent ids are offset per file so disconnected fragments (e.g. the
    several SNT exports traced from one FOV) coexist without id collisions.
    """
    parts: List[pd.DataFrame] = []
    offset = 0
    for p in paths:
        df = parse_swc(p)
        n_max = int(df["n"].max())
        df = df.copy()
        df["n"] = df["n"] + offset
        df["parent"] = df["parent"].where(df["parent"] == -1, df["parent"] + offset)
        parts.append(df)
        offset += n_max + 1
    return pd.concat(parts, ignore_index=True)


def _z_schedule_from_cfg(cfg_zs: dict) -> ZSchedule:
    return ZSchedule(
        breakpoints=cfg_zs.get("breakpoints", []),
        steps=cfg_zs.get("steps", [1.0]),
        z0_um=float(cfg_zs.get("z0_um", 0.0)),
    )


def build_whole_cell(cfg: dict, base: Path, outdir: Path, log: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
    """Steps 1/3/4/5/7 for the whole-cell overview. Returns graph + branch points."""
    wc = cfg.get("whole_cell", {})
    swc_files = _resolve_swc_list(base, wc.get("swc", ""))
    xml_path = _resolve(base, wc.get("pvscan_xml", ""))
    if not swc_files:
        log.append(f"[whole_cell] SKIP: swc not found ({wc.get('swc') or 'empty'})")
        return None
    if not _exists(xml_path):
        log.append(f"[whole_cell] SKIP: pvscan_xml not found ({wc.get('pvscan_xml') or 'empty'})")
        return None

    meta = parse_pvscan(xml_path)
    mpp_x, mpp_y, _ = meta.microns_per_pixel
    schedule = _z_schedule_from_cfg(wc["z_schedule"])

    nodes = calibrate(_load_combined_swc(swc_files), microns_per_pixel_x=mpp_x, microns_per_pixel_y=mpp_y, z_schedule=schedule)
    graph = build_tree(nodes, prefer_um=True)
    bp = extract_branchpoints(graph)

    nodes.to_csv(outdir / "cell_nodes.csv", index=False)
    graph.to_csv(outdir / "cell_graph.csv", index=False)
    bp.to_csv(outdir / "cell_bp.csv", index=False)
    log.append(f"[whole_cell] OK: {len(graph)} nodes, {len(bp)} branch points (mpp x={mpp_x:.4f})")
    return {"graph": graph, "bp": bp}


def _fov_swc_to_stage(swc_df: pd.DataFrame, meta: PVScanMeta, sign_x: int, sign_y: int) -> pd.DataFrame:
    """Convert a FOV SWC node table (pixels) into stage micrometres."""
    out = swc_df.copy()
    xs, ys, zs = [], [], []
    for _, r in out.iterrows():
        frame = _nearest_frame_index(meta, float(r["z"]))
        sx, sy, sz = fov_pixel_to_stage_um(meta, float(r["x"]), float(r["y"]), frame, sign_x=sign_x, sign_y=sign_y)
        xs.append(sx)
        ys.append(sy)
        zs.append(sz)
    out["x_um"], out["y_um"], out["z_um"] = xs, ys, zs
    return out


def _stitch_fov_skeletons(fov_swcs: List[Tuple[pd.DataFrame, PVScanMeta, str]], sign_x: int, sign_y: int) -> pd.DataFrame:
    """Combine multiple FOV SWCs (stage um) into one forest with unique ids.

    A ``fov`` column records which FOV each node came from, so downstream QC
    (e.g. the registration canvas) can colour branch points by FOV.
    """
    parts: List[pd.DataFrame] = []
    offset = 0
    for swc_df, meta, label in fov_swcs:
        staged = _fov_swc_to_stage(swc_df, meta, sign_x, sign_y)
        staged = staged.copy()
        staged["n"] = staged["n"] + offset
        staged["parent"] = staged["parent"].where(staged["parent"] == -1, staged["parent"] + offset)
        staged["fov"] = label
        parts.append(staged)
        offset += int(swc_df["n"].max()) + 1
    return pd.concat(parts, ignore_index=True)


def process_session(
    name: str,
    session: dict,
    cfg: dict,
    base: Path,
    cell: Optional[Dict[str, pd.DataFrame]],
    outdir: Path,
    inlier_um: float,
    dedup_um: float,
    log: List[str],
) -> None:
    """Steps 2/6 (always) + 8/9/10/11 (when whole-cell + FOV traces exist)."""
    axis = cfg.get("axis_convention", {})
    sign_x = int(axis.get("sign_x", 1))
    sign_y = int(axis.get("sign_y", 1))

    fovs = session.get("fovs", [])
    spine_parts: List[pd.DataFrame] = []
    fov_swcs: List[Tuple[pd.DataFrame, PVScanMeta, str]] = []
    for fov in fovs:
        xml_path = _resolve(base, fov.get("pvscan_xml", ""))
        sp_path = _resolve(base, fov.get("respan_detected_spines", ""))
        swc_files = _resolve_swc_list(base, fov.get("swc", ""))
        label = f"fov{fov.get('fov', '?')}"
        if not _exists(xml_path):
            log.append(f"[{name}/{label}] SKIP: pvscan_xml not found")
            continue
        meta = parse_pvscan(xml_path)
        if _exists(sp_path):
            staged = fov_spines_to_stage(parse_detected_spines(sp_path), meta, label, sign_x=sign_x, sign_y=sign_y)
            spine_parts.append(staged)
        else:
            log.append(f"[{name}/{label}] note: no detected_spines (spines skipped)")
        if swc_files:
            fov_swcs.append((_load_combined_swc(swc_files), meta, label))
            log.append(f"[{name}/{label}] swc: {len(swc_files)} trace file(s)")
        else:
            log.append(f"[{name}/{label}] note: no FOV swc (skeleton branch points skipped)")

    if not spine_parts:
        log.append(f"[{name}] SKIP: no spines stitched")
        return
    mosaic = pd.concat(spine_parts, ignore_index=True)
    mosaic.to_csv(outdir / f"{name}_mosaic.csv", index=False)
    log.append(f"[{name}] OK: {len(mosaic)} spines stitched into mosaic")

    if cell is None or not fov_swcs:
        log.append(f"[{name}] partial: registration needs whole-cell graph + FOV swc traces")
        return

    mosaic_skel = _stitch_fov_skeletons(fov_swcs, sign_x, sign_y)
    mosaic_graph = build_tree(mosaic_skel, prefer_um=True)
    mosaic_graph.to_csv(outdir / f"{name}_mosaic_graph.csv", index=False)
    mosaic_bp = extract_branchpoints(mosaic_graph)
    mosaic_bp.to_csv(outdir / f"{name}_mosaic_bp.csv", index=False)
    if len(mosaic_bp) < 3:
        log.append(f"[{name}] SKIP register: only {len(mosaic_bp)} mosaic branch points (need >=3)")
        return

    # Labeled branch-point map: lets you build the manual pairs CSV by reading
    # off corresponding node ids. Always produced, even if registration is poor.
    try:
        from qc_overlay import plot_branchpoint_map

        plot_branchpoint_map(
            cell["graph"], cell["bp"], mosaic_graph, mosaic_bp,
            outdir / "qc" / f"{name}_branchpoint_map.png",
        )
        log.append(f"[{name}] branch-point map -> qc/{name}_branchpoint_map.png")
    except Exception as exc:
        log.append(f"[{name}] branch-point map skipped: {exc}")

    # Manual 3D registration (from manual_register_3d.html) takes highest priority.
    manual_path = _resolve(base, session.get("manual_registration", ""))
    if _exists(manual_path):
        manual = load_manual_registration(manual_path)
        result = to_pipeline_report(manual)
        (outdir / f"{name}_transform.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.append(f"[{name}] register: manual_3d ({len(manual.get('fov_transforms', {}))} FOV transforms) from {manual_path.name}")

        spines_cell = apply_manual_to_mosaic(mosaic, manual)
        spines_cell.to_csv(outdir / f"{name}_spines_in_cell.csv", index=False)

        assigned = assign_spines(spines_cell, cell["graph"])
        assigned.to_csv(outdir / f"{name}_spines_assigned.csv", index=False)
        summ = assignment_summary(assigned, cell["graph"])
        log.append(
            f"[{name}] assign: {summ['n_spines']} spines, "
            f"median dist {summ['median_dist_to_dendrite_um']:.2f} um "
            f"(xy {summ['median_dist_xy_um']:.2f}, z {summ['median_dist_z_um']:.2f}), "
            f"orders {summ['branch_order_range']}"
        )
        try:
            from qc_overlay import plot_skeleton_spines

            qc = outdir / "qc"
            plot_skeleton_spines(cell["graph"], assigned, qc / f"{name}_skeleton_spines.png")
            log.append(f"[{name}] QC figures written to {qc}")
        except Exception as exc:
            log.append(f"[{name}] QC skipped: {exc}")
        return

    # Manual branch-point pairs (mosaic_n,cell_n) take precedence over automatic.
    pairs = None
    pairs_path = _resolve(base, session.get("branchpoint_pairs", ""))
    if _exists(pairs_path):
        pairs = _load_pairs(str(pairs_path))
        log.append(f"[{name}] using manual pairs: {len(pairs)} from {pairs_path.name}")
    elif session.get("branchpoint_pairs"):
        log.append(f"[{name}] note: branchpoint_pairs set but not found ({session.get('branchpoint_pairs')}); using automatic")

    try:
        result = register(mosaic_bp, cell["bp"], pairs=pairs, inlier_um=inlier_um)
    except ValueError as exc:
        log.append(f"[{name}] SKIP register: {exc}")
        return
    (outdir / f"{name}_transform.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.append(
        f"[{name}] register: {result['method']}, scale={result['scale']:.3f}, "
        f"matched={result['n_matched']}/{result['n_mosaic_bp']}, rms={result['icp_rms_um']:.2f} um"
    )

    T = Similarity3D.from_dict(result["transform"])
    spines_cell = apply_transform(mosaic, T)
    spines_cell.to_csv(outdir / f"{name}_spines_in_cell.csv", index=False)

    assigned = assign_spines(spines_cell, cell["graph"])
    assigned.to_csv(outdir / f"{name}_spines_assigned.csv", index=False)
    summ = assignment_summary(assigned, cell["graph"])
    log.append(
        f"[{name}] assign: {summ['n_spines']} spines, "
        f"median dist {summ['median_dist_to_dendrite_um']:.2f} um "
        f"(xy {summ['median_dist_xy_um']:.2f}, z {summ['median_dist_z_um']:.2f}), "
        f"orders {summ['branch_order_range']}"
    )

    try:
        from qc_overlay import plot_bp_residuals, plot_skeleton_spines

        qc = outdir / "qc"
        plot_skeleton_spines(cell["graph"], assigned, qc / f"{name}_skeleton_spines.png")
        plot_bp_residuals(
            T.apply(mosaic_bp[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)),
            cell["bp"][["x_um", "y_um", "z_um"]].to_numpy(dtype=float),
            qc / f"{name}_bp_residuals.png",
            inlier_um=inlier_um,
        )
        log.append(f"[{name}] QC figures written to {qc}")
    except Exception as exc:  # matplotlib optional / headless issues
        log.append(f"[{name}] QC skipped: {exc}")


def preflight(cfg: dict, base: Path) -> List[Tuple[str, str, str]]:
    """Check every input path in the config exists before the heavy run.

    Returns rows of ``(label, status, resolved_path)``. For ``swc`` entries the
    status reports how many ``.swc`` files were found (folders are globbed).
    """
    rows: List[Tuple[str, str, str]] = []

    def check_file(label: str, value: str) -> None:
        if not value:
            rows.append((label, "EMPTY", ""))
            return
        p = _resolve(base, value)
        rows.append((label, "OK" if _exists(p) else "MISSING", str(p)))

    def check_swc(label: str, value) -> None:
        if not value:
            rows.append((label, "EMPTY", ""))
            return
        files = _resolve_swc_list(base, value)
        target = _resolve(base, value if not isinstance(value, list) else value[0])
        rows.append((label, f"OK ({len(files)} swc)" if files else "MISSING", str(target)))

    wc = cfg.get("whole_cell", {})
    check_file("whole_cell.pvscan_xml", wc.get("pvscan_xml", ""))
    check_swc("whole_cell.swc", wc.get("swc", ""))

    for name, session in cfg.get("sessions", {}).items():
        for fov in session.get("fovs", []):
            lab = f"{name}/fov{fov.get('fov', '?')}"
            check_file(f"{lab}.pvscan_xml", fov.get("pvscan_xml", ""))
            check_file(f"{lab}.detected_spines", fov.get("respan_detected_spines", ""))
            check_swc(f"{lab}.swc", fov.get("swc", ""))
    return rows


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Run the full TreeAnalysis pipeline from a config file.")
    parser.add_argument("config", help="path to config/<ANIMAL>.json")
    parser.add_argument("--outdir", default="results", help="parent results directory (default: results)")
    parser.add_argument("--inlier-um", type=float, default=5.0, help="branch-point inlier threshold (um)")
    parser.add_argument("--dedup-um", type=float, default=0.0, help="overlap dedup radius (um); 0 disables")
    parser.add_argument("--check", action="store_true", help="preflight only: verify paths and exit")
    args = parser.parse_args(argv[1:])

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    base = cfg_path.parent.parent  # config/ lives under the project root
    animal = cfg.get("animal_id", cfg_path.stem)

    checks = preflight(cfg, base)
    missing = [r for r in checks if r[1] == "MISSING"]
    print(f"=== preflight: {animal} ({len(checks)} inputs, {len(missing)} missing) ===")
    for label, status, path in checks:
        mark = "OK " if status.startswith("OK") else ("-- " if status == "EMPTY" else "!! ")
        print(f"  {mark}{label:34} {status:12} {path}")
    if args.check:
        return 1 if missing else 0
    if missing:
        print(f"\nWARNING: {len(missing)} path(s) missing; those stages will be skipped.\n")

    outdir = Path(args.outdir) / f"{date.today().isoformat()}_treeanalysis_{animal}"
    outdir.mkdir(parents=True, exist_ok=True)

    log: List[str] = [f"=== TreeAnalysis: {animal} ===", f"config: {cfg_path}", f"output: {outdir}", ""]
    log.append("preflight:")
    for label, status, _ in checks:
        log.append(f"  {label:34} {status}")
    log.append("")

    cell = build_whole_cell(cfg, base, outdir, log)

    for name, session in cfg.get("sessions", {}).items():
        process_session(name, session, cfg, base, cell, outdir, args.inlier_um, args.dedup_um, log)

    report = "\n".join(log)
    (outdir / "run_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nrun report -> {outdir / 'run_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
