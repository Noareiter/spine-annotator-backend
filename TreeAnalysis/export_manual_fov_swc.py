"""Export FOV SWC traces transformed into the whole-cell frame.

Reads ``manual_registration.json`` and writes per-FOV SWC files with geometry
``dressed`` onto the whole-cell SWC (same coordinate frame as the cell trace).

Outputs (under ``--out-dir``):

- ``<fov>_in_cell_um.swc`` — x/y/z in cell µm
- ``<fov>_in_cell_px.swc`` — x/y/z in pixels on ``fov1.tif`` (z = slice index)
- ``all_fovs_in_cell_px.swc`` — merged coloured fragments for QC in Fiji

CLI:
    python export_manual_fov_swc.py config/GP04.json --session pre-droplet \\
        --registration C:/Users/Jackie/Downloads/manual_registration_GP04_pre-droplet.json \\
        --out-dir results/manual_registration_swc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from export_registration_scene import export_scene
from manual_transform import apply_point, fov_matrices, load_manual_registration
from run_tree_analysis import (
    _fov_swc_to_stage,
    _load_combined_swc,
    _resolve_swc_list,
    load_config,
)


def write_swc(df: pd.DataFrame, path: Path, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if header:
            fh.write(f"# {header}\n")
        for _, r in df.iterrows():
            fh.write(
                f"{int(r['n'])} {int(r['type'])} {float(r['x']):.6f} {float(r['y']):.6f} "
                f"{float(r['z']):.6f} {float(r['radius']):.6f} {int(r['parent'])}\n"
            )


def z_um_to_slice(z_um: float, z_um_per_page: List[float]) -> float:
    if not z_um_per_page:
        return z_um
    best_i = 0
    best_d = abs(z_um_per_page[0] - z_um)
    for i, zu in enumerate(z_um_per_page):
        d = abs(zu - z_um)
        if d < best_d:
            best_d = d
            best_i = i
    return float(best_i + 1)


def transform_staged_to_cell_um(staged: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    out = staged.copy()
    xs, ys, zs = [], [], []
    for _, row in staged.iterrows():
        q = apply_point(matrix, float(row["x_um"]), float(row["y_um"]), float(row["z_um"]))
        xs.append(q[0])
        ys.append(q[1])
        zs.append(q[2])
    out["x_um_cell"] = xs
    out["y_um_cell"] = ys
    out["z_um_cell"] = zs
    return out


def to_swc_um(df: pd.DataFrame) -> pd.DataFrame:
    swc = df[["n", "type", "radius", "parent"]].copy()
    swc["x"] = df["x_um_cell"]
    swc["y"] = df["y_um_cell"]
    swc["z"] = df["z_um_cell"]
    return swc


def to_swc_px(df: pd.DataFrame, mpp_x: float, mpp_y: float, z_um_per_page: List[float]) -> pd.DataFrame:
    swc = df[["n", "type", "radius", "parent"]].copy()
    swc["x"] = df["x_um_cell"] / mpp_x
    swc["y"] = df["y_um_cell"] / mpp_y
    swc["z"] = [z_um_to_slice(z, z_um_per_page) for z in df["z_um_cell"]]
    return swc


def export_aligned_swc(
    cfg_path: Path,
    session: str,
    registration_path: Path,
    out_dir: Path,
) -> dict:
    cfg = load_config(cfg_path)
    base = cfg_path.parent.parent
    scene = export_scene(cfg, base, session)
    manual = load_manual_registration(registration_path)
    mats = fov_matrices(manual)

    mpp_x, mpp_y = scene["cell"]["mpp"]
    z_um_per_page = scene["cell"].get("z_um_per_page", [])
    axis = cfg.get("axis_convention", {})
    sign_x = int(axis.get("sign_x", 1))
    sign_y = int(axis.get("sign_y", 1))

    merged_parts: List[pd.DataFrame] = []
    node_offset = 0
    summary = {}

    for fov in cfg.get("sessions", {}).get(session, {}).get("fovs", []):
        label = f"fov{fov.get('fov', '?')}"
        if label not in mats:
            continue
        swc_files = _resolve_swc_list(base, fov.get("swc", ""))
        xml_path = Path(fov.get("pvscan_xml", ""))
        if not swc_files:
            continue
        from io_pvscan_xml import parse_pvscan

        meta = parse_pvscan(xml_path)
        staged = _fov_swc_to_stage(_load_combined_swc(swc_files), meta, sign_x, sign_y)
        cell_df = transform_staged_to_cell_um(staged, mats[label])

        um_swc = to_swc_um(cell_df)
        px_swc = to_swc_px(cell_df, mpp_x, mpp_y, z_um_per_page)

        um_path = out_dir / f"{label}_in_cell_um.swc"
        px_path = out_dir / f"{label}_in_cell_px.swc"
        write_swc(um_swc, um_path, "coordinates: cell_um (x_um, y_um, z_um)")
        write_swc(px_swc, px_path, "coordinates: cell pixels on fov1.tif; z = slice index")

        part = px_swc.copy()
        part["n"] = part["n"] + node_offset
        part["parent"] = part["parent"].apply(lambda p: p + node_offset if p != -1 else -1)
        part["type"] = 5  # custom colour in SNT
        merged_parts.append(part)
        node_offset = int(part["n"].max()) + 1
        summary[label] = {"nodes": len(px_swc), "um_swc": str(um_path), "px_swc": str(px_path)}

    if merged_parts:
        merged = pd.concat(merged_parts, ignore_index=True)
        merged_path = out_dir / "all_fovs_in_cell_px.swc"
        write_swc(
            merged,
            merged_path,
            "all FOV dendrites in cell pixel frame (load with fov1.tif + whole-cell swc)",
        )
        summary["merged_px"] = str(merged_path)

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Export FOV SWCs dressed onto whole-cell frame")
    parser.add_argument("config", help="config/<ANIMAL>.json")
    parser.add_argument("--session", required=True)
    parser.add_argument("--registration", required=True, help="manual_registration.json")
    parser.add_argument("--out-dir", default="results/manual_registration_swc")
    args = parser.parse_args(argv[1:] if argv else None)

    cfg_path = Path(args.config).resolve()
    reg_path = Path(args.registration).resolve()
    out_dir = Path(args.out_dir)

    summary = export_aligned_swc(cfg_path, args.session, reg_path, out_dir)
    print(f"wrote -> {out_dir.resolve()}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
