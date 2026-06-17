"""Export a calibrated 3D scene for the manual registration HTML tool.

Reads ``config/<ANIMAL>.json`` and writes ``registration_scene.json`` containing:

- whole-cell skeleton segments in **cell µm** (``x_um, y_um, z_um``),
- one FOV group per high-mag field, skeleton in **stage µm** (``x_um, y_um, z_um``),
- metadata (animal, session, axis convention).

Load this file in ``manual_register_3d.html`` (recommended) instead of raw SWCs
so calibration (mpp, z-schedule, stage positions) is already applied.

CLI:
    python export_registration_scene.py config/GP04.json --session pre-droplet \\
        --out registration_scene_GP04_pre-droplet.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from run_tree_analysis import (
    _exists,
    _fov_swc_to_stage,
    _load_combined_swc,
    _resolve,
    _resolve_swc_list,
    _z_schedule_from_cfg,
    load_config,
)
from calibrate_swc import calibrate
from correct_swc_z import slice_to_depth_um
from io_pvscan_xml import parse_pvscan
from endpoint_register import trace_endpoints
from tiff_slices import stack_info

_FOV_COLORS = {
    "fov1": "#7B64B8",
    "fov2": "#1F8A65",
    "fov3": "#E8C030",
    "fov4": "#C85898",
    "fov5": "#2E79B5",
    "fov6": "#F0A040",
    "fov7": "#2A9A8A",
    "fov8": "#C06028",
}


def _segments_from_nodes(df: pd.DataFrame, x: str = "x_um", y: str = "y_um", z: str = "z_um") -> List[List[List[float]]]:
    """Parent→child line segments for Three.js LineSegments."""
    coord: Dict[int, List[float]] = {
        int(r["n"]): [round(float(r[x]), 2), round(float(r[y]), 2), round(float(r[z]), 2)] for _, r in df.iterrows()
    }
    parent: Dict[int, int] = {int(r["n"]): int(r["parent"]) for _, r in df.iterrows()}
    segs: List[List[List[float]]] = []
    for n, p in parent.items():
        if p == -1 or p not in coord:
            continue
        segs.append([coord[p], coord[n]])
    return segs


def _bounds(segs: List[List[List[float]]]) -> List[float]:
    if not segs:
        return [0, 0, 0, 0, 0, 0]
    xs, ys, zs = [], [], []
    for a, b in segs:
        for p in (a, b):
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    return [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]


def _decimate_segments(segs: List[List[List[float]]], max_segments: int, stride: int = 1) -> List[List[List[float]]]:
    """Reduce segment count for browser / API payloads."""
    if stride > 1:
        segs = segs[::stride]
    if len(segs) <= max_segments:
        return segs
    step = max(1, len(segs) // max_segments)
    return segs[::step]


def _resolve_cell_tiff(wc: dict, xml_path: Path, base: Path) -> Optional[str]:
    """Resolve whole-cell TIFF stack path (multi-page ``fov1.tif`` style)."""
    explicit = wc.get("tiff_stack") or wc.get("tiff")
    if explicit:
        p = _resolve(base, explicit)
        if _exists(p):
            return str(Path(p).resolve())
    xml_path = Path(xml_path)
    # .../26_02_17/1_aligned/TSeries-.../file.xml -> .../26_02_17/fov1.tif
    session_dir = xml_path.parent.parent.parent
    for name in ("fov1.tif", "FOV1.tif", "whole_cell.tif"):
        candidate = session_dir / name
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def export_scene(cfg: dict, base: Path, session_name: str) -> dict:
    animal = cfg.get("animal_id", "unknown")
    session = cfg.get("sessions", {}).get(session_name)
    if not session:
        raise ValueError(f"session {session_name!r} not found in config")

    axis = cfg.get("axis_convention", {})
    sign_x = int(axis.get("sign_x", 1))
    sign_y = int(axis.get("sign_y", 1))

    wc = cfg.get("whole_cell", {})
    swc_files = _resolve_swc_list(base, wc.get("swc", ""))
    xml_path = _resolve(base, wc.get("pvscan_xml", ""))
    if not swc_files or not _exists(xml_path):
        raise ValueError("whole_cell swc and pvscan_xml required")

    meta = parse_pvscan(xml_path)
    cell_mpp_x, cell_mpp_y, _ = meta.microns_per_pixel
    cell_pixels = meta.pixels
    schedule = _z_schedule_from_cfg(wc["z_schedule"])
    raw_cell = _load_combined_swc(swc_files)
    cell_nodes = calibrate(
        raw_cell,
        microns_per_pixel_x=cell_mpp_x,
        microns_per_pixel_y=cell_mpp_y,
        z_schedule=schedule,
    )
    cell_segs = _segments_from_nodes(cell_nodes)
    cell_segs_px = _segments_from_nodes(raw_cell, x="x", y="y", z="z")
    cell_bounds_um = _bounds(cell_segs)

    cell_tiff = _resolve_cell_tiff(wc, xml_path, base)
    n_pages = 0
    z_um_per_page: List[float] = []
    if cell_tiff and Path(cell_tiff).is_file():
        n_pages, _, _ = stack_info(Path(cell_tiff))
        for p in range(n_pages):
            z_val = slice_to_depth_um(float(p + 1), schedule)
            z_um_per_page.append(round(float(np.asarray(z_val).ravel()[0]), 3))

    fovs_out = {}
    for fov in session.get("fovs", []):
        label = f"fov{fov.get('fov', '?')}"
        swc_files = _resolve_swc_list(base, fov.get("swc", ""))
        xml_path = _resolve(base, fov.get("pvscan_xml", ""))
        if not swc_files or not _exists(xml_path):
            continue
        meta = parse_pvscan(xml_path)
        raw_swc = _load_combined_swc(swc_files)
        staged = _fov_swc_to_stage(raw_swc, meta, sign_x, sign_y)
        segs = _segments_from_nodes(staged)
        segs_px = _segments_from_nodes(raw_swc, x="x", y="y", z="z")
        fov_mpp_x, fov_mpp_y, _ = meta.microns_per_pixel
        seen_idx: set[int] = set()
        frames = []
        for fr in meta.frames:
            if fr.index in seen_idx:
                continue
            seen_idx.add(fr.index)
            frames.append(
                {
                    "index": fr.index,
                    "z_um": round(fr.z_um, 3),
                    "file": fr.files[0] if fr.files else None,
                }
            )
        frames.sort(key=lambda x: x["index"])
        fb = _bounds(segs)
        fov_cx = (fb[0] + fb[1]) / 2
        fov_cy = (fb[2] + fb[3]) / 2
        fov_cz = (fb[4] + fb[5]) / 2
        cell_cx = (cell_bounds_um[0] + cell_bounds_um[1]) / 2
        cell_cy = (cell_bounds_um[2] + cell_bounds_um[3]) / 2
        cell_cz = (cell_bounds_um[4] + cell_bounds_um[5]) / 2
        try:
            ep_a, ep_b = trace_endpoints(segs)
        except ValueError:
            ep_a, ep_b = [round(fov_cx, 2), round(fov_cy, 2), round(fov_cz, 2)], [
                round(fov_cx, 2),
                round(fov_cy, 2),
                round(fov_cz, 2),
            ]
        fovs_out[label] = {
            "fov": int(fov.get("fov", 0)),
            "color": _FOV_COLORS.get(label, "#599CE7"),
            "segments": segs,
            "segments_pixel": segs_px,
            "bounds_um": fb,
            "bounds_px": _bounds(segs_px),
            "stage_center_um": list(meta.stage_center_um),
            "centroid_stage_um": [round(fov_cx, 2), round(fov_cy, 2), round(fov_cz, 2)],
            "endpoints_stage_um": [
                [round(float(c), 2) for c in ep_a],
                [round(float(c), 2) for c in ep_b],
            ],
            "initial_offset_um": [
                round(cell_cx - fov_cx, 2),
                round(cell_cy - fov_cy, 2),
                round(cell_cz - fov_cz, 2),
            ],
            "mpp": [round(fov_mpp_x, 6), round(fov_mpp_y, 6)],
            "size_px": [int(meta.pixels[0]), int(meta.pixels[1])],
            "n_frames": len(frames),
            "frames": frames,
            "pvscan_xml": str(xml_path),
            "n_segments": len(segs),
        }

    return {
        "version": 2,
        "animal_id": animal,
        "session": session_name,
        "coordinate_frames": {
            "cell": "cell_um (x_um, y_um, z_um from whole-cell trace + z_schedule)",
            "fovs": "stage_um (x_um, y_um, z_um from FOV trace + PVScan stage)",
        },
        "cell": {
            "segments": cell_segs,
            "segments_pixel": cell_segs_px,
            "bounds_um": cell_bounds_um,
            "bounds_px": _bounds(cell_segs_px),
            "centroid_px": [
                round((cell_bounds_um[0] + cell_bounds_um[1]) / 2 / cell_mpp_x, 1),
                round((cell_bounds_um[2] + cell_bounds_um[3]) / 2 / cell_mpp_y, 1),
            ],
            "mpp": [round(cell_mpp_x, 6), round(cell_mpp_y, 6)],
            "size_px": [int(cell_pixels[0]), int(cell_pixels[1])],
            "tiff_stack": cell_tiff,
            "n_pages": n_pages,
            "z_um_per_page": z_um_per_page,
            "n_segments": len(cell_segs),
        },
        "fovs": fovs_out,
    }


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Export registration_scene.json for manual_register_3d.html")
    parser.add_argument("config", help="config/<ANIMAL>.json")
    parser.add_argument("--session", required=True, help="session name (e.g. pre-droplet)")
    parser.add_argument("--out", help="output JSON path (default: registration_scene_<animal>_<session>.json)")
    args = parser.parse_args(argv[1:] if argv else None)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    base = cfg_path.parent.parent
    animal = cfg.get("animal_id", cfg_path.stem)
    session = args.session

    scene = export_scene(cfg, base, session)
    out = Path(args.out) if args.out else Path(f"registration_scene_{animal}_{session}.json")
    out.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    print(f"animal   : {animal}")
    print(f"session  : {session}")
    print(f"cell segs: {scene['cell']['n_segments']}")
    print(f"fovs     : {list(scene['fovs'].keys())}")
    print(f"wrote    : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
