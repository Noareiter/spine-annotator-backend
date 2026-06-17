"""Export assigned spines as SWC point clouds on the whole-cell image.

Reads ``*_spines_assigned.csv`` (cell µm) and writes marker SWC in pixel
coordinates for overlay on ``fov1.tif`` + whole-cell dendrite SWC in Fiji.

CLI:
    python export_spines_to_swc.py --spines results/.../pre-droplet_spines_assigned.csv \\
        --config config/GP04.json --out results/.../pre-droplet_spines_in_cell_px.swc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from export_registration_scene import export_scene
from run_tree_analysis import load_config


def write_marker_swc(
    path: Path,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    header: str,
    radius: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {header}\n")
        for i, (x, y, z) in enumerate(zip(xs, ys, zs), start=1):
            fh.write(f"{i} 6 {x:.4f} {y:.4f} {z:.4f} {radius:.4f} -1\n")


def z_um_to_slice(z_um: float, z_um_per_page: list[float]) -> float:
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


def main(argv: Optional[list[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Export spines as SWC markers in cell frame")
    parser.add_argument("--spines", required=True, help="assigned spines CSV (x_cell, y_cell, z_cell)")
    parser.add_argument("--config", required=True, help="animal config for mpp / z schedule")
    parser.add_argument("--session", default="pre-droplet")
    parser.add_argument("--out", required=True, help="output .swc path")
    parser.add_argument("--radius", type=float, default=2.0, help="marker radius in SWC units")
    args = parser.parse_args(argv[1:] if argv else None)

    cfg_path = Path(args.config).resolve()
    base = cfg_path.parent.parent
    scene = export_scene(load_config(cfg_path), base, args.session)
    mpp_x, mpp_y = scene["cell"]["mpp"]
    z_um_per_page = scene["cell"].get("z_um_per_page", [])

    spines = pd.read_csv(args.spines)
    for col in ("x_cell", "y_cell", "z_cell"):
        if col not in spines.columns:
            raise ValueError(f"spines CSV missing {col}")

    xs_um = spines["x_cell"].tolist()
    ys_um = spines["y_cell"].tolist()
    zs_um = spines["z_cell"].tolist()

    xs_px = [x / mpp_x for x in xs_um]
    ys_px = [y / mpp_y for y in ys_um]
    zs_px = [z_um_to_slice(z, z_um_per_page) for z in zs_um]

    out_um = Path(args.out).with_suffix(".um.swc")
    out_px = Path(args.out)
    if out_px.suffix.lower() != ".swc":
        out_px = out_px.with_suffix(".swc")

    write_marker_swc(out_um, xs_um, ys_um, zs_um, "spine markers: cell_um (x,y,z)", args.radius)
    write_marker_swc(out_px, xs_px, ys_px, zs_px, "spine markers: cell pixels on fov1.tif; z=slice", args.radius)

    print(f"spines   : {len(spines)}")
    print(f"wrote um : {out_um}")
    print(f"wrote px : {out_px}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
