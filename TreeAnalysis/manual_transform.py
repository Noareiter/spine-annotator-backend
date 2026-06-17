"""Load and apply manual 3D registration from manual_register_3d.html.

The HTML tool exports ``manual_registration.json`` with one 4×4 transform per FOV,
mapping **stage µm** (FOV trace / spine coordinates) → **cell µm** (whole-cell frame).

Each transform is stored as ``matrix4_row_major`` (16 floats, row-major, homogeneous).

CLI:
    python manual_transform.py --mosaic mosaic.csv --registration manual_registration.json \\
        --out spines_in_cell.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

STAGE_COLS = ("x_um", "y_um", "z_um")
CELL_COLS = ("x_cell", "y_cell", "z_cell")


def _normalize_fov_key(fov: str) -> str:
    s = str(fov).strip()
    if s.isdigit():
        return f"fov{s}"
    if not s.startswith("fov"):
        return f"fov{s}"
    return s


def load_manual_registration(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "fov_transforms" not in data:
        raise ValueError(f"{path}: missing fov_transforms")
    return data


def matrix_from_fov_entry(entry: dict) -> np.ndarray:
    """Build a 4×4 matrix from an export entry."""
    if "matrix4_column_major" in entry:
        return np.asarray(entry["matrix4_column_major"], dtype=float).reshape(4, 4, order="F")
    if "matrix4_row_major" in entry:
        return np.asarray(entry["matrix4_row_major"], dtype=float).reshape(4, 4)
    # Fallback: compose from translation + euler + uniform scale
    t = np.asarray(entry.get("translation_um", [0, 0, 0]), dtype=float)
    rot = np.asarray(entry.get("rotation_euler_deg", [0, 0, 0]), dtype=float) * np.pi / 180.0
    s = float(entry.get("uniform_scale", 1.0))
    cx, cy, cz = np.cos(rot)
    sx, sy, sz = np.sin(rot)
    # XYZ euler
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    m = np.eye(4)
    m[:3, :3] = s * R
    m[:3, 3] = t
    return m


def fov_matrices(manual: dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for fov, entry in manual["fov_transforms"].items():
        out[_normalize_fov_key(fov)] = matrix_from_fov_entry(entry)
    return out


def apply_point(m: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    p = np.array([x, y, z, 1.0])
    q = m @ p
    return q[:3]


def apply_manual_to_mosaic(mosaic: pd.DataFrame, manual: dict) -> pd.DataFrame:
    """Map each mosaic spine to cell µm using its FOV's manual transform."""
    mats = fov_matrices(manual)
    if "fov" not in mosaic.columns:
        raise ValueError("mosaic missing 'fov' column; run stitch_session_fovs first")
    missing = [c for c in STAGE_COLS if c not in mosaic.columns]
    if missing:
        raise ValueError(f"mosaic missing columns {missing}")

    out = mosaic.copy()
    xs, ys, zs = [], [], []
    for _, row in out.iterrows():
        key = _normalize_fov_key(row["fov"])
        if key not in mats:
            raise ValueError(f"no manual transform for {key!r} (have {list(mats.keys())})")
        q = apply_point(mats[key], float(row["x_um"]), float(row["y_um"]), float(row["z_um"]))
        xs.append(q[0])
        ys.append(q[1])
        zs.append(q[2])
    out["x_cell"], out["y_cell"], out["z_cell"] = xs, ys, zs
    return out


def to_pipeline_report(manual: dict) -> dict:
    """Wrap manual registration in the same envelope as register_mosaic_to_cell output."""
    mats = fov_matrices(manual)
    return {
        "method": "manual_3d_html",
        "animal_id": manual.get("animal_id"),
        "session": manual.get("session"),
        "n_fovs": len(mats),
        "fov_transforms": manual.get("fov_transforms"),
        "source": manual.get("exported_at"),
        "transform": {"type": "per_fov_matrix4", "fovs": list(mats.keys())},
    }


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Apply manual 3D registration to a mosaic CSV.")
    parser.add_argument("--mosaic", required=True, help="stitched mosaic CSV")
    parser.add_argument("--registration", required=True, help="manual_registration.json from HTML tool")
    parser.add_argument("--out", help="output CSV with x_cell/y_cell/z_cell")
    args = parser.parse_args(argv[1:])

    mosaic = pd.read_csv(args.mosaic)
    manual = load_manual_registration(args.registration)
    out = apply_manual_to_mosaic(mosaic, manual)
    print(f"spines transformed: {len(out)}")
    print(f"fovs in registration: {list(fov_matrices(manual).keys())}")
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
