"""Map mosaic spines into the whole-cell frame using a fitted transform.

Step 9 of the TreeAnalysis pipeline. Given the similarity transform from
:mod:`register_mosaic_to_cell` (a JSON file) and the stitched mosaic spine
table (stage micrometres, from :mod:`stitch_session_fovs`), this applies the
transform to every spine so its coordinates live in the same frame as the
whole-cell SWC. After this step spines and skeleton are directly comparable, so
:mod:`assign_to_branch` can snap each spine to its dendrite.

The transformed coordinates are written as ``x_cell``/``y_cell``/``z_cell`` and
the original stage coordinates are preserved.

CLI:
    python apply_transform.py --mosaic mosaic.csv --transform transform.json \\
        --out spines_in_cell.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List

import pandas as pd

from manual_transform import apply_manual_to_mosaic, load_manual_registration

if TYPE_CHECKING:
    from register_mosaic_to_cell import Similarity3D

STAGE_COLS = ("x_um", "y_um", "z_um")
CELL_COLS = ("x_cell", "y_cell", "z_cell")


def apply_transform(mosaic: pd.DataFrame, transform: "Similarity3D") -> pd.DataFrame:
    """Add ``x_cell``/``y_cell``/``z_cell`` columns to a stage-mosaic table."""
    missing = [c for c in STAGE_COLS if c not in mosaic.columns]
    if missing:
        raise ValueError(f"mosaic is missing stage columns {missing}; run stitch_session_fovs first")
    out = mosaic.copy()
    moved = transform.apply(out[list(STAGE_COLS)].to_numpy(dtype=float))
    for i, c in enumerate(CELL_COLS):
        out[c] = moved[:, i]
    return out


def _load_transform(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("method") == "manual_3d_html" or "fov_transforms" in data:
        return ("manual", load_manual_registration(path))
    if "transform" in data:
        data = data["transform"]
    from register_mosaic_to_cell import Similarity3D
    return ("similarity", Similarity3D.from_dict(data))


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Apply a similarity transform to mosaic spines.")
    parser.add_argument("--mosaic", required=True, help="stitched mosaic CSV (with x_um/y_um/z_um)")
    parser.add_argument("--transform", help="transform JSON from register_mosaic_to_cell or manual_registration")
    parser.add_argument("--out", help="optional CSV path for spines in cell frame")
    parser.add_argument("--registration", help="alias for --transform (manual_registration.json)")
    args = parser.parse_args(argv[1:])

    transform_path = args.transform or args.registration
    if not transform_path:
        parser.error("provide --transform or --registration")

    mosaic = pd.read_csv(args.mosaic)
    kind, payload = _load_transform(transform_path)
    if kind == "manual":
        out = apply_manual_to_mosaic(mosaic, payload)
        print(f"spines transformed: {len(out)} (manual per-FOV)")
    else:
        out = apply_transform(mosaic, payload)
        print(f"spines transformed: {len(out)}")
        print(f"scale applied     : {payload.scale:.4f}")
    if len(out):
        print(
            "cell extent (um)  : "
            f"{out['x_cell'].max() - out['x_cell'].min():.1f} x "
            f"{out['y_cell'].max() - out['y_cell'].min():.1f} x "
            f"{out['z_cell'].max() - out['z_cell'].min():.1f}"
        )
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote spines in cell frame -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
