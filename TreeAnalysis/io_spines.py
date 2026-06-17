"""Load RESPAN ``detected_spines`` coordinates into a tidy table.

Step 2 of the TreeAnalysis pipeline. RESPAN exports one CSV per FOV/timepoint
named ``fov<N>_detected_spines.csv`` (under each session's ``Tables/`` folder).
Each row is one spine; the columns this module cares about are::

    spine_id, x, y, z, dendrite_id, ...

``x`` and ``y`` are **pixel** coordinates in that FOV's image; ``z`` is the
spine's plane (slice index / depth as RESPAN reports it). The many morphology
columns (areas, volumes, intensities) are carried through untouched so nothing
is lost downstream.

This module only reads/normalises geometry + identity. Converting pixels to
stage micrometres needs the matching ``PVScan`` XML and happens later
(:mod:`stitch_session_fovs` via :func:`io_pvscan_xml.fov_pixel_to_stage_um`).

CLI:
    python io_spines.py "<path-to>fov1_detected_spines.csv"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Canonical identity/geometry columns produced by this loader.
SPINE_ID_COL = "spine_id"
COORD_COLS = ("x", "y", "z")
DENDRITE_COL = "dendrite_id"

# Accepted aliases -> canonical name (RESPAN versions differ slightly).
_ID_ALIASES = ("spine_id", "id", "label")
_X_ALIASES = ("x", "x_px", "x_pixel", "centroid_x")
_Y_ALIASES = ("y", "y_px", "y_pixel", "centroid_y")
_Z_ALIASES = ("z", "z_px", "z_slice", "centroid_z")


def _first_present(df: pd.DataFrame, names) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in lower:
            return lower[name]
    return None


def parse_detected_spines(csv_path: str | Path) -> pd.DataFrame:
    """Parse a ``detected_spines`` CSV into a normalised spine table.

    The returned frame is indexed by ``spine_id`` (kept as a column too) and is
    guaranteed to carry numeric ``x``, ``y``, ``z`` columns plus, when present,
    ``dendrite_id`` (as string). All other columns are preserved as-is.

    Raises ``ValueError`` if the spine id or any coordinate column is missing.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    id_col = _first_present(df, _ID_ALIASES)
    x_col = _first_present(df, _X_ALIASES)
    y_col = _first_present(df, _Y_ALIASES)
    z_col = _first_present(df, _Z_ALIASES)
    missing = [n for n, c in (("spine_id", id_col), ("x", x_col), ("y", y_col), ("z", z_col)) if c is None]
    if missing:
        raise ValueError(f"{csv_path.name}: missing required column(s): {missing} (have {list(df.columns)[:8]}...)")

    rename = {id_col: SPINE_ID_COL, x_col: "x", y_col: "y", z_col: "z"}
    out = df.rename(columns=rename)

    out[SPINE_ID_COL] = out[SPINE_ID_COL].astype(str)
    for c in COORD_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[list(COORD_COLS)].isna().any().any():
        bad = int(out[list(COORD_COLS)].isna().any(axis=1).sum())
        raise ValueError(f"{csv_path.name}: {bad} row(s) have non-numeric x/y/z")

    if DENDRITE_COL in out.columns:
        out[DENDRITE_COL] = out[DENDRITE_COL].astype(str)

    if out[SPINE_ID_COL].duplicated().any():
        dups = out.loc[out[SPINE_ID_COL].duplicated(), SPINE_ID_COL].tolist()
        raise ValueError(f"{csv_path.name}: duplicate spine ids: {dups[:10]}")

    return out.set_index(SPINE_ID_COL, drop=False)


def basic_stats(df: pd.DataFrame) -> dict:
    """Lightweight integrity summary used by the CLI and tests."""
    stats = {
        "n_spines": int(len(df)),
        "x_range": (float(df["x"].min()), float(df["x"].max())),
        "y_range": (float(df["y"].min()), float(df["y"].max())),
        "z_range": (float(df["z"].min()), float(df["z"].max())),
    }
    if DENDRITE_COL in df.columns:
        stats["n_dendrites"] = int(df[DENDRITE_COL].nunique())
    return stats


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if len(argv) != 2:
        print(__doc__)
        return 2
    df = parse_detected_spines(argv[1])
    stats = basic_stats(df)
    print(f"file        : {Path(argv[1]).name}")
    for key, val in stats.items():
        print(f"{key:12}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
