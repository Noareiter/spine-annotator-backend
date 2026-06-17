"""Stitch one session's high-mag FOVs into a single stage-coordinate mosaic.

Step 6 of the TreeAnalysis pipeline. Within one imaging session the 5-6 high-mag
spine FOVs are acquired without moving the sample origin, so each FOV's
microscope **stage** coordinates (read from its ``PVScan`` XML) are directly
comparable. This module converts every spine's pixel position into stage
micrometres and concatenates the FOVs into one mosaic point cloud.

Pixel -> stage conversion uses :func:`io_pvscan_xml.fov_pixel_to_stage_um`:
``positionCurrent`` is the FOV centre, so a spine offset from centre (in pixels)
is scaled by ``micronsPerPixel`` and added to the stage centre, with optional
per-axis sign flips (``sign_x`` / ``sign_y``) for the microscope's convention.

Z handling: a spine's ``z`` is its plane within the FOV stack. Its absolute
stage depth is the frame's absolute Z (from the XML, summed device offsets) for
that plane. We map the spine z (slice index, 1-based) to the nearest frame.

Overlap dedup: FOVs can overlap, so the same physical spine may appear twice.
Points from different FOVs closer than ``dedup_radius_um`` in stage space are
collapsed to the first occurrence (kept row), with the dropped ids recorded.

CLI:
    python stitch_session_fovs.py --xml f1.xml --spines f1_detected_spines.csv \\
        --xml f2.xml --spines f2_detected_spines.csv --out mosaic.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from io_pvscan_xml import PVScanMeta, fov_pixel_to_stage_um, parse_pvscan
from io_spines import parse_detected_spines


def _nearest_frame_index(meta: PVScanMeta, spine_z: float) -> int:
    """Map a spine z (1-based slice index) to the nearest XML frame index."""
    if not meta.frames:
        return 1
    indices = np.array([fr.index for fr in meta.frames], dtype=float)
    # Spine z is 1-based; frame.index is typically 1-based too. Snap to nearest.
    return int(indices[int(np.argmin(np.abs(indices - float(spine_z))))])


def fov_spines_to_stage(
    spines: pd.DataFrame,
    meta: PVScanMeta,
    fov_label: str,
    sign_x: int = 1,
    sign_y: int = 1,
) -> pd.DataFrame:
    """Add ``x_um``, ``y_um``, ``z_um`` (stage micrometres) to a spine table."""
    out = spines.copy()
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for _, row in out.iterrows():
        frame = _nearest_frame_index(meta, float(row["z"]))
        sx, sy, sz = fov_pixel_to_stage_um(
            meta, float(row["x"]), float(row["y"]), frame, sign_x=sign_x, sign_y=sign_y
        )
        xs.append(sx)
        ys.append(sy)
        zs.append(sz)
    out["x_um"] = xs
    out["y_um"] = ys
    out["z_um"] = zs
    out["fov"] = fov_label
    out["fov_spine_id"] = out["fov"].astype(str) + ":" + out["spine_id"].astype(str)
    return out


def _dedup_overlap(df: pd.DataFrame, radius_um: float) -> Tuple[pd.DataFrame, List[str]]:
    """Drop near-duplicate spines that come from *different* FOVs.

    Greedy: iterate in order, keep a point unless it is within ``radius_um`` of
    an already-kept point from another FOV. Same-FOV points are never merged
    (RESPAN already deduplicates within a FOV).
    """
    if radius_um <= 0 or df.empty:
        return df, []
    coords = df[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
    fovs = df["fov"].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    kept_idx: List[int] = []
    dropped: List[str] = []
    for i in range(len(df)):
        merged = False
        for j in kept_idx:
            if fovs[i] == fovs[j]:
                continue
            if float(np.linalg.norm(coords[i] - coords[j])) <= radius_um:
                merged = True
                break
        if merged:
            keep[i] = False
            dropped.append(str(df.iloc[i]["fov_spine_id"]))
        else:
            kept_idx.append(i)
    return df.loc[keep].reset_index(drop=True), dropped


def stitch_session(
    fovs: Sequence[Tuple[str, str]],
    sign_x: int = 1,
    sign_y: int = 1,
    dedup_radius_um: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Stitch a list of ``(xml_path, spines_csv_path)`` into one mosaic table.

    Returns ``(mosaic_df, dropped_ids)``. Each FOV is labelled from its spine
    CSV filename stem so provenance is preserved in ``fov`` / ``fov_spine_id``.
    """
    parts: List[pd.DataFrame] = []
    for xml_path, spines_path in fovs:
        meta = parse_pvscan(xml_path)
        spines = parse_detected_spines(spines_path)
        label = Path(spines_path).stem.replace("_detected_spines", "")
        parts.append(fov_spines_to_stage(spines, meta, label, sign_x=sign_x, sign_y=sign_y))
    if not parts:
        return pd.DataFrame(), []
    mosaic = pd.concat(parts, ignore_index=True)
    return _dedup_overlap(mosaic, dedup_radius_um)


def _pair_args(xmls: Optional[Sequence[str]], spines: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    xmls = list(xmls or [])
    spines = list(spines or [])
    if len(xmls) != len(spines):
        raise SystemExit(f"--xml ({len(xmls)}) and --spines ({len(spines)}) counts must match")
    return list(zip(xmls, spines))


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Stitch one session's FOV spines into a stage mosaic.")
    parser.add_argument("--xml", action="append", help="PVScan XML for a FOV (repeat per FOV)")
    parser.add_argument("--spines", action="append", help="detected_spines CSV for a FOV (repeat per FOV)")
    parser.add_argument("--sign-x", type=int, default=1, choices=(1, -1))
    parser.add_argument("--sign-y", type=int, default=1, choices=(1, -1))
    parser.add_argument("--dedup-um", type=float, default=0.0, help="overlap dedup radius (um); 0 disables")
    parser.add_argument("--out", help="optional CSV path for the mosaic")
    args = parser.parse_args(argv[1:])

    pairs = _pair_args(args.xml, args.spines)
    if not pairs:
        parser.error("provide at least one --xml/--spines pair")

    mosaic, dropped = stitch_session(
        pairs, sign_x=args.sign_x, sign_y=args.sign_y, dedup_radius_um=args.dedup_um
    )
    print(f"fovs           : {len(pairs)}")
    print(f"spines (mosaic): {len(mosaic)}")
    print(f"dropped overlap: {len(dropped)}")
    if len(mosaic):
        print(
            "extent_um      : "
            f"{mosaic['x_um'].max() - mosaic['x_um'].min():.1f} x "
            f"{mosaic['y_um'].max() - mosaic['y_um'].min():.1f} x "
            f"{mosaic['z_um'].max() - mosaic['z_um'].min():.1f}"
        )
    if args.out:
        mosaic.to_csv(args.out, index=False)
        print(f"wrote mosaic -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
