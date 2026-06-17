"""Convert a pixel-unit SWC into isotropic micrometres.

SNT/FIJI exports SWC with "All positions and radii in pixels" (confirmed in the
file headers). Before any distance, branch-length, or registration computation
the trace must be put into real, isotropic micrometres:

- lateral: ``x_um = x_px * microns_per_pixel_x`` (and y),
- axial: ``z_um`` via :mod:`correct_swc_z` -- a non-uniform slice->depth
  schedule for the whole-cell stack, or a uniform step for high-mag FOVs.

This keeps the cheap lateral scaling and the non-trivial z logic in one place
(step 4 of the pipeline), so downstream modules always receive ``x_um/y_um/z_um``.

CLI:
    # whole-cell (non-uniform z schedule):
    python calibrate_swc.py "<cell>.swc" --mpp 0.55030 --z-before 4 --z-after 2 --z-cutoff 60
    # high-mag FOV (uniform z step):
    python calibrate_swc.py "<fov>.swc"  --mpp 0.12594 --z-uniform 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from correct_swc_z import ZSchedule, slice_to_depth_um
from io_swc import parse_swc


def calibrate(
    df: pd.DataFrame,
    microns_per_pixel_x: float,
    microns_per_pixel_y: float,
    z_schedule: Optional[ZSchedule] = None,
    z_uniform_step_um: Optional[float] = None,
    z_first_slice: int = 1,
) -> pd.DataFrame:
    """Add ``x_um``, ``y_um``, ``z_um`` columns to a pixel-unit SWC table.

    Exactly one of ``z_schedule`` or ``z_uniform_step_um`` must be provided. The
    original pixel columns are preserved; the radius is also scaled laterally
    into ``radius_um`` (mean of x/y calibration).

    ``z_first_slice`` is the slice index that the first z-plane corresponds to
    (SNT z is 1-based by default).
    """
    if (z_schedule is None) == (z_uniform_step_um is None):
        raise ValueError("provide exactly one of z_schedule or z_uniform_step_um")

    out = df.copy()
    out["x_um"] = out["x"] * microns_per_pixel_x
    out["y_um"] = out["y"] * microns_per_pixel_y
    out["radius_um"] = out["radius"] * 0.5 * (microns_per_pixel_x + microns_per_pixel_y)

    if z_schedule is not None:
        out["z_um"] = slice_to_depth_um(out["z"].to_numpy(dtype=float), z_schedule)
    else:
        out["z_um"] = (out["z"].to_numpy(dtype=float) - z_first_slice) * float(z_uniform_step_um)

    out["z_slice"] = out["z"]
    return out


def _summary(df: pd.DataFrame) -> str:
    def rng(c: str) -> str:
        return f"{df[c].min():.2f} .. {df[c].max():.2f}"

    return "\n".join(
        [
            f"n_nodes      : {len(df)}",
            f"x_um         : {rng('x_um')}",
            f"y_um         : {rng('y_um')}",
            f"z_um         : {rng('z_um')}",
            f"extent_um    : {df['x_um'].max() - df['x_um'].min():.1f} x "
            f"{df['y_um'].max() - df['y_um'].min():.1f} x "
            f"{df['z_um'].max() - df['z_um'].min():.1f}",
        ]
    )


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Calibrate a pixel SWC into micrometres.")
    parser.add_argument("swc")
    parser.add_argument("--mpp", type=float, required=True, help="lateral microns/pixel (isotropic x=y)")
    parser.add_argument("--mpp-y", type=float, default=None, help="microns/pixel for y if anisotropic")
    parser.add_argument("--z-uniform", type=float, default=None, help="uniform z-step (um) per slice")
    parser.add_argument("--z-before", type=float, default=None, help="z-step (um) before cutoff slice")
    parser.add_argument("--z-after", type=float, default=None, help="z-step (um) after cutoff slice")
    parser.add_argument("--z-cutoff", type=float, default=None, help="cutoff slice index")
    parser.add_argument("--z0", type=float, default=0.0, help="depth (um) at first slice")
    parser.add_argument("--out", default=None, help="optional CSV output path")
    args = parser.parse_args(argv[1:])

    schedule = None
    if args.z_uniform is None:
        if None in (args.z_before, args.z_after, args.z_cutoff):
            parser.error("provide --z-uniform OR all of --z-before/--z-after/--z-cutoff")
        schedule = ZSchedule.from_cutoff(args.z_before, args.z_after, args.z_cutoff, z0_um=args.z0)

    df = parse_swc(args.swc)
    cal = calibrate(
        df,
        microns_per_pixel_x=args.mpp,
        microns_per_pixel_y=args.mpp_y if args.mpp_y is not None else args.mpp,
        z_schedule=schedule,
        z_uniform_step_um=args.z_uniform,
    )
    print(f"file: {Path(args.swc).name}")
    print(_summary(cal))
    if args.out:
        cal.to_csv(args.out, index=False)
        print(f"wrote calibrated nodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
