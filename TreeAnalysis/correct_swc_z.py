"""Convert whole-cell SWC z (slice index) to true physical depth.

Step 4 of the TreeAnalysis pipeline. The whole-cell stack is acquired with a
**non-uniform** z-spacing, but SNT/FIJI traces it with the z-voxel set to 1, so
the SWC ``z`` column is effectively a (1-based) slice index, not micrometres.

This module integrates a per-animal spacing schedule to map slice index ->
cumulative depth in micrometres. The schedule is the same information entered
when building the dendrogram: a cutoff slice plus the z-step before and after
it. It generalises to any number of segments.

Model
-----
Slice 1 is the reference depth ``z0_um`` (default 0). Moving from slice ``k`` to
slice ``k+1`` adds the spacing that applies to that interval. With a single
cutoff ``c`` and steps ``step_before`` (intervals up to ``c``) and ``step_after``
(intervals beyond ``c``)::

    depth(slice) = z0 + sum over interval i<slice of spacing(i)

Non-integer slice indices (interpolated SWC nodes) are handled by linear
interpolation within their interval.

CLI:
    python correct_swc_z.py --before 4 --after 2 --cutoff 60   # show a few mappings
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class ZSchedule:
    """Piecewise-constant z-spacing schedule for a whole-cell stack.

    Attributes
    ----------
    breakpoints:
        Slice indices (1-based) at which the spacing changes, ascending. Each
        value is the **last slice** of a segment. The final segment runs to the
        end of the stack and needs no breakpoint.
    steps:
        Spacing (um per slice interval) for each segment. ``len(steps)`` must be
        ``len(breakpoints) + 1``.
    z0_um:
        Physical depth assigned to slice 1 (default 0.0).
    """

    breakpoints: Sequence[float]
    steps: Sequence[float]
    z0_um: float = 0.0

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.breakpoints) + 1:
            raise ValueError(
                f"steps ({len(self.steps)}) must be breakpoints+1 ({len(self.breakpoints) + 1})"
            )
        if list(self.breakpoints) != sorted(self.breakpoints):
            raise ValueError("breakpoints must be ascending")
        if any(s <= 0 for s in self.steps):
            raise ValueError("z-steps must be positive")

    @classmethod
    def from_cutoff(
        cls,
        step_before: float,
        step_after: float,
        cutoff_slice: float,
        z0_um: float = 0.0,
    ) -> "ZSchedule":
        """Convenience constructor for the common two-segment case."""
        return cls(breakpoints=[cutoff_slice], steps=[step_before, step_after], z0_um=z0_um)

    def spacing_for_interval(self, interval_end_slice: float) -> float:
        """Spacing (um) for the interval ending at ``interval_end_slice``."""
        for bp, step in zip(self.breakpoints, self.steps):
            if interval_end_slice <= bp:
                return step
        return self.steps[-1]


def slice_to_depth_um(slice_index: np.ndarray | float, schedule: ZSchedule) -> np.ndarray:
    """Map 1-based slice index/indices to cumulative depth in micrometres.

    Builds the exact depth at every integer slice by cumulative summation, then
    linearly interpolates for fractional slice indices. Vectorised over arrays.
    """
    s = np.atleast_1d(np.asarray(slice_index, dtype=float))
    if s.size == 0:
        return s

    max_slice = int(np.ceil(np.nanmax(s)))
    max_slice = max(max_slice, 1)

    # Depth at each integer slice 1..max_slice.
    slices = np.arange(1, max_slice + 1, dtype=float)
    depths = np.empty_like(slices)
    depths[0] = schedule.z0_um
    for i in range(1, len(slices)):
        # Interval from slice i to i+1 (1-based); attribute it to its end slice.
        depths[i] = depths[i - 1] + schedule.spacing_for_interval(slices[i])

    # Interpolate (handles fractional indices); clamp below slice 1 to z0.
    out = np.interp(s, slices, depths, left=schedule.z0_um, right=depths[-1])
    return out


def apply_to_swc(df, schedule: ZSchedule, z_col: str = "z", out_col: str = "z_um"):
    """Return a copy of an SWC DataFrame with a corrected-depth column added.

    The original slice-index column is preserved as ``<z_col>_slice``.
    """
    out = df.copy()
    out[f"{z_col}_slice"] = out[z_col]
    out[out_col] = slice_to_depth_um(out[z_col].to_numpy(dtype=float), schedule)
    return out


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Map SWC slice index -> true depth (um).")
    parser.add_argument("--before", type=float, required=True, help="z-step (um) before the cutoff slice")
    parser.add_argument("--after", type=float, required=True, help="z-step (um) after the cutoff slice")
    parser.add_argument("--cutoff", type=float, required=True, help="cutoff slice index")
    parser.add_argument("--z0", type=float, default=0.0, help="depth (um) of slice 1 (default 0)")
    parser.add_argument(
        "--slices",
        type=float,
        nargs="*",
        default=[1, 30, 60, 61, 90, 120],
        help="slice indices to preview",
    )
    args = parser.parse_args(argv[1:])

    schedule = ZSchedule.from_cutoff(args.before, args.after, args.cutoff, z0_um=args.z0)
    depths = slice_to_depth_um(np.asarray(args.slices, dtype=float), schedule)
    print(f"schedule: before={args.before} after={args.after} cutoff={args.cutoff} z0={args.z0}")
    for sl, d in zip(args.slices, depths):
        print(f"  slice {sl:>7.2f}  ->  {d:>10.3f} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
