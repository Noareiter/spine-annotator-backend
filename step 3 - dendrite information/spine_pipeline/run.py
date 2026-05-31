#!/usr/bin/env python3
"""Universal launcher with easy-to-edit defaults at the top of this file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

# Study data: respan folder (default = current working directory when you run).
DEFAULT_INPUT: Optional[Path] = None
DEFAULT_ANIMAL_ID: Optional[str] = None
DEFAULT_MIN_VALID_FRAC = 0.5
# Output: None -> ./spine_summary/ in the directory from which you run the script.
DEFAULT_OUT_DIR: Optional[Path] = None
DEFAULT_FOVS: Optional[List[int]] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spine summary pipeline: run from respan folder; outputs in ./spine_summary/ (cwd)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Study respan root (default: current working directory).",
    )
    p.add_argument("--animal-id", type=str, default=DEFAULT_ANIMAL_ID)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: <cwd>/spine_summary).",
    )
    p.add_argument("--min-valid-frac", type=float, default=DEFAULT_MIN_VALID_FRAC)
    p.add_argument("--fovs", type=int, nargs="*", default=DEFAULT_FOVS)
    return p.parse_args()


def main() -> int:
    from spine_pipeline.pipeline import run_pipeline

    args = parse_args()
    input_root = (args.input or Path.cwd()).resolve()
    out_dir = (args.out_dir or (Path.cwd() / "spine_summary")).resolve()
    print(f"Study root: {input_root}")
    print(f"Output dir: {out_dir}")
    if not args.fovs:
        print("FOV mode: auto-discovery (results/fov*)")
    run_pipeline(
        input_root,
        animal_id=args.animal_id,
        out_dir=out_dir,
        fovs=args.fovs,
        min_valid_frac=args.min_valid_frac,
    )
    return 0


if __name__ == "__main__":
    import sys

    _step3 = Path(__file__).resolve().parent.parent
    if str(_step3) not in sys.path:
        sys.path.insert(0, str(_step3))
    raise SystemExit(main())
