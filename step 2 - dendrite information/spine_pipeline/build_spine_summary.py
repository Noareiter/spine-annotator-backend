#!/usr/bin/env python3
"""
Build respan spine registry + per-dendrite spine density tables.

Run from inside a study `respan` folder, or pass --input explicitly.
"""

from __future__ import annotations

from typing import Optional


def main(argv: Optional[list[str]] = None) -> None:
    from spine_pipeline.pipeline import main as _pipeline_main

    _pipeline_main(argv)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _step2 = Path(__file__).resolve().parent.parent
    if str(_step2) not in sys.path:
        sys.path.insert(0, str(_step2))
    main()
