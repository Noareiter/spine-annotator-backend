#!/usr/bin/env python3
"""Backward-compatible launcher (deprecated).

Prefer: spine_pipeline/run.py or `python -m spine_pipeline` from the scripts folder.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    import sys

    _scripts = Path(__file__).resolve().parent.parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    print("INFO: build_gp04_spine_summary.py is deprecated. Use spine_pipeline/run.py instead.")
    runpy.run_path(str(Path(__file__).with_name("build_spine_summary.py")), run_name="__main__")
