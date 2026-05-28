#!/usr/bin/env python3
"""
Step 2 — dendrite / spine summary pipeline.

Run from your animal respan folder (study data). Outputs go to ./spine_summary/
in the directory you run from (current working directory).

Example:
  cd E:\\...\\GP04\\respan
  python "D:\\learning_project_spines\\code final\\step 2 - dendrite information\\run.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

_STEP2_DIR = Path(__file__).resolve().parent
if str(_STEP2_DIR) not in sys.path:
    sys.path.insert(0, str(_STEP2_DIR))

from spine_pipeline.run import main

if __name__ == "__main__":
    raise SystemExit(main())
