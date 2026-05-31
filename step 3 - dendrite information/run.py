#!/usr/bin/env python3
"""
Step 3 — dendrite / spine summary pipeline.

Run from your animal respan folder (study data). Outputs go to ./spine_summary/
in the directory you run from (current working directory).

Example:
  cd E:\\...\\GP04\\respan
  python "D:\\learning_project_spines\\code final\\step 3 - dendrite information\\run.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

_STEP3_DIR = Path(__file__).resolve().parent
if str(_STEP3_DIR) not in sys.path:
    sys.path.insert(0, str(_STEP3_DIR))

from spine_pipeline.run import main

if __name__ == "__main__":
    raise SystemExit(main())
