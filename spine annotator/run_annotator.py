#!/usr/bin/env python3
"""Launch the dendritic spine annotator (FastAPI + browser UI)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import uvicorn
    from spine_annotator_backend.app import app

    uvicorn.run(app, host="127.0.0.1", port=8010, reload=False)
