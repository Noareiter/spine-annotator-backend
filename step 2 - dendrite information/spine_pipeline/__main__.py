"""Allow: python -m spine_pipeline (from the scripts directory)."""

from __future__ import annotations

from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
