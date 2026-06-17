"""Parse an SWC neuron trace into a tidy node table.

Step 3 of the TreeAnalysis pipeline. SWC is the standard single-neuron
morphology format (as exported by SNT / FIJI). Each non-comment line is::

    n  type  x  y  z  radius  parent

where ``n`` is a 1-based sample id and ``parent`` is the id of the preceding
sample (-1 for a root). Coordinates are in whatever units the tracing used; for
our whole-cell traces x/y are micrometres and **z is a slice index** (SNT
z-voxel set to 1), which :mod:`correct_swc_z` later converts to true depth.

This module only reads geometry/topology into a DataFrame. Graph construction
(branch order, path length) lives in :mod:`build_tree_graph`.

CLI:
    python io_swc.py "<path-to>.swc"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

# SWC structure-identifier conventions (column ``type``).
SWC_TYPE_LABELS = {
    -1: "undefined",
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
    5: "custom",
    6: "unspecified_neurite",
    7: "glia_processes",
}

SWC_COLUMNS = ["n", "type", "x", "y", "z", "radius", "parent"]


def parse_swc(swc_path: str | Path) -> pd.DataFrame:
    """Parse an SWC file into a DataFrame with columns :data:`SWC_COLUMNS`.

    Lines beginning with ``#`` are comments and ignored. Whitespace of any width
    separates the seven fields. ``n`` and ``parent`` are integers; the rest are
    floats. A ``type_label`` column is added for readability.

    Raises ``ValueError`` if a data line does not have exactly 7 fields or if
    duplicate sample ids are present.
    """
    swc_path = Path(swc_path)
    rows: List[tuple] = []
    with swc_path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 7:
                raise ValueError(
                    f"{swc_path.name}:{lineno}: expected 7 fields, got {len(parts)}: {line!r}"
                )
            n, type_, x, y, z, radius, parent = parts
            rows.append(
                (int(n), int(float(type_)), float(x), float(y), float(z), float(radius), int(float(parent)))
            )

    if not rows:
        raise ValueError(f"{swc_path}: no SWC data lines found")

    df = pd.DataFrame(rows, columns=SWC_COLUMNS)

    if df["n"].duplicated().any():
        dups = df.loc[df["n"].duplicated(), "n"].tolist()
        raise ValueError(f"{swc_path.name}: duplicate sample ids: {dups[:10]}")

    df["type_label"] = df["type"].map(SWC_TYPE_LABELS).fillna("other")
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    """Lightweight integrity summary used by the CLI and tests."""
    n_roots = int((df["parent"] == -1).sum())
    valid_parents = set(df["n"]) | {-1}
    n_orphans = int((~df["parent"].isin(valid_parents)).sum())
    return {
        "n_nodes": int(len(df)),
        "n_roots": n_roots,
        "n_orphans": n_orphans,
        "n_soma_nodes": int((df["type"] == 1).sum()),
        "type_counts": df["type_label"].value_counts().to_dict(),
        "x_range": (float(df["x"].min()), float(df["x"].max())),
        "y_range": (float(df["y"].min()), float(df["y"].max())),
        "z_range": (float(df["z"].min()), float(df["z"].max())),
        "radius_range": (float(df["radius"].min()), float(df["radius"].max())),
    }


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if len(argv) != 2:
        print(__doc__)
        return 2
    df = parse_swc(argv[1])
    stats = basic_stats(df)
    print(f"file        : {Path(argv[1]).name}")
    for key, val in stats.items():
        print(f"{key:12}: {val}")
    if stats["n_roots"] != 1:
        print(f"NOTE: {stats['n_roots']} roots (parent == -1); expected 1 for a single neuron.")
    if stats["n_orphans"]:
        print(f"WARNING: {stats['n_orphans']} nodes reference a missing parent id.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
