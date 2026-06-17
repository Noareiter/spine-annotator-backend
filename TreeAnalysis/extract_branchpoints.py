"""Extract branch-point coordinates from a tree graph.

Step 7 of the TreeAnalysis pipeline. Branch points (bifurcations) are the most
reliable landmarks shared between the whole-cell skeleton and the high-mag FOV
mosaic, so registration (step 8) aligns the two clouds by matching them.

A branch point is any node with >= 2 children (``node_kind == "branch"`` from
:mod:`build_tree_graph`). This module returns their coordinates plus context
(branch order, path-to-soma, child count) that the matcher can use to prune
implausible correspondences.

Works on any annotated node table: the whole-cell graph *or* a mosaic graph, as
long as it has been through :func:`build_tree_graph.build_tree`.

CLI:
    python extract_branchpoints.py "<calibrated_nodes>.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from build_tree_graph import NODE_BRANCH, build_tree
from io_swc import parse_swc

BP_COORD_COLS = ("x_um", "y_um", "z_um")


def extract_branchpoints(graph: pd.DataFrame) -> pd.DataFrame:
    """Return the branch-point rows of an annotated tree graph.

    Requires the columns added by :func:`build_tree_graph.build_tree`. Chooses
    calibrated ``x_um/y_um/z_um`` when present, else raw ``x/y/z`` (reported in
    the ``bp_coords`` column). The result is sorted by ``branch_order`` then
    ``path_to_soma`` so the ordering is deterministic across runs.
    """
    if "node_kind" not in graph.columns:
        raise ValueError("input is not an annotated graph; run build_tree first")

    has_um = all(c in graph.columns for c in BP_COORD_COLS)
    cx, cy, cz = BP_COORD_COLS if has_um else ("x", "y", "z")

    bps = graph[graph["node_kind"] == NODE_BRANCH].copy()
    out = pd.DataFrame(
        {
            "n": bps["n"].to_numpy(),
            "x_um": bps[cx].to_numpy(dtype=float),
            "y_um": bps[cy].to_numpy(dtype=float),
            "z_um": bps[cz].to_numpy(dtype=float),
            "children": bps["children"].to_numpy(),
            "branch_order": bps["branch_order"].to_numpy(),
            "path_to_soma": bps["path_to_soma"].to_numpy(dtype=float),
        }
    )
    out["bp_coords"] = f"{cx},{cy},{cz}"
    return out.sort_values(["branch_order", "path_to_soma"]).reset_index(drop=True)


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Extract branch points from an SWC or annotated CSV.")
    parser.add_argument("path", help=".swc, or a node CSV (annotated, or calibrated with x_um/y_um/z_um)")
    parser.add_argument("--pixels", action="store_true", help="use raw pixel columns even if x_um exists")
    parser.add_argument("--out", help="optional CSV path for branch points")
    args = parser.parse_args(argv[1:])

    if str(args.path).lower().endswith(".csv"):
        df = pd.read_csv(args.path)
    else:
        df = parse_swc(args.path)

    graph = df if "node_kind" in df.columns else build_tree(df, prefer_um=not args.pixels)
    bps = extract_branchpoints(graph)
    print(f"file          : {Path(args.path).name}")
    print(f"branch points : {len(bps)}")
    print(f"coords used   : {bps['bp_coords'].iloc[0] if len(bps) else 'n/a'}")
    if len(bps):
        print(f"order range   : {int(bps['branch_order'].min())} .. {int(bps['branch_order'].max())}")
    if args.out:
        bps.to_csv(args.out, index=False)
        print(f"wrote branch points -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
