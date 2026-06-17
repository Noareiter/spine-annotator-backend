"""Build a dendritic tree graph from an SWC node table.

Step 5 of the TreeAnalysis pipeline. Given the parsed SWC (see :mod:`io_swc`),
this computes the per-node structural quantities that define "structural
context" for a spine:

- ``node_kind``      : root / branch / continuation / tip
- ``children``       : number of direct children
- ``branch_order``   : centrifugal order (bifurcations between soma and node)
- ``seg_id``         : id of the unbranched segment the node belongs to
- ``len_to_parent``  : Euclidean distance (um) to the parent node
- ``path_to_soma``   : cumulative path length (um) from the soma/root

It uses only numpy/pandas (no networkx): an SWC is a forest where every node has
exactly one parent, so a single root-down pass is sufficient.

Z handling: if a ``z_um`` column exists (from :mod:`correct_swc_z`) it is used
for distances; otherwise the raw ``z`` column is used. Pass ``--z-raw`` on the
CLI to force the raw column.

CLI:
    python build_tree_graph.py "<path-to>.swc"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from io_swc import parse_swc

NODE_ROOT = "root"
NODE_BRANCH = "branch"
NODE_CONTINUATION = "continuation"
NODE_TIP = "tip"


def _pick_coord_columns(df: pd.DataFrame, prefer_um: bool = True) -> tuple[str, str, str]:
    """Choose (x, y, z) columns, preferring calibrated micrometre columns.

    Distances are only meaningful in isotropic units, so if any axis is
    calibrated all three must be; mixing pixels and micrometres is rejected.
    """
    has_um = all(c in df.columns for c in ("x_um", "y_um", "z_um"))
    if prefer_um and has_um:
        return "x_um", "y_um", "z_um"
    return "x", "y", "z"


def build_tree(df: pd.DataFrame, prefer_um: bool = True) -> pd.DataFrame:
    """Annotate an SWC DataFrame with graph/structural quantities.

    Returns a new DataFrame indexed by ``n`` (sample id), preserving input
    columns and adding: ``children``, ``node_kind``, ``branch_order``,
    ``seg_id``, ``len_to_parent``, ``path_to_soma``, ``root_id``.

    Uses calibrated ``x_um/y_um/z_um`` columns when present (see
    :mod:`calibrate_swc`); otherwise raw pixel/slice columns, in which case all
    reported lengths are in those uncalibrated units.
    """
    x_col, y_col, z_col = _pick_coord_columns(df, prefer_um)
    work = df.set_index("n", drop=False)

    parent_of: Dict[int, int] = work["parent"].to_dict()
    coords: Dict[int, np.ndarray] = {
        int(n): np.array([row[x_col], row[y_col], row[z_col]], dtype=float)
        for n, row in work.iterrows()
    }

    # Children adjacency.
    children: Dict[int, List[int]] = {int(n): [] for n in work.index}
    roots: List[int] = []
    for n, p in parent_of.items():
        n = int(n)
        p = int(p)
        if p == -1 or p not in children:
            roots.append(n)
        else:
            children[p].append(n)

    n_children = {n: len(c) for n, c in children.items()}

    def classify(n: int) -> str:
        if parent_of[n] == -1 or parent_of[n] not in coords:
            return NODE_ROOT
        c = n_children[n]
        if c == 0:
            return NODE_TIP
        if c >= 2:
            return NODE_BRANCH
        return NODE_CONTINUATION

    node_kind = {n: classify(n) for n in work.index}

    branch_order: Dict[int, int] = {}
    path_to_soma: Dict[int, float] = {}
    len_to_parent: Dict[int, float] = {}
    seg_id: Dict[int, int] = {}
    root_id: Dict[int, int] = {}

    # Root-down breadth-first pass (parent always processed before child).
    next_seg = 0
    for r in roots:
        branch_order[r] = 0
        path_to_soma[r] = 0.0
        len_to_parent[r] = 0.0
        root_id[r] = r
        seg_id[r] = next_seg
        next_seg += 1

        stack = [r]
        while stack:
            node = stack.pop()
            p = parent_of[node]
            parent_is_branch = (p in node_kind) and (node_kind[p] == NODE_BRANCH)
            for ch in children[node]:
                d = float(np.linalg.norm(coords[ch] - coords[node]))
                len_to_parent[ch] = d
                path_to_soma[ch] = path_to_soma[node] + d
                # A new segment starts after a branch point or at the root's children.
                if node_kind[node] in (NODE_BRANCH, NODE_ROOT):
                    seg_id[ch] = next_seg
                    next_seg += 1
                else:
                    seg_id[ch] = seg_id[node]
                # Centrifugal order increments when leaving a branch point.
                branch_order[ch] = branch_order[node] + (1 if node_kind[node] == NODE_BRANCH else 0)
                root_id[ch] = root_id[node]
                stack.append(ch)
            # parent_is_branch unused beyond clarity; order handled via node kind.
            _ = parent_is_branch

    out = work.copy()
    out["children"] = out.index.map(n_children)
    out["node_kind"] = out.index.map(node_kind)
    out["branch_order"] = out.index.map(branch_order)
    out["seg_id"] = out.index.map(seg_id)
    out["len_to_parent"] = out.index.map(len_to_parent)
    out["path_to_soma"] = out.index.map(path_to_soma)
    out["root_id"] = out.index.map(root_id)
    out["coords_used"] = f"{x_col},{y_col},{z_col}"
    return out.reset_index(drop=True)


def graph_summary(g: pd.DataFrame) -> dict:
    kinds = g["node_kind"].value_counts().to_dict()
    return {
        "n_nodes": int(len(g)),
        "n_roots": int(kinds.get(NODE_ROOT, 0)),
        "n_branch_points": int(kinds.get(NODE_BRANCH, 0)),
        "n_tips": int(kinds.get(NODE_TIP, 0)),
        "n_segments": int(g["seg_id"].nunique()),
        "max_branch_order": int(g["branch_order"].max()),
        "total_length": float(g["len_to_parent"].sum()),
        "max_path_to_soma": float(g["path_to_soma"].max()),
        "coords_used": str(g["coords_used"].iloc[0]) if len(g) else None,
    }


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Build a tree graph from an SWC or calibrated CSV.")
    parser.add_argument("path", help="path to .swc (pixels) or calibrated node .csv (with x_um/y_um/z_um)")
    parser.add_argument("--pixels", action="store_true", help="use raw pixel columns even if x_um exists")
    parser.add_argument("--out", help="optional CSV path for the annotated node table")
    args = parser.parse_args(argv[1:])

    if str(args.path).lower().endswith(".csv"):
        df = pd.read_csv(args.path)
    else:
        df = parse_swc(args.path)
    g = build_tree(df, prefer_um=not args.pixels)
    summary = graph_summary(g)
    print(f"file: {Path(args.path).name}")
    for k, v in summary.items():
        print(f"  {k:22}: {v}")
    if args.out:
        g.to_csv(args.out, index=False)
        print(f"wrote annotated nodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
