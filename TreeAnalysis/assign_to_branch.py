"""Assign each spine to its dendritic branch and emit structural coordinates.

Step 10 of the TreeAnalysis pipeline -- the payoff. With spines now in the
whole-cell frame (:mod:`apply_transform`) and the annotated tree graph
(:mod:`build_tree_graph`), each spine is snapped to the nearest skeleton
**edge** (parent->child segment). From that projection we read out the spine's
structural context:

- ``seg_id``           : unbranched segment the spine sits on
- ``branch_order``     : centrifugal order of that segment
- ``path_to_soma_um``  : path length from soma to the projected point
- ``dist_to_dendrite_um`` : perpendicular distance spine->skeleton (a QC handle)
- ``nearest_node``     : the child node id of the host edge

This is what lets spine dynamics be analysed by dendritic location.

CLI:
    python assign_to_branch.py --spines spines_in_cell.csv --graph cell_graph.csv \\
        --out spines_assigned.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_SPINE_CELL_COLS = ("x_cell", "y_cell", "z_cell")
_SPINE_STAGE_COLS = ("x_um", "y_um", "z_um")
_GRAPH_COLS = ("x_um", "y_um", "z_um")


def _spine_xyz(spines: pd.DataFrame) -> np.ndarray:
    if all(c in spines.columns for c in _SPINE_CELL_COLS):
        cols = _SPINE_CELL_COLS
    elif all(c in spines.columns for c in _SPINE_STAGE_COLS):
        cols = _SPINE_STAGE_COLS
    else:
        raise ValueError("spines need x_cell/y_cell/z_cell (preferred) or x_um/y_um/z_um")
    return spines[list(cols)].to_numpy(dtype=float)


def build_edges(graph: pd.DataFrame) -> dict:
    """Build edge arrays from an annotated tree graph.

    Returns a dict of parallel arrays keyed by edge (one per non-root node):
    endpoint coords ``A`` (parent) and ``B`` (child), the child id ``child_n``,
    and per-edge ``seg_id`` / ``branch_order`` plus the parent's
    ``path_to_soma`` and the edge length.
    """
    for c in ("n", "parent", *_GRAPH_COLS, "seg_id", "branch_order", "path_to_soma", "len_to_parent"):
        if c not in graph.columns:
            raise ValueError(f"graph missing column {c!r}; run build_tree_graph first")

    coord = {int(r["n"]): np.array([r["x_um"], r["y_um"], r["z_um"]], dtype=float) for _, r in graph.iterrows()}
    path = {int(r["n"]): float(r["path_to_soma"]) for _, r in graph.iterrows()}

    A: List[np.ndarray] = []
    B: List[np.ndarray] = []
    child_n: List[int] = []
    seg_id: List[int] = []
    branch_order: List[int] = []
    parent_path: List[float] = []
    edge_len: List[float] = []
    for _, r in graph.iterrows():
        p = int(r["parent"])
        n = int(r["n"])
        if p == -1 or p not in coord:
            continue
        A.append(coord[p])
        B.append(coord[n])
        child_n.append(n)
        seg_id.append(int(r["seg_id"]))
        branch_order.append(int(r["branch_order"]))
        parent_path.append(path[p])
        edge_len.append(float(r["len_to_parent"]))

    return {
        "A": np.array(A) if A else np.zeros((0, 3)),
        "B": np.array(B) if B else np.zeros((0, 3)),
        "child_n": np.array(child_n, dtype=int),
        "seg_id": np.array(seg_id, dtype=int),
        "branch_order": np.array(branch_order, dtype=int),
        "parent_path": np.array(parent_path, dtype=float),
        "edge_len": np.array(edge_len, dtype=float),
    }


def _project_point(p: np.ndarray, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project point ``p`` onto every segment ``A->B``.

    Returns ``(t, dist)`` arrays: ``t`` clamped to ``[0, 1]`` is the fractional
    position from A (parent) to B (child); ``dist`` is the Euclidean distance.
    """
    AB = B - A
    denom = np.einsum("ij,ij->i", AB, AB)
    denom_safe = np.where(denom > 0, denom, 1.0)
    t = np.einsum("ij,ij->i", p[None, :] - A, AB) / denom_safe
    t = np.clip(t, 0.0, 1.0)
    proj = A + t[:, None] * AB
    dist = np.linalg.norm(p[None, :] - proj, axis=1)
    return t, dist


def assign_spines(spines: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    """Assign each spine to its nearest skeleton edge with structural readouts."""
    edges = build_edges(graph)
    if len(edges["child_n"]) == 0:
        raise ValueError("graph has no edges")

    pts = _spine_xyz(spines)
    out = spines.copy()
    seg_ids: List[int] = []
    orders: List[int] = []
    paths: List[float] = []
    dists: List[float] = []
    nodes: List[int] = []
    for p in pts:
        t, dist = _project_point(p, edges["A"], edges["B"])
        e = int(np.argmin(dist))
        seg_ids.append(int(edges["seg_id"][e]))
        orders.append(int(edges["branch_order"][e]))
        paths.append(float(edges["parent_path"][e] + t[e] * edges["edge_len"][e]))
        dists.append(float(dist[e]))
        nodes.append(int(edges["child_n"][e]))

    out["seg_id"] = seg_ids
    out["branch_order"] = orders
    out["path_to_soma_um"] = paths
    out["dist_to_dendrite_um"] = dists
    out["nearest_node"] = nodes
    return out


def distance_decomposition(spines: pd.DataFrame, graph: pd.DataFrame) -> dict:
    """Split spine→skeleton distance into in-plane (XY) and axial (Z) parts."""
    edges = build_edges(graph)
    pts = _spine_xyz(spines)
    xy_dists: List[float] = []
    z_dists: List[float] = []
    for p in pts:
        t, dist = _project_point(p, edges["A"], edges["B"])
        e = int(np.argmin(dist))
        proj = edges["A"][e] + t[e] * (edges["B"][e] - edges["A"][e])
        xy_dists.append(float(np.linalg.norm(p[:2] - proj[:2])))
        z_dists.append(float(abs(p[2] - proj[2])))
    return {
        "median_dist_xy_um": float(np.median(xy_dists)),
        "median_dist_z_um": float(np.median(z_dists)),
    }


def assignment_summary(assigned: pd.DataFrame, graph: Optional[pd.DataFrame] = None) -> dict:
    summary = {
        "n_spines": int(len(assigned)),
        "median_dist_to_dendrite_um": float(assigned["dist_to_dendrite_um"].median()),
        "max_dist_to_dendrite_um": float(assigned["dist_to_dendrite_um"].max()),
        "branch_order_range": (int(assigned["branch_order"].min()), int(assigned["branch_order"].max())),
        "path_to_soma_range_um": (
            float(assigned["path_to_soma_um"].min()),
            float(assigned["path_to_soma_um"].max()),
        ),
        "n_segments_hit": int(assigned["seg_id"].nunique()),
    }
    if graph is not None:
        summary.update(distance_decomposition(assigned, graph))
    return summary


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Assign spines to dendritic branches.")
    parser.add_argument("--spines", required=True, help="spines in cell frame (x_cell/y_cell/z_cell)")
    parser.add_argument("--graph", required=True, help="annotated cell graph CSV (from build_tree_graph)")
    parser.add_argument("--max-dist-um", type=float, default=None, help="warn if any spine exceeds this distance")
    parser.add_argument("--out", help="optional CSV path for assigned spines")
    args = parser.parse_args(argv[1:])

    spines = pd.read_csv(args.spines)
    graph = pd.read_csv(args.graph)
    assigned = assign_spines(spines, graph)
    summary = assignment_summary(assigned)
    print(f"spines assigned : {summary['n_spines']}")
    for k, v in summary.items():
        if k == "n_spines":
            continue
        print(f"  {k:26}: {v}")
    if args.max_dist_um is not None:
        far = int((assigned["dist_to_dendrite_um"] > args.max_dist_um).sum())
        if far:
            print(f"WARNING: {far} spine(s) farther than {args.max_dist_um} um from any dendrite")
    if args.out:
        assigned.to_csv(args.out, index=False)
        print(f"wrote assigned spines -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
