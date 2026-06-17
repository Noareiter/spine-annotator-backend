"""Quality-control overlays for the TreeAnalysis registration + assignment.

Step 11 of the TreeAnalysis pipeline. Registration is only trustworthy if it can
be eyeballed, so this module renders the two visual checks that matter:

1. **Skeleton + spine overlay** -- the whole-cell skeleton (2D projection) with
   the transformed spines on top, coloured by branch order. If spines hug the
   dendrites, the alignment is good.
2. **Branch-point residuals** -- mapped mosaic branch points vs their nearest
   cell branch points (connector segments) plus a residual-distance histogram.

Style follows the lab convention: minimalist, sans-serif, clean blue/red
palette (Nature/Cell-like). Figures are saved, not shown.

CLI:
    python qc_overlay.py --graph cell_graph.csv --spines spines_assigned.csv \\
        --outdir results/<date>/qc
    python qc_overlay.py --mosaic-bp mosaic_bp.csv --cell-bp cell_bp.csv \\
        --transform transform.json --outdir results/<date>/qc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


# Clean palette: blue skeleton, red highlights.
_BLUE = "#2c5f9e"
_RED = "#c0392b"
_GREY = "#9aa0a6"
_FOV_PALETTE = [
    "#7B64B8",  # purple
    "#1F8A65",  # green
    "#E8C030",  # yellow
    "#C85898",  # pink
    "#2E79B5",  # blue
    "#F0A040",  # orange
    "#2A9A8A",  # teal
    "#C06028",  # deep orange
]


def _edges_xy(graph: pd.DataFrame, ax0: str, ax1: str):
    coord = {int(r["n"]): (float(r[ax0]), float(r[ax1])) for _, r in graph.iterrows()}
    segs = []
    for _, r in graph.iterrows():
        p = int(r["parent"])
        if p == -1 or p not in coord:
            continue
        segs.append([coord[p], coord[int(r["n"])]])
    return segs


def plot_skeleton_spines(
    graph: pd.DataFrame,
    spines: pd.DataFrame,
    out_path: Path,
    axes: Tuple[str, str] = ("x_um", "y_um"),
) -> Path:
    """Overlay transformed spines on the skeleton, coloured by FOV."""
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ax0, ax1 = axes
    fig, ax = plt.subplots(figsize=(6, 6))

    segs = _edges_xy(graph, ax0, ax1)
    if segs:
        ax.add_collection(LineCollection(segs, colors=_BLUE, linewidths=0.8, alpha=0.9, zorder=1))

    sx = "x_cell" if "x_cell" in spines.columns else ax0
    sy = "y_cell" if "y_cell" in spines.columns else ax1
    if "fov" in spines.columns:
        fov_vals = sorted(spines["fov"].astype(str).unique().tolist())
        color_map = {fov: _FOV_PALETTE[i % len(_FOV_PALETTE)] for i, fov in enumerate(fov_vals)}
        for fov in fov_vals:
            sub = spines[spines["fov"].astype(str) == fov]
            ax.scatter(
                sub[sx],
                sub[sy],
                s=16,
                c=color_map[fov],
                edgecolors="k",
                linewidths=0.25,
                zorder=2,
                label=fov,
            )
        ax.legend(title="FOV", frameon=False, fontsize=7, title_fontsize=8, loc="upper right")
    else:
        ax.scatter(
            spines[sx],
            spines[sy],
            s=14,
            c=_RED,
            edgecolors="k",
            linewidths=0.25,
            zorder=2,
        )

    ax.set_aspect("equal")
    ax.set_xlabel(f"{ax0} (um)")
    ax.set_ylabel(f"{ax1} (um)")
    ax.set_title("Spines on whole-cell skeleton (colored by FOV)", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_branchpoint_map(
    cell_graph: pd.DataFrame,
    cell_bp: pd.DataFrame,
    mosaic_graph: pd.DataFrame,
    mosaic_bp: pd.DataFrame,
    out_path: Path,
    axes: Tuple[str, str] = ("x_um", "y_um"),
) -> Path:
    """Side-by-side labeled branch-point maps to help pick manual pairs.

    Left = whole-cell skeleton + branch points; right = FOV-mosaic skeleton +
    branch points. Each branch point is annotated with its node id ``n``, which
    is what goes into the ``mosaic_n,cell_n`` pairs CSV. The two clouds live in
    different coordinate frames, so they are drawn in separate panels.
    """
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ax0, ax1 = axes
    fig, (axc, axm) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, graph, bp, title, col in (
        (axc, cell_graph, cell_bp, "Whole-cell (cell_n)", _BLUE),
        (axm, mosaic_graph, mosaic_bp, "FOV mosaic (mosaic_n)", _RED),
    ):
        segs = _edges_xy(graph, ax0, ax1)
        if segs:
            ax.add_collection(LineCollection(segs, colors=_GREY, linewidths=0.7, alpha=0.8, zorder=1))
        ax.scatter(bp[ax0], bp[ax1], s=22, c=col, zorder=2)
        for _, r in bp.iterrows():
            ax.annotate(str(int(r["n"])), (r[ax0], r[ax1]), fontsize=7, color=col,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_xlabel(f"{ax0} (um)")
        ax.set_ylabel(f"{ax1} (um)")
        ax.set_title(title, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_bp_residuals(
    moved_bp: np.ndarray,
    cell_bp: np.ndarray,
    out_path: Path,
    axes: Tuple[int, int] = (0, 1),
    inlier_um: float = 5.0,
) -> Path:
    """Mapped-vs-nearest branch-point connectors + residual histogram."""
    _style()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    a0, a1 = axes
    # Nearest cell BP for each mapped mosaic BP.
    d = np.linalg.norm(moved_bp[:, None, :] - cell_bp[None, :, :], axis=2)
    nn = np.argmin(d, axis=1)
    resid = d[np.arange(len(moved_bp)), nn]

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})

    ax.scatter(cell_bp[:, a0], cell_bp[:, a1], s=18, c=_GREY, label="cell BP", zorder=1)
    ax.scatter(moved_bp[:, a0], moved_bp[:, a1], s=18, c=_RED, label="mapped mosaic BP", zorder=3)
    connectors = [[(moved_bp[i, a0], moved_bp[i, a1]), (cell_bp[nn[i], a0], cell_bp[nn[i], a1])] for i in range(len(moved_bp))]
    ax.add_collection(LineCollection(connectors, colors=_BLUE, linewidths=0.6, alpha=0.7, zorder=2))
    ax.set_aspect("equal")
    ax.set_xlabel(f"axis {a0} (um)")
    ax.set_ylabel(f"axis {a1} (um)")
    ax.set_title("Branch-point correspondence", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    axh.hist(resid, bins=20, color=_BLUE, alpha=0.85)
    axh.axvline(inlier_um, color=_RED, linestyle="--", linewidth=1.0, label=f"inlier {inlier_um} um")
    axh.axvline(float(np.median(resid)), color="k", linewidth=1.0, label=f"median {np.median(resid):.2f}")
    axh.set_xlabel("residual (um)")
    axh.set_ylabel("count")
    axh.set_title("Residual distribution", fontsize=10)
    axh.legend(frameon=False, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Render TreeAnalysis QC overlays.")
    parser.add_argument("--graph", help="annotated cell graph CSV")
    parser.add_argument("--spines", help="assigned spines CSV (in cell frame)")
    parser.add_argument("--mosaic-bp", help="mosaic branch-points CSV")
    parser.add_argument("--cell-bp", help="cell branch-points CSV")
    parser.add_argument("--transform", help="transform JSON (required with --mosaic-bp/--cell-bp)")
    parser.add_argument("--inlier-um", type=float, default=5.0)
    parser.add_argument("--outdir", default=".", help="output directory for figures")
    args = parser.parse_args(argv[1:])

    outdir = Path(args.outdir)
    wrote: List[Path] = []

    if args.graph and args.spines:
        p = plot_skeleton_spines(pd.read_csv(args.graph), pd.read_csv(args.spines), outdir / "skeleton_spines.png")
        wrote.append(p)

    if args.mosaic_bp and args.cell_bp and args.transform:
        import json

        from register_mosaic_to_cell import Similarity3D

        data = json.loads(Path(args.transform).read_text(encoding="utf-8"))
        T = Similarity3D.from_dict(data["transform"] if "transform" in data else data)
        mbp = pd.read_csv(args.mosaic_bp)[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
        cbp = pd.read_csv(args.cell_bp)[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
        p = plot_bp_residuals(T.apply(mbp), cbp, outdir / "bp_residuals.png", inlier_um=args.inlier_um)
        wrote.append(p)

    if not wrote:
        parser.error("provide --graph+--spines and/or --mosaic-bp+--cell-bp+--transform")
    for p in wrote:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
