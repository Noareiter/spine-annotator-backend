"""Register the FOV mosaic into the whole-cell frame by branch-point matching.

Step 8 of the TreeAnalysis pipeline -- the geometric heart of the analysis. The
high-mag mosaic and the whole-cell overview are separate acquisitions whose
stage origins are not comparable, so we align them by their shared structural
landmarks: dendritic **branch points** (step 7).

The transform is a 3D **similarity** (uniform scale + rotation + translation),
which absorbs the residual zoom/orientation difference between the two
acquisitions while preserving angles (so branch topology is not distorted).

Strategy
--------
1. ``umeyama`` -- closed-form least-squares similarity from known point pairs.
2. ``--pairs`` -- if you supply matched branch-point ids, fit directly (gold
   standard; recommended for the first animal to validate the convention).
3. Otherwise an automatic path: a local *shape descriptor* (sorted distances to
   each point's nearest neighbours, scale-normalised) proposes candidate
   correspondences, and ``ransac_similarity`` finds the largest consistent
   subset.
4. ``icp_refine`` -- polishes the fit by iterated nearest-neighbour matching of
   the full branch-point clouds.

All maths is numpy-only (brute-force nearest neighbour is fine for the hundreds
of branch points a single neuron has).

CLI:
    python register_mosaic_to_cell.py --mosaic mosaic_bp.csv --cell cell_bp.csv \\
        --out transform.json
    python register_mosaic_to_cell.py --mosaic mosaic_bp.csv --cell cell_bp.csv \\
        --pairs pairs.csv --out transform.json   # pairs: mosaic_n,cell_n
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

XYZ = ("x_um", "y_um", "z_um")


@dataclass
class Similarity3D:
    """Uniform-scale rigid transform: ``y = scale * R @ x + t``."""

    scale: float
    R: np.ndarray  # (3, 3) rotation
    t: np.ndarray  # (3,) translation

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform an ``(N, 3)`` (or ``(3,)``) array of points."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        out = self.scale * (p @ self.R.T) + self.t
        return out

    def to_dict(self) -> dict:
        return {"scale": float(self.scale), "R": self.R.tolist(), "t": self.t.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Similarity3D":
        return cls(scale=float(d["scale"]), R=np.asarray(d["R"], dtype=float), t=np.asarray(d["t"], dtype=float))

    @classmethod
    def identity(cls) -> "Similarity3D":
        return cls(scale=1.0, R=np.eye(3), t=np.zeros(3))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Similarity3D:
    """Least-squares similarity mapping ``src -> dst`` (Umeyama 1991).

    ``src`` and ``dst`` are ``(N, 3)`` corresponding points. Returns the
    transform minimising ``sum ||dst_i - (s R src_i + t)||^2``.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need matching (N>=3, 3) src/dst arrays")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / src.shape[0]
        scale = float((D * np.diag(S)).sum() / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return Similarity3D(scale=scale, R=R, t=t)


def _nn(query: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force nearest neighbour: for each query row, nearest ref row.

    Returns ``(indices, distances)``.
    """
    d = np.linalg.norm(query[:, None, :] - ref[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    return idx, d[np.arange(len(query)), idx]


def _descriptors(points: np.ndarray, k: int = 4) -> np.ndarray:
    """Scale-normalised local descriptor: sorted distances to k neighbours.

    Each row is the k smallest pairwise distances (excluding self), divided by
    their own mean so the descriptor is invariant to the unknown scale factor.
    """
    n = len(points)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    desc = np.zeros((n, k), dtype=float)
    kk = min(k, max(n - 1, 1))
    for i in range(n):
        nearest = np.sort(d[i])[1 : kk + 1]
        pad = np.pad(nearest, (0, k - len(nearest)), constant_values=nearest[-1] if len(nearest) else 1.0)
        m = pad.mean()
        desc[i] = pad / m if m > 0 else pad
    return desc


def propose_matches(src: np.ndarray, dst: np.ndarray, k: int = 4, max_candidates: int = 3) -> List[Tuple[int, int]]:
    """Candidate ``(src_i, dst_j)`` matches from descriptor similarity.

    For each src point, keep up to ``max_candidates`` closest descriptors in dst.
    These are noisy proposals; ``ransac_similarity`` selects the consistent set.
    """
    ds = _descriptors(src, k)
    dd = _descriptors(dst, k)
    cost = np.linalg.norm(ds[:, None, :] - dd[None, :, :], axis=2)
    pairs: List[Tuple[int, int]] = []
    for i in range(len(src)):
        for j in np.argsort(cost[i])[:max_candidates]:
            pairs.append((i, int(j)))
    return pairs


def ransac_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    candidates: List[Tuple[int, int]],
    inlier_um: float = 5.0,
    iters: int = 2000,
    seed: int = 0,
) -> Tuple[Optional[Similarity3D], np.ndarray]:
    """Robustly fit a similarity from noisy candidate matches.

    Samples minimal triplets, fits with :func:`umeyama`, and scores by how many
    candidate matches land within ``inlier_um`` after transform. Returns the
    best transform refit on its inliers and the boolean inlier mask over
    ``candidates``.
    """
    rng = np.random.default_rng(seed)
    cand = np.array(candidates)
    if len(cand) < 3:
        return None, np.zeros(len(cand), dtype=bool)
    src_c = src[cand[:, 0]]
    dst_c = dst[cand[:, 1]]

    best_inliers = np.zeros(len(cand), dtype=bool)
    best_count = 0
    for _ in range(iters):
        pick = rng.choice(len(cand), size=3, replace=False)
        try:
            T = umeyama(src_c[pick], dst_c[pick])
        except (ValueError, np.linalg.LinAlgError):
            continue
        resid = np.linalg.norm(T.apply(src_c) - dst_c, axis=1)
        inliers = resid <= inlier_um
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_count < 3:
        return None, best_inliers
    T = umeyama(src_c[best_inliers], dst_c[best_inliers])
    return T, best_inliers


def icp_refine(
    src: np.ndarray,
    dst: np.ndarray,
    init: Similarity3D,
    max_iters: int = 50,
    inlier_um: float = 5.0,
    tol: float = 1e-4,
) -> Tuple[Similarity3D, float]:
    """Refine an initial transform by iterated nearest-neighbour matching.

    Each iteration maps ``src`` with the current transform, pairs each mapped
    point to its nearest ``dst`` point, drops pairs beyond ``inlier_um``, and
    refits. Returns the refined transform and the final RMS inlier residual.
    """
    T = init
    prev_rms = np.inf
    rms = np.inf
    for _ in range(max_iters):
        moved = T.apply(src)
        idx, dist = _nn(moved, dst)
        mask = dist <= inlier_um
        if mask.sum() < 3:
            break
        T = umeyama(src[mask], dst[idx[mask]])
        rms = float(np.sqrt(np.mean(np.linalg.norm(T.apply(src[mask]) - dst[idx[mask]], axis=1) ** 2)))
        if abs(prev_rms - rms) < tol:
            break
        prev_rms = rms
    return T, rms


def _bp_xyz(df: pd.DataFrame) -> np.ndarray:
    return df[list(XYZ)].to_numpy(dtype=float)


def register(
    mosaic_bp: pd.DataFrame,
    cell_bp: pd.DataFrame,
    pairs: Optional[List[Tuple[int, int]]] = None,
    inlier_um: float = 5.0,
    do_icp: bool = True,
    seed: int = 0,
) -> dict:
    """Register mosaic branch points into the cell frame.

    ``pairs`` is an optional list of ``(mosaic_n, cell_n)`` sample-id matches;
    when given, the transform is fit directly from them (then optionally ICP).
    Returns a result dict with the transform, residuals, and bookkeeping.
    """
    src = _bp_xyz(mosaic_bp)
    dst = _bp_xyz(cell_bp)
    src_n = mosaic_bp["n"].to_numpy()
    dst_n = cell_bp["n"].to_numpy()

    method = ""
    inlier_mask: Optional[np.ndarray] = None
    if pairs:
        s_lookup = {int(n): i for i, n in enumerate(src_n)}
        d_lookup = {int(n): i for i, n in enumerate(dst_n)}
        idx_pairs = [(s_lookup[a], d_lookup[b]) for a, b in pairs if a in s_lookup and b in d_lookup]
        if len(idx_pairs) < 3:
            raise ValueError(f"need >=3 valid pairs, got {len(idx_pairs)}")
        ip = np.array(idx_pairs)
        T = umeyama(src[ip[:, 0]], dst[ip[:, 1]])
        method = f"manual_pairs(n={len(idx_pairs)})"
    else:
        candidates = propose_matches(src, dst)
        T, inlier_mask = ransac_similarity(src, dst, candidates, inlier_um=inlier_um, seed=seed)
        if T is None:
            raise ValueError("automatic matching failed; supply --pairs")
        method = f"ransac(inliers={int(inlier_mask.sum())}/{len(candidates)})"

    rms = float("nan")
    if do_icp:
        T, rms = icp_refine(src, dst, T, inlier_um=inlier_um)
        method += "+icp"

    # Final residual: mapped mosaic BP -> nearest cell BP.
    moved = T.apply(src)
    _, dist = _nn(moved, dst)
    matched = dist <= inlier_um
    return {
        "transform": T.to_dict(),
        "method": method,
        "scale": float(T.scale),
        "n_mosaic_bp": int(len(src)),
        "n_cell_bp": int(len(dst)),
        "n_matched": int(matched.sum()),
        "icp_rms_um": rms,
        "median_residual_um": float(np.median(dist[matched])) if matched.any() else None,
        "max_residual_um": float(dist[matched].max()) if matched.any() else None,
        "inlier_um": inlier_um,
    }


def _load_pairs(path: str) -> List[Tuple[int, int]]:
    df = pd.read_csv(path)
    cols = list(df.columns)
    a = "mosaic_n" if "mosaic_n" in cols else cols[0]
    b = "cell_n" if "cell_n" in cols else cols[1]
    return [(int(r[a]), int(r[b])) for _, r in df.iterrows()]


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Register FOV-mosaic branch points to whole-cell branch points.")
    parser.add_argument("--mosaic", required=True, help="mosaic branch-points CSV (x_um/y_um/z_um, n)")
    parser.add_argument("--cell", required=True, help="whole-cell branch-points CSV (x_um/y_um/z_um, n)")
    parser.add_argument("--pairs", help="optional CSV of known matches: mosaic_n,cell_n")
    parser.add_argument("--inlier-um", type=float, default=5.0, help="inlier distance threshold (um)")
    parser.add_argument("--no-icp", action="store_true", help="skip ICP refinement")
    parser.add_argument("--seed", type=int, default=0, help="RANSAC RNG seed")
    parser.add_argument("--out", help="optional JSON path for the transform + report")
    args = parser.parse_args(argv[1:])

    mosaic_bp = pd.read_csv(args.mosaic)
    cell_bp = pd.read_csv(args.cell)
    pairs = _load_pairs(args.pairs) if args.pairs else None

    result = register(
        mosaic_bp, cell_bp, pairs=pairs, inlier_um=args.inlier_um, do_icp=not args.no_icp, seed=args.seed
    )
    print(f"method            : {result['method']}")
    print(f"scale             : {result['scale']:.4f}")
    print(f"matched BP        : {result['n_matched']} / {result['n_mosaic_bp']} (cell has {result['n_cell_bp']})")
    print(f"icp rms (um)      : {result['icp_rms_um']:.3f}")
    print(f"median residual   : {result['median_residual_um']}")
    print(f"max residual (um) : {result['max_residual_um']}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote transform -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
