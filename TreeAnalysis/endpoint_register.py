"""Per-FOV registration from two endpoint correspondences (FOV trace → cell).

Each high-mag FOV SWC has two tips in stage µm. Mark where those tips sit on the
whole-cell SWC (start + end). A rigid XY rotation + translation and mean Z shift
are fit from the pair — enough to fix the ~80 µm XY offset seen with translation-only
manual alignment.

Exports the same ``manual_registration.json`` envelope used by :mod:`manual_transform`.

CLI:
    python endpoint_register.py --pairs config/endpoint_pairs_pre-droplet.json \\
        --out config/manual_registration_pre-droplet.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from manual_transform import to_pipeline_report


def _round_pt(p: Sequence[float], tol: float) -> Tuple[float, float, float]:
    return tuple(round(float(c) / tol) * tol for c in p[:3])


def trace_endpoints(
    segments: List[List[List[float]]],
    tol_um: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """Return the two trace tips farthest apart (stage or cell µm segments)."""
    if not segments:
        raise ValueError("no segments")

    counts: dict[Tuple[float, float, float], int] = {}
    unique: List[List[float]] = []

    def add(pt: Sequence[float]) -> None:
        key = _round_pt(pt, tol_um)
        if key not in counts:
            counts[key] = 0
            unique.append(list(key))
        counts[key] += 1

    for a, b in segments:
        add(a)
        add(b)

    tips = [p for p in unique if counts[_round_pt(p, tol_um)] == 1]
    pool = tips if len(tips) >= 2 else unique
    if len(pool) < 2:
        raise ValueError("could not find two trace endpoints")

    best_i, best_j, best_d = 0, 1, -1.0
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            d = float(np.linalg.norm(np.asarray(pool[i]) - np.asarray(pool[j])))
            if d > best_d:
                best_i, best_j, best_d = i, j, d
    return pool[best_i], pool[best_j]


def rigid_xy_matrix_from_point_pairs(
    src_pts: Sequence[Sequence[float]],
    dst_pts: Sequence[Sequence[float]],
) -> np.ndarray:
    """Rigid XY transform + mean Z shift from N>=2 point correspondences (Kabsch 2D)."""
    src = np.asarray(src_pts, dtype=float)[:, :3]
    dst = np.asarray(dst_pts, dtype=float)[:, :3]
    if src.shape != dst.shape or len(src) < 2:
        raise ValueError("need >=2 matching 3D points")

    mu_s = src[:, :2].mean(axis=0)
    mu_d = dst[:, :2].mean(axis=0)
    sc = src[:, :2] - mu_s
    dc = dst[:, :2] - mu_d
    h00 = float(np.dot(sc[:, 0], dc[:, 0]))
    h01 = float(np.dot(sc[:, 0], dc[:, 1]))
    h10 = float(np.dot(sc[:, 1], dc[:, 0]))
    h11 = float(np.dot(sc[:, 1], dc[:, 1]))
    theta = float(np.arctan2(h10 - h01, h00 + h11))
    c, s = np.cos(theta), np.sin(theta)
    r2 = np.array([[c, -s], [s, c]], dtype=float)
    t2 = mu_d - r2 @ mu_s
    tz = float(np.mean(dst[:, 2] - src[:, 2]))

    m = np.eye(4, dtype=float)
    m[:2, :2] = r2
    m[:3, 3] = [t2[0], t2[1], tz]
    return m


def rigid_xy_matrix_from_two_pairs(
    src_a: Sequence[float],
    src_b: Sequence[float],
    dst_a: Sequence[float],
    dst_b: Sequence[float],
) -> np.ndarray:
    """4×4 column-major affine: rigid rotation in XY + uniform Z shift."""
    sa = np.asarray(src_a[:3], dtype=float)
    sb = np.asarray(src_b[:3], dtype=float)
    da = np.asarray(dst_a[:3], dtype=float)
    db = np.asarray(dst_b[:3], dtype=float)

    vs = sb[:2] - sa[:2]
    vd = db[:2] - da[:2]
    if np.linalg.norm(vs) < 1e-6 or np.linalg.norm(vd) < 1e-6:
        raise ValueError("endpoint pair too short for rotation fit")

    cross = vs[0] * vd[1] - vs[1] * vd[0]
    dot = vs[0] * vd[0] + vs[1] * vd[1]
    angle = float(np.arctan2(cross, dot))
    c, s = np.cos(angle), np.sin(angle)
    r2 = np.array([[c, -s], [s, c]], dtype=float)
    t2 = da[:2] - r2 @ sa[:2]
    tz = float((da[2] - sa[2] + db[2] - sb[2]) / 2.0)

    m = np.eye(4, dtype=float)
    m[:2, :2] = r2
    m[:3, 3] = [t2[0], t2[1], tz]
    return m


def matrix_to_export_entry(m: np.ndarray) -> dict:
    cm = m.flatten(order="F").tolist()
    rm = [m[r, c] for r in range(4) for c in range(4)]
    return {"matrix4_column_major": cm, "matrix4_row_major": rm}


def build_manual_from_endpoint_pairs(
    pairs_doc: dict,
    animal_id: str | None = None,
    session: str | None = None,
) -> dict:
    """Build manual_registration dict from endpoint_pairs JSON."""
    fov_transforms = {}
    for fov_id, entry in pairs_doc.get("fov_pairs", {}).items():
        src_start = entry["fov_start_um"]
        src_end = entry["fov_end_um"]
        dst_start = entry["cell_start_um"]
        dst_end = entry["cell_end_um"]
        m = rigid_xy_matrix_from_two_pairs(src_start, src_end, dst_start, dst_end)
        fov_transforms[fov_id] = matrix_to_export_entry(m)

    return {
        "version": 1,
        "animal_id": animal_id or pairs_doc.get("animal_id"),
        "session": session or pairs_doc.get("session"),
        "frame": "cell_um",
        "source": "manual_register_endpoints.html",
        "exported_at": pairs_doc.get("exported_at"),
        "axis_note": "Transforms map stage_um (FOV) -> cell_um (2-point rigid XY + Z shift)",
        "fov_transforms": fov_transforms,
    }


def main(argv: List[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Build manual_registration.json from endpoint pairs.")
    parser.add_argument("--pairs", required=True, help="endpoint_pairs JSON from HTML tool")
    parser.add_argument("--out", required=True, help="output manual_registration.json")
    args = parser.parse_args(argv[1:])

    pairs_doc = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    manual = build_manual_from_endpoint_pairs(pairs_doc)
    Path(args.out).write_text(json.dumps(manual, indent=2), encoding="utf-8")
    print(f"wrote {len(manual['fov_transforms'])} FOV transforms -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
