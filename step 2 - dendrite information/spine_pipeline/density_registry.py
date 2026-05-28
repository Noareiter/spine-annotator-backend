from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .common import (
    ARTIFACT_STATUS,
    BASELINE_TIMEPOINT,
    EXPORT_MATCHED_FILE,
    N_EFF_LANDMARK_STATUSES,
    N_EFF_STATUSES,
    NOT_RELEVANT_STATUSES,
    STATUS_MATCHED,
    STATUS_UNMATCHED,
    FovData,
    TP_ORDER,
    TIMEPOINTS,
    UnionFind,
)
from .io_and_qc import linked_dendrite_ids_at_tp


def spine_counts_for_density(fd: FovData, tp: str, dendrite_id: str) -> Dict[str, int]:
    df = fd.spines[tp]
    sub = df[df["dendrite_id"].astype(str) == str(dendrite_id)]
    counts = defaultdict(int)
    counts["n_detected"] = int(len(sub))
    for sid in sub.index.astype(str):
        st = fd.spine_status.get((tp, sid), STATUS_UNMATCHED)
        counts[f"n_{st}"] = counts.get(f"n_{st}", 0) + 1
        if st not in {ARTIFACT_STATUS, STATUS_UNMATCHED, *NOT_RELEVANT_STATUSES}:
            counts["n_counted"] += 1
        if st in N_EFF_STATUSES:
            counts["n_eff"] += 1
    for k in ("n_counted", "n_eff", "n_artifact", "n_unmatched"):
        counts[k] = counts.get(k, 0)
    return counts


def _spine_xyz_um(row: pd.Series) -> Optional[np.ndarray]:
    cols = ("x", "y", "z")
    if not all(c in row.index for c in cols):
        return None
    vals = [pd.to_numeric(row[c], errors="coerce") for c in cols]
    if any(not np.isfinite(v) for v in vals):
        return None
    return np.array(vals, dtype=float)


def _spine_axis_position_um(row: pd.Series, sub: pd.DataFrame, L_d: float) -> float:
    if L_d <= 0:
        L_d = 1.0
    if "geodesic_dist_to_soma" in row.index:
        g = pd.to_numeric(row["geodesic_dist_to_soma"], errors="coerce")
        if np.isfinite(g) and g >= 0:
            return float(min(L_d, g))
    xyz = _spine_xyz_um(row)
    if xyz is None:
        return 0.0
    pts = [_spine_xyz_um(sub.loc[sid]) for sid in sub.index.astype(str)]
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return 0.0
    pts_arr = np.stack(pts)
    centroid = pts_arr.mean(axis=0)
    X = pts_arr - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    axis = vh[0]
    proj = X @ axis
    p_this = float((xyz - centroid) @ axis)
    p_min, p_max = float(proj.min()), float(proj.max())
    if p_max > p_min:
        rel = (p_this - p_min) / (p_max - p_min)
        return float(np.clip(rel * L_d, 0.0, L_d))
    return 0.0


def _landmark_boundaries_um(fd: FovData, tp: str, sub: pd.DataFrame, L_d: float) -> Tuple[List[float], int]:
    matched_pos = []
    for sid in sub.index.astype(str):
        if fd.spine_status.get((tp, sid), STATUS_UNMATCHED) != STATUS_MATCHED:
            continue
        matched_pos.append(_spine_axis_position_um(sub.loc[sid], sub, L_d))
    matched_pos = sorted(set(matched_pos))
    n_landmarks = len(matched_pos)
    if n_landmarks == 0:
        return [0.0, L_d], 0
    boundaries = [0.0] + matched_pos + [L_d]
    out = [0.0]
    for b in boundaries[1:]:
        b = float(np.clip(b, 0.0, L_d))
        if b > out[-1]:
            out.append(b)
    if out[-1] < L_d:
        out.append(L_d)
    return out, n_landmarks


def _segment_index_for_position(pos: float, boundaries: List[float]) -> int:
    n_seg = len(boundaries) - 1
    if n_seg <= 0:
        return 0
    pos = float(np.clip(pos, boundaries[0], boundaries[-1]))
    for i in range(n_seg):
        if i < n_seg - 1:
            if boundaries[i] <= pos < boundaries[i + 1]:
                return i
        elif boundaries[i] <= pos <= boundaries[i + 1]:
            return i
    return n_seg - 1


def compute_landmark_metrics(fd: FovData, tp: str, dendrite_id: str, L_d: float, min_valid_frac: float) -> Dict[str, float]:
    nan_out = {
        "min_valid_frac": min_valid_frac,
        "n_matched_landmarks": 0,
        "n_segments_total": 0,
        "n_segments_invalid": 0,
        "n_segments_valid": 0,
        "frac_length_valid_landmark": float("nan"),
        "landmark_qc_pass": False,
        "L_eff_landmark_um": float("nan"),
        "n_eff_landmark": float("nan"),
        "n_not_relevant_spines": 0,
    }
    if not np.isfinite(L_d) or L_d <= 0:
        return nan_out
    sub = fd.spines[tp]
    sub = sub[sub["dendrite_id"].astype(str) == str(dendrite_id)]
    boundaries, n_landmarks = _landmark_boundaries_um(fd, tp, sub, L_d)
    n_segments = max(0, len(boundaries) - 1)
    seg_lengths = [boundaries[i + 1] - boundaries[i] for i in range(n_segments)]
    spines_by_seg = [[] for _ in range(n_segments)]
    n_bad = 0
    for sid in sub.index.astype(str):
        st = fd.spine_status.get((tp, sid), STATUS_UNMATCHED)
        if st in NOT_RELEVANT_STATUSES:
            n_bad += 1
        pos = _spine_axis_position_um(sub.loc[sid], sub, L_d)
        spines_by_seg[_segment_index_for_position(pos, boundaries)].append(sid)
    # Segment disqualification: uncertainty only (ignored / not_in_focus). Artifacts do NOT invalidate.
    invalid = set()
    for i, sids in enumerate(spines_by_seg):
        if any(fd.spine_status.get((tp, sid), STATUS_UNMATCHED) in NOT_RELEVANT_STATUSES for sid in sids):
            invalid.add(i)
    valid_seg_idxs = [i for i in range(n_segments) if i not in invalid]
    L_eff = float(sum(seg_lengths[i] for i in valid_seg_idxs))
    frac_valid = L_eff / L_d if L_d > 0 else float("nan")
    qc_pass = bool(np.isfinite(frac_valid) and frac_valid >= min_valid_frac)
    if qc_pass:
        n_eff = 0
        for i in valid_seg_idxs:
            for sid in spines_by_seg[i]:
                if fd.spine_status.get((tp, sid), STATUS_UNMATCHED) in N_EFF_LANDMARK_STATUSES:
                    n_eff += 1
        L_out, n_out = L_eff, float(n_eff)
    else:
        L_out, n_out = float("nan"), float("nan")
    return {
        "min_valid_frac": min_valid_frac,
        "n_matched_landmarks": int(n_landmarks),
        "n_segments_total": int(n_segments),
        "n_segments_invalid": int(len(invalid)),
        "n_segments_valid": int(len(valid_seg_idxs)),
        "frac_length_valid_landmark": float(frac_valid) if np.isfinite(frac_valid) else float("nan"),
        "landmark_qc_pass": qc_pass,
        "L_eff_landmark_um": L_out,
        "n_eff_landmark": n_out,
        "n_not_relevant_spines": int(n_bad),
    }


def build_dendrite_density(fd: FovData, animal_id: str, min_valid_frac: float) -> pd.DataFrame:
    rows = []
    for tp in TP_ORDER:
        dsum = fd.dendrite_summary[tp]
        for did in sorted(linked_dendrite_ids_at_tp(fd, tp)):
            link_id = fd.link_id_by_dendrite.get((tp, did), "")
            length = float(pd.to_numeric(dsum.loc[did, "dendrite_length"], errors="coerce")) if did in dsum.index else float("nan")
            counts = spine_counts_for_density(fd, tp, did)
            lm = compute_landmark_metrics(fd, tp, did, length, min_valid_frac)
            rho = counts["n_counted"] / length if np.isfinite(length) and length > 0 else float("nan")
            rho_eff = float("nan")
            if lm["landmark_qc_pass"] and np.isfinite(lm["L_eff_landmark_um"]) and lm["L_eff_landmark_um"] > 0:
                rho_eff = float(lm["n_eff_landmark"]) / float(lm["L_eff_landmark_um"])
            rows.append(
                {
                    "animal_id": animal_id,
                    "fov": fd.fov,
                    "timepoint": tp,
                    "timepoint_order": TIMEPOINTS[tp]["order"],
                    "link_id": link_id,
                    "dendrite_id": did,
                    "dendrite_length_um": length,
                    **counts,
                    **lm,
                    "rho_spines_per_um": rho,
                    "rho_eff_spines_per_um": rho_eff,
                }
            )
    return pd.DataFrame(rows)


def _landmark_qc_fail_reason(row: pd.Series) -> str:
    if bool(row.get("landmark_qc_pass")):
        return ""
    length = pd.to_numeric(row.get("dendrite_length_um"), errors="coerce")
    if not np.isfinite(length) or length <= 0:
        return "invalid_dendrite_length"
    frac = pd.to_numeric(row.get("frac_length_valid_landmark"), errors="coerce")
    min_frac = pd.to_numeric(row.get("min_valid_frac"), errors="coerce")
    if not np.isfinite(frac):
        return "landmark_metrics_unavailable"
    if np.isfinite(min_frac) and frac < min_frac:
        return "valid_length_below_min_valid_frac"
    return "landmark_qc_fail"


def build_dendrite_landmark_qc_report(ddf: pd.DataFrame) -> pd.DataFrame:
    """
    Per dendrite×timepoint landmark QC report with FOV volume fractions.

    A dendrite is rejected when landmark_qc_pass is False (typically
    frac_length_valid_landmark < min_valid_frac). Rejected volume uses full
    dendrite_length_um; FOV fractions sum length over linked dendrites per fov×timepoint.
    """
    if ddf.empty:
        return pd.DataFrame()

    detail_cols = [
        "animal_id",
        "fov",
        "timepoint",
        "timepoint_order",
        "link_id",
        "dendrite_id",
        "dendrite_length_um",
        "landmark_qc_pass",
        "frac_length_valid_landmark",
        "min_valid_frac",
        "n_matched_landmarks",
        "n_segments_total",
        "n_segments_invalid",
        "n_segments_valid",
        "L_eff_landmark_um",
        "n_eff_landmark",
        "rho_eff_spines_per_um",
    ]
    out = ddf[[c for c in detail_cols if c in ddf.columns]].copy()
    length = pd.to_numeric(out["dendrite_length_um"], errors="coerce")
    passed = out["landmark_qc_pass"].astype(bool)
    out["rejected_volume_um"] = np.where(~passed, length.fillna(0.0), 0.0)
    out["qc_fail_reason"] = out.apply(_landmark_qc_fail_reason, axis=1)

    keys_tp = ["animal_id", "fov", "timepoint"]
    out["fov_tp_total_volume_um"] = out.groupby(keys_tp)["dendrite_length_um"].transform("sum")
    out["fov_tp_rejected_volume_um"] = out.groupby(keys_tp)["rejected_volume_um"].transform("sum")
    denom = out["fov_tp_total_volume_um"].replace(0, np.nan)
    out["fov_tp_rejected_volume_frac"] = out["fov_tp_rejected_volume_um"] / denom
    out["fov_tp_n_dendrites_total"] = out.groupby(keys_tp)["dendrite_id"].transform("count")
    out["fov_tp_n_dendrites_rejected"] = out.groupby(keys_tp)["landmark_qc_pass"].transform(
        lambda s: int((~s.astype(bool)).sum())
    )

    keys_fov = ["animal_id", "fov"]
    out["fov_total_volume_um"] = out.groupby(keys_fov)["dendrite_length_um"].transform("sum")
    out["fov_rejected_volume_um"] = out.groupby(keys_fov)["rejected_volume_um"].transform("sum")
    fov_denom = out["fov_total_volume_um"].replace(0, np.nan)
    out["fov_rejected_volume_frac"] = out["fov_rejected_volume_um"] / fov_denom
    out["fov_n_dendrites_total"] = out.groupby(keys_fov)["dendrite_id"].transform("count")
    out["fov_n_dendrites_rejected"] = out.groupby(keys_fov)["landmark_qc_pass"].transform(
        lambda s: int((~s.astype(bool)).sum())
    )

    sort_cols = ["fov", "timepoint_order" if "timepoint_order" in out.columns else "timepoint", "landmark_qc_pass", "dendrite_id"]
    out = out.sort_values(sort_cols, ascending=[True, True, False, True])
    return out.reset_index(drop=True)


def _sum_dendrite_lengths_um(dsum: pd.DataFrame, dendrite_ids: Set[str]) -> float:
    total = 0.0
    for did in dendrite_ids:
        if did not in dsum.index:
            continue
        length = float(pd.to_numeric(dsum.loc[did, "dendrite_length"], errors="coerce"))
        if np.isfinite(length) and length > 0:
            total += length
    return total


def collect_fov_tp_coverage_records(fd: FovData, ddf: pd.DataFrame, animal_id: str) -> List[dict]:
    """Per FOV×timepoint: how many / how much dendrite volume is outside effective analysis."""
    records: List[dict] = []
    for tp in TP_ORDER:
        if tp not in fd.dendrite_summary:
            continue
        dsum = fd.dendrite_summary[tp]
        if dsum.empty:
            continue
        all_ids = set(dsum.index.astype(str))
        excluded = set(fd.excluded_dendrites.get(tp, set())) & all_ids
        linked = set(linked_dendrite_ids_at_tp(fd, tp))
        sub = ddf[ddf["timepoint"] == tp] if not ddf.empty else pd.DataFrame()
        in_analysis = set(
            sub.loc[sub["landmark_qc_pass"].astype(bool), "dendrite_id"].astype(str).tolist()
        ) if not sub.empty and "landmark_qc_pass" in sub.columns else set()
        qc_fail = linked - in_analysis

        n_detected = len(all_ids)
        n_excluded = len(excluded)
        n_linked = len(linked)
        n_in_analysis = len(in_analysis)
        n_qc_fail = len(qc_fail)
        n_not_in = n_detected - n_in_analysis

        vol_all = _sum_dendrite_lengths_um(dsum, all_ids)
        vol_excluded = _sum_dendrite_lengths_um(dsum, excluded)
        vol_in = _sum_dendrite_lengths_um(dsum, in_analysis)
        vol_qc_fail = _sum_dendrite_lengths_um(dsum, qc_fail)
        vol_not_in = max(0.0, vol_all - vol_in)

        pct_count_not = 100.0 * n_not_in / n_detected if n_detected else float("nan")
        pct_vol_not = 100.0 * vol_not_in / vol_all if vol_all > 0 else float("nan")

        records.append(
            {
                "animal_id": animal_id,
                "fov": fd.fov,
                "timepoint": tp,
                "timepoint_order": TIMEPOINTS[tp]["order"],
                "n_detected": n_detected,
                "n_excluded_annotator": n_excluded,
                "n_linked": n_linked,
                "n_qc_fail": n_qc_fail,
                "n_in_analysis": n_in_analysis,
                "n_not_in_analysis": n_not_in,
                "pct_count_not_in_analysis": pct_count_not,
                "pct_count_in_analysis": 100.0 - pct_count_not if np.isfinite(pct_count_not) else float("nan"),
                "vol_all_um": vol_all,
                "vol_excluded_um": vol_excluded,
                "vol_qc_fail_um": vol_qc_fail,
                "vol_in_analysis_um": vol_in,
                "vol_not_in_analysis_um": vol_not_in,
                "pct_vol_not_in_analysis": pct_vol_not,
                "pct_vol_in_analysis": 100.0 - pct_vol_not if np.isfinite(pct_vol_not) else float("nan"),
            }
        )
    return records


def format_analysis_coverage_text(
    records: List[dict],
    *,
    animal_id: str,
    input_root: str,
    min_valid_frac: float,
) -> str:
    lines = [
        f"Analysis coverage summary - {animal_id}",
        f"min_valid_frac = {min_valid_frac}",
        f"Study root: {input_root}",
        "",
        "Definitions:",
        "  - All dendrites: every row in dendrite_summary at that timepoint (RESPAN).",
        "  - In analysis: linked across timepoints and passed landmark QC (used for rho_eff).",
        "  - Not in analysis: excluded in annotator and/or failed landmark QC.",
        "",
        "Per FOV x timepoint:",
        "  Not in analysis (count % / volume %) | breakdown: annotator | QC",
        "=" * 72,
        "",
    ]
    for r in sorted(records, key=lambda x: (x["fov"], x["timepoint_order"])):
        lines.extend(
            [
                f"FOV {r['fov']} - {r['timepoint']}",
                (
                    f"  Not in analysis: {r['n_not_in_analysis']}/{r['n_detected']} dendrites "
                    f"({r['pct_count_not_in_analysis']:.1f}%)  |  "
                    f"{r['vol_not_in_analysis_um']:.1f}/{r['vol_all_um']:.1f} um "
                    f"({r['pct_vol_not_in_analysis']:.1f}%)"
                ),
                (
                    f"    Excluded (annotator): {r['n_excluded_annotator']} dendrites, "
                    f"{r['vol_excluded_um']:.1f} um"
                ),
                f"    Failed (landmark QC): {r['n_qc_fail']} dendrites, {r['vol_qc_fail_um']:.1f} um",
                (
                    f"  In analysis: {r['n_in_analysis']}/{r['n_detected']} dendrites "
                    f"({r['pct_count_in_analysis']:.1f}%)  |  "
                    f"{r['vol_in_analysis_um']:.1f} um ({r['pct_vol_in_analysis']:.1f}%)"
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def add_density_normalization(ddf: pd.DataFrame) -> pd.DataFrame:
    out = ddf.copy()
    for tp in TP_ORDER:
        mask = out["timepoint"] == tp
        m = np.nanmean(out.loc[mask, "rho_spines_per_um"].to_numpy(dtype=float))
        out.loc[mask, "rho_fold_vs_animal_tp_mean"] = out.loc[mask, "rho_spines_per_um"] / m if m and np.isfinite(m) else np.nan
    out["rho_fold_vs_fov_tp_mean"] = np.nan
    for (fov, tp), grp in out.groupby(["fov", "timepoint"]):
        m = np.nanmean(grp["rho_spines_per_um"].to_numpy(dtype=float))
        out.loc[grp.index, "rho_fold_vs_fov_tp_mean"] = out.loc[grp.index, "rho_spines_per_um"] / m if m and np.isfinite(m) else np.nan
    pre_mean = np.nanmean(
        out.loc[out["timepoint"] == BASELINE_TIMEPOINT, "rho_spines_per_um"].to_numpy(dtype=float)
    )
    out["rho_fold_vs_animal_pre_mean"] = out["rho_spines_per_um"] / pre_mean if pre_mean and np.isfinite(pre_mean) else np.nan
    out["rho_eff_fold_vs_fov_tp_mean"] = np.nan
    for (fov, tp), grp in out.groupby(["fov", "timepoint"]):
        m = np.nanmean(grp["rho_eff_spines_per_um"].to_numpy(dtype=float))
        out.loc[grp.index, "rho_eff_fold_vs_fov_tp_mean"] = out.loc[grp.index, "rho_eff_spines_per_um"] / m if m and np.isfinite(m) else np.nan
    pre_eff = np.nanmean(
        out.loc[out["timepoint"] == BASELINE_TIMEPOINT, "rho_eff_spines_per_um"].to_numpy(dtype=float)
    )
    out["rho_eff_fold_vs_animal_pre_mean"] = out["rho_eff_spines_per_um"] / pre_eff if pre_eff and np.isfinite(pre_eff) else np.nan
    return out


def build_fov_density(ddf: pd.DataFrame, animal_id: str) -> pd.DataFrame:
    rows = []
    for (fov, tp), grp in ddf.groupby(["fov", "timepoint"]):
        length = grp["dendrite_length_um"].sum(skipna=True)
        n_counted = grp["n_counted"].sum()
        rho = n_counted / length if length and length > 0 else float("nan")
        qc = grp[grp["landmark_qc_pass"] == True]
        length_eff = qc["L_eff_landmark_um"].sum(skipna=True) if "L_eff_landmark_um" in qc.columns else float("nan")
        n_eff_l = pd.to_numeric(qc["n_eff_landmark"], errors="coerce").sum(skipna=True) if "n_eff_landmark" in qc.columns else 0.0
        rho_eff = n_eff_l / length_eff if length_eff and length_eff > 0 and np.isfinite(n_eff_l) else float("nan")
        rows.append(
            {
                "animal_id": animal_id,
                "fov": int(fov),
                "timepoint": tp,
                "timepoint_order": TIMEPOINTS[tp]["order"],
                "n_linked_dendrites": int(len(grp)),
                "n_linked_dendrites_qc_pass": int(len(qc)),
                "dendrite_length_um_linked_sum": float(length),
                "L_eff_landmark_um_linked_sum": float(length_eff),
                "n_counted": int(n_counted),
                "n_eff_landmark": float(n_eff_l),
                "n_artifact": int(grp["n_artifact"].sum()),
                "n_unmatched": int(grp["n_unmatched"].sum()),
                "rho_spines_per_um": float(rho),
                "rho_eff_spines_per_um": float(rho_eff),
            }
        )
    fdf = pd.DataFrame(rows)
    for tp in TP_ORDER:
        mask = fdf["timepoint"] == tp
        m = np.nanmean(fdf.loc[mask, "rho_spines_per_um"].to_numpy(dtype=float))
        fdf.loc[mask, "rho_fold_vs_animal_tp_mean"] = fdf.loc[mask, "rho_spines_per_um"] / m if m and np.isfinite(m) else np.nan
        me = np.nanmean(fdf.loc[mask, "rho_eff_spines_per_um"].to_numpy(dtype=float))
        fdf.loc[mask, "rho_eff_fold_vs_animal_tp_mean"] = fdf.loc[mask, "rho_eff_spines_per_um"] / me if me and np.isfinite(me) else np.nan
    pre_mean = np.nanmean(
        fdf.loc[fdf["timepoint"] == BASELINE_TIMEPOINT, "rho_spines_per_um"].to_numpy(dtype=float)
    )
    fdf["rho_fold_vs_animal_pre_mean"] = fdf["rho_spines_per_um"] / pre_mean if pre_mean and np.isfinite(pre_mean) else np.nan
    pre_eff = np.nanmean(
        fdf.loc[fdf["timepoint"] == BASELINE_TIMEPOINT, "rho_eff_spines_per_um"].to_numpy(dtype=float)
    )
    fdf["rho_eff_fold_vs_animal_pre_mean"] = fdf["rho_eff_spines_per_um"] / pre_eff if pre_eff and np.isfinite(pre_eff) else np.nan
    return fdf


def build_animal_density(fdf: pd.DataFrame, animal_id: str) -> pd.DataFrame:
    rows = []
    for tp, grp in fdf.groupby("timepoint"):
        v = grp["rho_spines_per_um"].to_numpy(dtype=float)
        n = int(np.sum(np.isfinite(v)))
        sem = float(np.nanstd(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ve = grp["rho_eff_spines_per_um"].to_numpy(dtype=float)
        ne = int(np.sum(np.isfinite(ve)))
        rec = {
            "animal_id": animal_id,
            "timepoint": tp,
            "timepoint_order": TIMEPOINTS[tp]["order"],
            "n_fovs": n,
            "mean_rho_spines_per_um": float(np.nanmean(v)),
            "sem_rho_spines_per_um": sem,
            "mean_rho_fold_vs_animal_tp_mean": float(np.nanmean(grp["rho_fold_vs_animal_tp_mean"])),
            "mean_rho_fold_vs_animal_pre_mean": float(np.nanmean(grp["rho_fold_vs_animal_pre_mean"])),
            "mean_rho_eff_spines_per_um": float(np.nanmean(ve)),
            "sem_rho_eff_spines_per_um": float(np.nanstd(ve, ddof=1) / np.sqrt(ne)) if ne > 1 else 0.0,
            "mean_rho_eff_fold_vs_animal_pre_mean": float(np.nanmean(grp["rho_eff_fold_vs_animal_pre_mean"])),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def build_spine_lineages(fd: FovData) -> Tuple[UnionFind, List[dict]]:
    uf = UnionFind()
    qc: List[dict] = []
    for bundle in fd.exports:
        d = bundle.export_dir
        if not (d / EXPORT_MATCHED_FILE).is_file():
            continue
        mdf = pd.read_csv(d / EXPORT_MATCHED_FILE)
        for _, row in mdf.iterrows():
            t1_id = str(row.get("t1_spine_id", ""))
            t2_id = str(row.get("t2_spine_id", ""))
            if t1_id and t2_id:
                uf.union((bundle.t1_tp, t1_id), (bundle.t2_tp, t2_id))
    for tp, df in fd.spines.items():
        for sid in df.index.astype(str):
            uf._add((tp, sid))
    return uf, qc


def build_registry_wide(fd: FovData, animal_id: str, spine_uf: UnionFind) -> pd.DataFrame:
    comps = spine_uf.components()
    ref_tp = next(tp for tp in TP_ORDER if tp in fd.spines)
    feature_cols = [c for c in fd.spines[ref_tp].columns if c not in {"id"}]
    rows = []
    idx = 0
    for _, nodes in sorted(comps.items(), key=lambda x: -len(x[1])):
        idx += 1
        lineage_id = f"{animal_id}_fov{fd.fov}_L{idx:05d}"
        row: dict = {"animal_id": animal_id, "fov": fd.fov, "lineage_id": lineage_id}
        seen_tps = []
        for tp in TP_ORDER:
            ids_tp = [sid for t, sid in nodes if t == tp]
            sid = ids_tp[0] if ids_tp else ""
            row[f"id_{tp}"] = sid
            row[f"status_{tp}"] = fd.spine_status.get((tp, sid), "absent") if sid else "absent"
            if sid and sid in fd.spines[tp].index:
                seen_tps.append(tp)
                srow = fd.spines[tp].loc[sid]
                for col in feature_cols:
                    row[f"{tp}_{col}"] = srow[col]
            else:
                for col in feature_cols:
                    row[f"{tp}_{col}"] = np.nan
        row["first_seen_tp"] = min(seen_tps, key=lambda t: TIMEPOINTS[t]["order"]) if seen_tps else ""
        row["last_seen_tp"] = max(seen_tps, key=lambda t: TIMEPOINTS[t]["order"]) if seen_tps else ""
        row["n_timepoints_seen"] = len(seen_tps)
        rows.append(row)
    return pd.DataFrame(rows)


def build_registry_long(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return wide
    id_cols = [f"id_{tp}" for tp in TP_ORDER]
    status_cols = [f"status_{tp}" for tp in TP_ORDER]
    base_cols = ["animal_id", "fov", "lineage_id", "first_seen_tp", "last_seen_tp", "n_timepoints_seen"]
    long_rows = []
    for _, w in wide.iterrows():
        for tp in TP_ORDER:
            sid = w.get(f"id_{tp}", "")
            if not sid or (isinstance(sid, float) and np.isnan(sid)):
                continue
            rec = {c: w[c] for c in base_cols if c in w}
            rec["timepoint"] = tp
            rec["spine_id"] = sid
            rec["status"] = w.get(f"status_{tp}", "")
            prefix = f"{tp}_"
            for col in wide.columns:
                if col.startswith(prefix) and col not in id_cols and col not in status_cols:
                    rec[col[len(prefix):]] = w[col]
            long_rows.append(rec)
    return pd.DataFrame(long_rows)
