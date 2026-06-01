#!/usr/bin/env python3
"""
Infer non-baseline pairwise spine classifications via baseline bridging.

For each follow-up timepoint T, manual exports exist for Baseline <-> T.
This script derives T_A <-> T_B exports (e.g. mid-droplet vs end-lever) using:

  - Matched: present at T_A and T_B (both matched to the same baseline spine)
  - Lost:    present at T_A, lost at T_B
  - New:     lost at T_A, present at T_B
  - Unresolved: new at T_A (no baseline bridge), or ignored/removed in either export

Input layout (case-insensitive RESPAN segment):
  .../IMAGING/<ANIMAL_ID>/respan/results/fovN/<comparison>/<timestamp>_spine_annotator_export/

Output:
  .../fovN/<T_A - T_B>/<timestamp>_inferred_spine_annotator_export/
      input_files/   (TIFF stacks copied from baseline comparison logs)
      matched.csv, new.csv, lost.csv, unresolved_manual_review.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

# --- Configuration (timepoint folder names on disk) ---
BASELINE_TP = "pre-droplet"

TP_ORDER: List[str] = [
    "pre-droplet",
    "mid-droplet",
    "end-droplet",
    "end-lever",
    "return to droplet",
]

TP_ORDER_INDEX: Dict[str, int] = {tp: i for i, tp in enumerate(TP_ORDER)}

NON_BASELINE_TPS: Tuple[str, ...] = tuple(tp for tp in TP_ORDER if tp != BASELINE_TP)

# Comparison folder names for baseline-linked exports (must match annotator / Step 3).
BASELINE_COMPARISON_MAP: Dict[str, str] = {
    "mid-droplet": "pre-mid droplet",
    "end-droplet": "pre-end droplet",
    "end-lever": "pre droplet-end lever",
    "return to droplet": "pre droplet - return to droplet",
}

COMPARISON_BY_PAIR: Dict[Tuple[str, str], str] = {
    ("end-droplet", "return to droplet"): "end droplet - return to droplet",
    ("end-droplet", "end-lever"): "end droplet - end lever",
}

DERIVED_EXPORT_MARKERS = ("inferred", "synthetic", "registry_derived", "transitive")
INFERRED_EXPORT_SUFFIX = "_inferred_spine_annotator_export"
LOG_CANDIDATES = ("matching_activity_log.txt", "matching_activity.log")


@dataclass
class BaselineExportData:
    """One Baseline <-> follow-up timepoint annotator export."""

    comparison: str
    follow_up_tp: str
    export_dir: Path
    matched: Dict[str, str] = field(default_factory=dict)  # baseline_id -> follow_up_id
    lost_baseline: Set[str] = field(default_factory=set)
    new_follow_up: Set[str] = field(default_factory=set)
    ignored_baseline: Set[str] = field(default_factory=set)
    ignored_follow_up: Set[str] = field(default_factory=set)
    removed_baseline: Set[str] = field(default_factory=set)
    removed_follow_up: Set[str] = field(default_factory=set)
    t1_tiff: Optional[Path] = None
    t2_tiff: Optional[Path] = None


@dataclass
class InferredPairTables:
    matched: List[dict] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    lost: List[str] = field(default_factory=list)
    unresolved: List[dict] = field(default_factory=list)


def _canonical_spine_id(value: object) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    s = str(value).strip()
    if re.fullmatch(r"\d+\.0+", s):
        return str(int(float(s)))
    return s


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    return not s or s.lower() in {"nan", "none", "nat"}


def _read_id_column(path: Path, *preferred_cols: str) -> List[str]:
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    for col in preferred_cols:
        if col in df.columns:
            return [_canonical_spine_id(x) for x in df[col].dropna() if not _is_blank(x)]
    if len(df.columns) == 1:
        col = df.columns[0]
        return [_canonical_spine_id(x) for x in df[col].dropna() if not _is_blank(x)]
    return []


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def comparison_folder_name(t1_tp: str, t2_tp: str) -> str:
    if TP_ORDER_INDEX[t1_tp] > TP_ORDER_INDEX[t2_tp]:
        t1_tp, t2_tp = t2_tp, t1_tp
    return COMPARISON_BY_PAIR.get((t1_tp, t2_tp), f"{t1_tp} - {t2_tp}")


def _is_derived_export_dir(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in DERIVED_EXPORT_MARKERS)


def resolve_results_root(imaging_root: Path, animal_id: str) -> Path:
    """Locate .../IMAGING/<animal>/respan/results (case-insensitive)."""
    animal_dir = imaging_root / animal_id
    if not animal_dir.is_dir():
        raise FileNotFoundError(f"Animal folder not found: {animal_dir}")

    for child in animal_dir.iterdir():
        if child.is_dir() and child.name.lower() == "respan":
            for sub in child.iterdir():
                if sub.is_dir() and sub.name.lower() == "results":
                    return sub.resolve()
    raise FileNotFoundError(
        f"Could not find respan/results under {animal_dir}. "
        "Expected: <imaging_root>/<animal_id>/respan/results"
    )


def discover_fovs(results_root: Path) -> List[int]:
    fovs: List[int] = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"fov(\d+)", d.name, flags=re.IGNORECASE)
        if m:
            fovs.append(int(m.group(1)))
    return fovs


def list_manual_export_dirs(comp_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in comp_dir.glob("*spine_annotator_export")
        if p.is_dir() and not _is_derived_export_dir(p)
    )


def latest_manual_export_dir(comp_dir: Path) -> Optional[Path]:
    exports = list_manual_export_dirs(comp_dir)
    return exports[-1] if exports else None


def _extract_json_objects(text: str) -> List[dict]:
    """Pull JSON objects from annotator activity logs (--- separated blocks)."""
    objects: List[dict] = []
    for block in re.split(r"\n---\n", text):
        block = block.strip()
        if not block:
            continue
        # Drop leading timestamp line if present.
        lines = block.splitlines()
        json_start = 0
        for i, line in enumerate(lines):
            if line.lstrip().startswith("{"):
                json_start = i
                break
        payload = "\n".join(lines[json_start:]).strip()
        if not payload.startswith("{"):
            continue
        try:
            objects.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return objects


def _paths_from_log_dict(obj: dict) -> Tuple[Optional[Path], Optional[Path]]:
    for key in ("selected_files", "files", "paths"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            obj = {**obj, **nested}
    t1 = obj.get("t1_tiff_path")
    t2 = obj.get("t2_tiff_path")
    p1 = Path(str(t1)).expanduser() if not _is_blank(t1) else None
    p2 = Path(str(t2)).expanduser() if not _is_blank(t2) else None
    return (
        p1.resolve() if p1 is not None and p1.is_file() else None,
        p2.resolve() if p2 is not None and p2.is_file() else None,
    )


def parse_tiff_paths_from_export(export_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Read T1/T2 TIFF paths from matching log, then metadata.json."""
    for log_name in LOG_CANDIDATES:
        log_path = export_dir / log_name
        if not log_path.is_file() or log_path.stat().st_size == 0:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        t1_path: Optional[Path] = None
        t2_path: Optional[Path] = None
        for obj in _extract_json_objects(text):
            p1, p2 = _paths_from_log_dict(obj)
            if p1 is not None:
                t1_path = p1
            if p2 is not None:
                t2_path = p2
        if t1_path or t2_path:
            return t1_path, t2_path

    meta_path = export_dir / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        p1, p2 = _paths_from_log_dict(meta)
        return p1, p2

    return None, None


def load_baseline_export(comp_dir: Path, follow_up_tp: str) -> Optional[BaselineExportData]:
    export_dir = latest_manual_export_dir(comp_dir)
    if export_dir is None:
        return None

    data = BaselineExportData(
        comparison=comp_dir.name,
        follow_up_tp=follow_up_tp,
        export_dir=export_dir,
    )

    matched_path = export_dir / "matched.csv"
    if matched_path.is_file():
        mdf = pd.read_csv(matched_path)
        for _, row in mdf.iterrows():
            b_id = _canonical_spine_id(row.get("t1_spine_id", ""))
            fu_id = _canonical_spine_id(row.get("t2_spine_id", ""))
            if b_id and fu_id:
                data.matched[b_id] = fu_id

    data.lost_baseline = set(_read_id_column(export_dir / "lost.csv", "t1_spine_id", "spine_id"))
    data.new_follow_up = set(_read_id_column(export_dir / "new.csv", "t2_spine_id", "spine_id"))
    data.ignored_baseline = set(_read_id_column(export_dir / "ignored_t1.csv", "t1_spine_id", "spine_id"))
    data.ignored_follow_up = set(
        _read_id_column(export_dir / "ignored_t2.csv", "t2_spine_id", "spine_id")
    )
    data.removed_baseline = set(_read_id_column(export_dir / "removed_t1.csv", "t1_spine_id", "spine_id"))
    data.removed_follow_up = set(
        _read_id_column(export_dir / "removed_t2.csv", "t2_spine_id", "spine_id")
    )

    t1_tiff, t2_tiff = parse_tiff_paths_from_export(export_dir)
    data.t1_tiff = t1_tiff
    data.t2_tiff = t2_tiff
    return data


def _baseline_status(data: BaselineExportData, baseline_id: str) -> str:
    """How baseline spine `baseline_id` relates to the follow-up timepoint."""
    if baseline_id in data.matched:
        return "matched"
    if baseline_id in data.lost_baseline:
        return "lost"
    return "absent"


def _follow_up_has_qc_issue(data: BaselineExportData, baseline_id: str, fu_id: str) -> Optional[str]:
    if baseline_id in data.ignored_baseline or baseline_id in data.removed_baseline:
        return "baseline_ignored_or_removed"
    if fu_id in data.ignored_follow_up or fu_id in data.removed_follow_up:
        return "follow_up_ignored_or_removed"
    return None


def infer_pair_via_baseline(
    export_a: BaselineExportData,
    export_b: BaselineExportData,
    *,
    t_a: str,
    t_b: str,
) -> InferredPairTables:
    """
    Infer T_A (earlier) vs T_B (later) from two Baseline <-> T exports.

    T_A is the earlier chronological timepoint; CSV columns follow annotator convention
    (t1 = T_A, t2 = T_B).
    """
    tables = InferredPairTables()
    seen_unresolved_fu: Set[str] = set()

    all_baseline_ids: Set[str] = set()
    all_baseline_ids |= set(export_a.matched) | export_a.lost_baseline
    all_baseline_ids |= set(export_b.matched) | export_b.lost_baseline
    all_baseline_ids |= export_a.ignored_baseline | export_a.removed_baseline
    all_baseline_ids |= export_b.ignored_baseline | export_b.removed_baseline

    for b_id in sorted(all_baseline_ids, key=lambda x: (len(x), x)):
        st_a = _baseline_status(export_a, b_id)
        st_b = _baseline_status(export_b, b_id)
        id_a = export_a.matched.get(b_id, "")
        id_b = export_b.matched.get(b_id, "")

        qc_a = _follow_up_has_qc_issue(export_a, b_id, id_a) if id_a else (
            "baseline_ignored_or_removed" if b_id in export_a.ignored_baseline | export_a.removed_baseline else None
        )
        qc_b = _follow_up_has_qc_issue(export_b, b_id, id_b) if id_b else (
            "baseline_ignored_or_removed" if b_id in export_b.ignored_baseline | export_b.removed_baseline else None
        )
        if qc_a or qc_b:
            tables.unresolved.append(
                {
                    "baseline_spine_id": b_id,
                    "t1_spine_id": id_a,
                    "t2_spine_id": id_b,
                    "reason": qc_a or qc_b,
                    "detail": f"{t_a} vs {t_b}",
                }
            )
            continue

        if st_a == "matched" and st_b == "matched" and id_a and id_b:
            tables.matched.append(
                {
                    "t1_spine_id": id_a,
                    "t2_spine_id": id_b,
                    "baseline_spine_id": b_id,
                    "source": "baseline_bridge",
                }
            )
        elif st_a == "matched" and st_b == "lost" and id_a:
            tables.lost.append(id_a)
        elif st_a == "lost" and st_b == "matched" and id_b:
            tables.new.append(id_b)
        elif st_a == "lost" and st_b == "lost":
            pass  # absent at both follow-ups — no row
        elif st_a == "absent" and st_b == "absent":
            pass
        else:
            tables.unresolved.append(
                {
                    "baseline_spine_id": b_id,
                    "t1_spine_id": id_a,
                    "t2_spine_id": id_b,
                    "reason": f"ambiguous_{st_a}_{st_b}",
                    "detail": f"{t_a} vs {t_b}",
                }
            )

    # Spines new at T_A cannot be bridged through baseline.
    for fu_id in sorted(export_a.new_follow_up, key=lambda x: (len(x), x)):
        if fu_id in seen_unresolved_fu:
            continue
        seen_unresolved_fu.add(fu_id)
        tables.unresolved.append(
            {
                "baseline_spine_id": "",
                "t1_spine_id": fu_id,
                "t2_spine_id": "",
                "reason": "new_at_t1_no_baseline_bridge",
                "detail": t_a,
            }
        )

    # Spines new at T_B without baseline (not reachable via bridge).
    for fu_id in sorted(export_b.new_follow_up, key=lambda x: (len(x), x)):
        if fu_id in export_a.new_follow_up:
            continue  # already flagged if also new at T_A
        tables.unresolved.append(
            {
                "baseline_spine_id": "",
                "t1_spine_id": "",
                "t2_spine_id": fu_id,
                "reason": "new_at_t2_no_baseline_bridge",
                "detail": t_b,
            }
        )

    # Follow-up IDs flagged as ignored/removed without appearing in matched rows.
    for fu_id in sorted(export_a.ignored_follow_up | export_a.removed_follow_up):
        if fu_id not in export_a.matched.values():
            tables.unresolved.append(
                {
                    "baseline_spine_id": "",
                    "t1_spine_id": fu_id,
                    "t2_spine_id": "",
                    "reason": "t1_ignored_or_removed",
                    "detail": t_a,
                }
            )
    for fu_id in sorted(export_b.ignored_follow_up | export_b.removed_follow_up):
        if fu_id not in export_b.matched.values():
            tables.unresolved.append(
                {
                    "baseline_spine_id": "",
                    "t1_spine_id": "",
                    "t2_spine_id": fu_id,
                    "reason": "t2_ignored_or_removed",
                    "detail": t_b,
                }
            )

    return tables


def _dedupe_sorted(ids: Iterable[str]) -> List[str]:
    return sorted({x for x in (_canonical_spine_id(i) for i in ids) if x})


def _dedupe_matched(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["t1_spine_id", "t2_spine_id", "baseline_spine_id", "source"])
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["t1_spine_id", "t2_spine_id"], keep="first").sort_values(
        ["t1_spine_id", "t2_spine_id"]
    )


def _dedupe_unresolved(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["baseline_spine_id", "t1_spine_id", "t2_spine_id", "reason", "detail"]
        )
    df = pd.DataFrame(rows)
    return df.drop_duplicates(
        subset=["baseline_spine_id", "t1_spine_id", "t2_spine_id", "reason"], keep="first"
    ).sort_values(["reason", "t1_spine_id", "t2_spine_id"])


def copy_tiff_to_input_files(
    src: Path,
    dest_dir: Path,
    *,
    role: str,
    tp: str,
    fov: int,
) -> Path:
    suffix = src.suffix if src.suffix else ".tif"
    safe_tp = re.sub(r"[^\w\-]+", "_", tp)
    dest_name = f"{role}_{safe_tp}_fov{fov}{suffix}"
    dest = dest_dir / dest_name
    if dest.exists():
        dest.unlink()
    shutil.copy2(src, dest)
    return dest.resolve()


def stage_input_files(
    export_dir: Path,
    *,
    fov: int,
    t_a: str,
    t_b: str,
    export_a: BaselineExportData,
    export_b: BaselineExportData,
    symlink: bool,
) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    input_dir = export_dir / "input_files"
    input_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    t_a_src = export_a.t2_tiff
    t_b_src = export_b.t2_tiff
    if t_a_src is None or not t_a_src.is_file():
        warnings.append(f"missing TIFF for {t_a} (from {export_a.comparison})")
    if t_b_src is None or not t_b_src.is_file():
        warnings.append(f"missing TIFF for {t_b} (from {export_b.comparison})")

    t_a_dest: Optional[Path] = None
    t_b_dest: Optional[Path] = None
    if t_a_src is not None and t_a_src.is_file():
        if symlink:
            safe_a = re.sub(r"[^\w\-]+", "_", t_a)
            dest = input_dir / f"t1_{safe_a}_fov{fov}{t_a_src.suffix or '.tif'}"
            if dest.exists():
                dest.unlink()
            dest.symlink_to(t_a_src)
            t_a_dest = dest.resolve()
        else:
            t_a_dest = copy_tiff_to_input_files(t_a_src, input_dir, role="t1", tp=t_a, fov=fov)
    if t_b_src is not None and t_b_src.is_file():
        if symlink:
            safe_b = re.sub(r"[^\w\-]+", "_", t_b)
            dest = input_dir / f"t2_{safe_b}_fov{fov}{t_b_src.suffix or '.tif'}"
            if dest.exists():
                dest.unlink()
            dest.symlink_to(t_b_src)
            t_b_dest = dest.resolve()
        else:
            t_b_dest = copy_tiff_to_input_files(t_b_src, input_dir, role="t2", tp=t_b, fov=fov)

    readme_lines = [
        f"Inferred pairwise review stacks (FOV {fov}).",
        f"T1 timepoint: {t_a}",
        f"T2 timepoint: {t_b}",
        "",
    ]
    if t_a_dest:
        readme_lines.append(f"T1 file: {t_a_dest.name}")
        readme_lines.append(f"  source: {t_a_src}")
    if t_b_dest:
        readme_lines.append(f"T2 file: {t_b_dest.name}")
        readme_lines.append(f"  source: {t_b_src}")
    (input_dir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return t_a_dest, t_b_dest, warnings


def write_inferred_export(
    tables: InferredPairTables,
    *,
    export_dir: Path,
    fov: int,
    t_a: str,
    t_b: str,
    comparison: str,
    export_a: BaselineExportData,
    export_b: BaselineExportData,
    symlink_tiffs: bool,
    skip_tiffs: bool,
) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)

    matched_df = _dedupe_matched(tables.matched)
    new_ids = _dedupe_sorted(tables.new)
    lost_ids = _dedupe_sorted(tables.lost)
    unresolved_df = _dedupe_unresolved(tables.unresolved)

    _safe_write_csv(matched_df, export_dir / "matched.csv")
    _safe_write_csv(pd.DataFrame({"t2_spine_id": new_ids}), export_dir / "new.csv")
    _safe_write_csv(pd.DataFrame({"t1_spine_id": lost_ids}), export_dir / "lost.csv")
    _safe_write_csv(unresolved_df, export_dir / "unresolved_manual_review.csv")

    metadata: dict = {
        "source": "baseline_bridge_inference",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "animal_comparison_folder": comparison,
        "fov": fov,
        "t1_timepoint": t_a,
        "t2_timepoint": t_b,
        "baseline_timepoint": BASELINE_TP,
        "source_baseline_exports": {
            t_a: str(export_a.export_dir),
            t_b: str(export_b.export_dir),
        },
        "matched_count": int(len(matched_df)),
        "new_count": int(len(new_ids)),
        "lost_count": int(len(lost_ids)),
        "unresolved_count": int(len(unresolved_df)),
    }

    if not skip_tiffs:
        t1_dest, t2_dest, warnings = stage_input_files(
            export_dir,
            fov=fov,
            t_a=t_a,
            t_b=t_b,
            export_a=export_a,
            export_b=export_b,
            symlink=symlink_tiffs,
        )
        if t1_dest:
            metadata["t1_tiff_path"] = str(t1_dest)
        if t2_dest:
            metadata["t2_tiff_path"] = str(t2_dest)
        if warnings:
            metadata["tiff_warnings"] = warnings

    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def pair_already_has_manual_export(fov_dir: Path, comparison: str) -> bool:
    comp_dir = fov_dir / comparison
    return latest_manual_export_dir(comp_dir) is not None if comp_dir.is_dir() else False


def nonbaseline_pairs_to_generate(
    existing_nonbaseline: Set[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    tps = list(NON_BASELINE_TPS)
    for i, t_a in enumerate(tps):
        for t_b in tps[i + 1 :]:
            if (t_a, t_b) not in existing_nonbaseline:
                pairs.append((t_a, t_b))
    return pairs


def existing_nonbaseline_pairs(fov_dir: Path) -> Set[Tuple[str, str]]:
    found: Set[Tuple[str, str]] = set()
    for comp_dir in sorted(p for p in fov_dir.iterdir() if p.is_dir()):
        if not list_manual_export_dirs(comp_dir):
            continue
        name = comp_dir.name
        if name in COMPARISON_BY_PAIR.values():
            for pair, folder in COMPARISON_BY_PAIR.items():
                if folder == name:
                    found.add(pair)
            continue
        if " - " in name:
            left, right = name.split(" - ", 1)
            left, right = left.strip(), right.strip()
            if left in TP_ORDER_INDEX and right in TP_ORDER_INDEX:
                t1, t2 = sorted((left, right), key=lambda t: TP_ORDER_INDEX[t])
                if t1 != BASELINE_TP and t2 != BASELINE_TP:
                    found.add((t1, t2))
    return found


def load_baseline_exports_for_fov(fov_dir: Path) -> Dict[str, BaselineExportData]:
    exports: Dict[str, BaselineExportData] = {}
    for tp, comp_name in BASELINE_COMPARISON_MAP.items():
        comp_dir = fov_dir / comp_name
        if not comp_dir.is_dir():
            continue
        data = load_baseline_export(comp_dir, tp)
        if data is not None:
            exports[tp] = data
    return exports


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Infer non-baseline pairwise spine exports by bridging through baseline "
            "(pre-droplet) classifications."
        )
    )
    p.add_argument(
        "--animal-id",
        default="GP04",
        help="Animal folder name under imaging root (e.g. GP04, GP08).",
    )
    p.add_argument(
        "--imaging-root",
        type=Path,
        default=None,
        help=(
            "Parent of animal folders (e.g. E:/.../Imaging). "
            "Default: search for IMAGING under cwd and script parents."
        ),
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override path to respan/results (skips imaging-root/animal-id discovery).",
    )
    p.add_argument("--fovs", type=int, nargs="*", default=None, help="Optional FOV subset.")
    p.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help='Only these pairs, e.g. "mid-droplet - end-lever" (chronological T1 - T2).',
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing inferred exports.")
    p.add_argument("--dry-run", action="store_true", help="Print planned writes only.")
    p.add_argument("--no-tiffs", action="store_true", help="Skip copying TIFFs into input_files/.")
    p.add_argument(
        "--symlink-tiffs",
        action="store_true",
        help="Symlink TIFFs into input_files/ instead of copying.",
    )
    return p.parse_args(argv)


def _default_imaging_root() -> Optional[Path]:
    candidates = [
        Path(r"E:\Noa\Pons - layer 5\Imaging"),
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for base in candidates:
        for name in ("IMAGING", "Imaging", "imaging"):
            p = base / name if (base / name).exists() else base
            if p.is_dir() and any(p.glob("GP*")):
                return p
    return None


def parse_pair_arg(raw: str) -> Tuple[str, str]:
    if " - " not in raw:
        raise argparse.ArgumentTypeError(f"Invalid pair {raw!r}; expected 'T1 - T2'.")
    left, right = raw.split(" - ", 1)
    left, right = left.strip(), right.strip()
    if left not in TP_ORDER_INDEX or right not in TP_ORDER_INDEX:
        raise argparse.ArgumentTypeError(f"Unknown timepoint in {raw!r}")
    if left == BASELINE_TP or right == BASELINE_TP:
        raise argparse.ArgumentTypeError("Only non-baseline pairs are supported.")
    if TP_ORDER_INDEX[left] >= TP_ORDER_INDEX[right]:
        raise argparse.ArgumentTypeError(f"T1 must be earlier than T2: {raw!r}")
    return left, right


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.results_root:
        results_root = args.results_root.expanduser().resolve()
    else:
        imaging_root = args.imaging_root or _default_imaging_root()
        if imaging_root is None:
            print(
                "ERROR: Could not locate imaging root. Pass --imaging-root or --results-root.",
                file=sys.stderr,
            )
            return 1
        results_root = resolve_results_root(imaging_root.expanduser().resolve(), args.animal_id)

    if not results_root.is_dir():
        print(f"ERROR: Results root not found: {results_root}", file=sys.stderr)
        return 1

    fovs = discover_fovs(results_root)
    if args.fovs:
        wanted = set(args.fovs)
        fovs = [f for f in fovs if f in wanted]
    if not fovs:
        print(f"ERROR: No fov* directories under {results_root}", file=sys.stderr)
        return 1

    if args.pairs:
        target_pairs = [parse_pair_arg(p) for p in args.pairs]
    else:
        target_pairs = None  # filled per FOV

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Animal:       {args.animal_id}")
    print(f"Results root: {results_root}")
    print(f"FOVs:         {fovs}")
    print()

    manifest: List[dict] = []

    for fov in fovs:
        fov_dir = results_root / f"fov{fov}"
        baseline_exports = load_baseline_exports_for_fov(fov_dir)
        if len(baseline_exports) < 2:
            print(
                f"fov{fov}: need >=2 baseline exports; found {len(baseline_exports)} — skipping."
            )
            continue

        pairs_for_fov = target_pairs or nonbaseline_pairs_to_generate(
            existing_nonbaseline_pairs(fov_dir)
        )
        if not pairs_for_fov:
            print(f"fov{fov}: all non-baseline pairs already have manual exports.")
            continue

        print(f"fov{fov}: baseline exports for {', '.join(sorted(baseline_exports))}")

        for t_a, t_b in pairs_for_fov:
            if t_a not in baseline_exports or t_b not in baseline_exports:
                print(f"  SKIP {t_a} - {t_b} (missing baseline export for one timepoint)")
                continue

            comparison = comparison_folder_name(t_a, t_b)
            comp_dir = fov_dir / comparison
            export_name = f"{stamp}{INFERRED_EXPORT_SUFFIX}"
            export_dir = comp_dir / export_name

            if pair_already_has_manual_export(fov_dir, comparison) and not args.force:
                print(f"  SKIP {comparison} (manual export exists; use --force)")
                continue

            if export_dir.exists() and not args.force:
                print(f"  SKIP {comparison} ({export_name} exists; use --force)")
                continue

            export_a = baseline_exports[t_a]
            export_b = baseline_exports[t_b]
            tables = infer_pair_via_baseline(export_a, export_b, t_a=t_a, t_b=t_b)

            summary = {
                "fov": fov,
                "comparison": comparison,
                "t1_tp": t_a,
                "t2_tp": t_b,
                "matched": len(_dedupe_matched(tables.matched)),
                "new": len(_dedupe_sorted(tables.new)),
                "lost": len(_dedupe_sorted(tables.lost)),
                "unresolved": len(_dedupe_unresolved(tables.unresolved)),
                "export_dir": str(export_dir),
            }
            manifest.append(summary)

            if args.dry_run:
                print(f"  DRY-RUN {comparison}: {summary}")
                continue

            meta = write_inferred_export(
                tables,
                export_dir=export_dir,
                fov=fov,
                t_a=t_a,
                t_b=t_b,
                comparison=comparison,
                export_a=export_a,
                export_b=export_b,
                symlink_tiffs=args.symlink_tiffs,
                skip_tiffs=args.no_tiffs,
            )
            print(
                f"  WROTE {comparison}: matched={meta['matched_count']} "
                f"new={meta['new_count']} lost={meta['lost_count']} "
                f"unresolved={meta['unresolved_count']} -> {export_dir}"
            )

    if manifest and not args.dry_run:
        manifest_path = results_root / "baseline_bridge_inference_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nManifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
