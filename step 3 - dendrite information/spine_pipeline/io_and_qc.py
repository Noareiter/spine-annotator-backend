from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from .common import (
    ARTIFACT_STATUS,
    EXPORT_ALL_CSV_NAMES,
    EXPORT_CAT1_BIOLOGICAL,
    EXPORT_CAT2_UNCERTAINTY,
    EXPORT_CAT3_ARTIFACT,
    EXPORT_CAT5_IGNORED,
    EXPORT_DENDRITE_EXCLUSION_FILE,
    EXPORT_MATCHED_FILE,
    ExportBundle,
    FovData,
    STATUS_MATCHED,
    STATUS_UNMATCHED,
    TIMEPOINTS,
    TP_ORDER,
)
from .layout import resolve_comparison_tps


def safe_write_csv(df: pd.DataFrame, path: Path, *, index: bool = False, retries: int = 5) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(retries):
        try:
            df.to_csv(tmp, index=index)
            os.replace(tmp, path)
            return path
        except PermissionError:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(0.5)
                continue
            alt = path.with_name(f"{path.stem}_locked_{int(time.time())}{path.suffix}")
            df.to_csv(alt, index=index)
            print(f"WARNING: Could not overwrite {path.name}. Wrote {alt.name} instead.")
            return alt
    return path


def safe_write_text(path: Path, text: str, *, retries: int = 5) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(retries):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            return path
        except PermissionError:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(0.5)
                continue
            alt = path.with_name(f"{path.stem}_locked_{int(time.time())}{path.suffix}")
            alt.write_text(text, encoding="utf-8")
            print(f"WARNING: Could not overwrite {path.name}. Wrote {alt.name} instead.")
            return alt
    return path


def _find_fov_tables_csv(tables_dir: Path, fov: int, keyword: str, tp: str) -> Path:
    """
    Find a CSV under tables_dir whose name matches fov{N}.*{keyword}.*.csv (case-insensitive).

    Timepoint is defined by the parent folder (tables_dir); tp is only used in errors.
    """
    if not tables_dir.is_dir():
        raise FileNotFoundError(f"Tables directory not found: {tables_dir} (timepoint={tp!r})")
    pattern = re.compile(rf"fov{fov}.*{re.escape(keyword)}.*\.csv$", re.IGNORECASE)
    matches = sorted(p for p in tables_dir.glob("*.csv") if pattern.search(p.name))
    if not matches:
        raise FileNotFoundError(
            f"No {keyword} CSV matching fov{fov} in {tables_dir} "
            f"(timepoint={tp!r}; search pattern: fov{fov}.*{keyword}.*.csv)"
        )
    return matches[0]


def resolve_detected_spines(tables_dir: Path, fov: int, tp: str) -> Path:
    return _find_fov_tables_csv(tables_dir, fov, "detected_spines", tp)


def resolve_dendrite_summary(tables_dir: Path, fov: int, tp: str) -> Path:
    return _find_fov_tables_csv(tables_dir, fov, "dendrite_summary", tp)


def normalize_spine_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "spine_id" in out.columns and "id" not in out.columns:
        out = out.rename(columns={"spine_id": "id"})
    if "id" not in out.columns:
        raise ValueError("Spine CSV missing id/spine_id column")
    out["id"] = out["id"].astype(str)
    if "dendrite_id" in out.columns:
        out["dendrite_id"] = out["dendrite_id"].astype(str)
    return out.set_index("id", drop=False)


def load_spines_for_fov(respan_root: Path, fov: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    spines: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, pd.DataFrame] = {}
    for tp, meta in TIMEPOINTS.items():
        tables = respan_root / meta["folder"] / "Tables"
        sp_path = resolve_detected_spines(tables, fov, tp)
        ds_path = resolve_dendrite_summary(tables, fov, tp)
        sdf = normalize_spine_df(pd.read_csv(sp_path))
        spines[tp] = sdf
        ddf = pd.read_csv(ds_path)
        if "dendrite_id" not in ddf.columns:
            raise ValueError(f"{ds_path} missing dendrite_id")
        ddf["dendrite_id"] = ddf["dendrite_id"].astype(str)
        summaries[tp] = ddf.set_index("dendrite_id", drop=False)
    return spines, summaries


def discover_exports(input_root: Path, fov: int) -> List[ExportBundle]:
    fov_dir = input_root / "results" / f"fov{fov}"
    if not fov_dir.is_dir():
        raise FileNotFoundError(f"Missing {fov_dir}")
    bundles: List[ExportBundle] = []
    for comp_dir in sorted(fov_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        comp_name = comp_dir.name
        exports = sorted(comp_dir.glob("*spine_annotator_export"))
        if not exports:
            continue
        export_dir = exports[-1]
        meta_path = export_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        resolved = resolve_comparison_tps(comp_name, meta)
        if not resolved:
            continue
        t1_tp, t2_tp = resolved
        bundles.append(
            ExportBundle(
                fov=fov,
                comparison=comp_name,
                t1_tp=t1_tp,
                t2_tp=t2_tp,
                export_dir=export_dir,
                metadata=meta,
            )
        )
    return bundles


def _read_id_set(path: Path, col: str) -> Set[str]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path)
    if col not in df.columns:
        return set()
    return {str(x) for x in df[col].dropna().astype(str)}


def _warn_unexpected_export_csvs(export_dir: Path) -> None:
    """Warn on unknown CSVs in an export folder; never load Category 5 audit files."""
    for path in sorted(export_dir.glob("*.csv")):
        name = path.name
        if name in EXPORT_CAT5_IGNORED:
            continue
        if name not in EXPORT_ALL_CSV_NAMES:
            print(f"WARNING: Unrecognized export CSV (ignored): {export_dir / name}")


def _set_spine_status(
    fd: FovData,
    tp: str,
    spine_ids: Iterable[str],
    status: str,
    *,
    overwrite: bool,
) -> None:
    for sid in spine_ids:
        key = (tp, str(sid))
        if overwrite or key not in fd.spine_status:
            fd.spine_status[key] = status
        elif fd.spine_status[key] == STATUS_UNMATCHED and status != STATUS_UNMATCHED:
            fd.spine_status[key] = status


def _apply_spine_status_exports(
    fd: FovData,
    export_dir: Path,
    t1_tp: str,
    t2_tp: str,
    specs: Tuple[Tuple[str, str, str, str], ...],
    *,
    overwrite: bool,
) -> None:
    """Load Category 1/2/3 spine-list CSVs: (filename, t1|t2, status, id_column)."""
    for filename, side, status, id_col in specs:
        if filename in EXPORT_CAT5_IGNORED:
            continue
        tp = t1_tp if side == "t1" else t2_tp
        ids = _read_id_set(export_dir / filename, id_col)
        _set_spine_status(fd, tp, ids, status, overwrite=overwrite)


def _apply_matched_export(fd: FovData, export_dir: Path, bundle: ExportBundle) -> None:
    """Category 1: matched.csv — verified anchors + dendrite union for linking."""
    path = export_dir / EXPORT_MATCHED_FILE
    if not path.is_file():
        return
    mdf = pd.read_csv(path)
    t1_tp, t2_tp = bundle.t1_tp, bundle.t2_tp
    for _, row in mdf.iterrows():
        t2_id = str(row.get("t2_spine_id", "")).strip()
        t1_id = str(row.get("t1_spine_id", "")).strip()
        if t2_id:
            _set_spine_status(fd, t2_tp, [t2_id], STATUS_MATCHED, overwrite=True)
        if t1_id:
            _set_spine_status(fd, t1_tp, [t1_id], STATUS_MATCHED, overwrite=True)
        if t1_id and t2_id and t1_id in fd.spines[t1_tp].index and t2_id in fd.spines[t2_tp].index:
            d1 = str(fd.spines[t1_tp].loc[t1_id, "dendrite_id"])
            d2 = str(fd.spines[t2_tp].loc[t2_id, "dendrite_id"])
            fd.uf.union((t1_tp, d1), (t2_tp, d2))


def _apply_dendrite_exclusion_export(fd: FovData, export_dir: Path, t1_tp: str, t2_tp: str) -> None:
    """Category 4: blacklist entire dendrites before density math."""
    path = export_dir / EXPORT_DENDRITE_EXCLUSION_FILE
    if not path.is_file():
        return
    exdf = pd.read_csv(path)
    for _, row in exdf.iterrows():
        tp_label = str(row.get("timepoint", "")).strip().lower()
        did = str(row.get("dendrite_id", "")).strip()
        if not did:
            continue
        if tp_label == "t1":
            fd.excluded_dendrites[t1_tp].add(did)
        elif tp_label == "t2":
            fd.excluded_dendrites[t2_tp].add(did)


def apply_export_qc(fd: FovData, bundle: ExportBundle) -> None:
    """
    Map annotator export CSVs to spine statuses and dendrite exclusions.

    Category 5 files (rejected_pairs, validation queues, manual click logs) are never read.
    Status priority (low -> high): biological -> matched -> artifact -> uncertainty.
    Uncertainty overrides matched so segments are disqualified; artifact does not.
    """
    d = bundle.export_dir
    t1_tp, t2_tp = bundle.t1_tp, bundle.t2_tp

    _warn_unexpected_export_csvs(d)

    # Category 1 (partial): new, lost
    _apply_spine_status_exports(fd, d, t1_tp, t2_tp, EXPORT_CAT1_BIOLOGICAL, overwrite=True)

    # Category 1: matched anchors (overwrites new/lost on same id if present)
    _apply_matched_export(fd, d, bundle)

    # Category 3: software artifacts (excluded from counts; does not use NOT_RELEVANT_STATUSES)
    _apply_spine_status_exports(fd, d, t1_tp, t2_tp, EXPORT_CAT3_ARTIFACT, overwrite=True)

    # Category 2: visual uncertainty (disqualifies landmark segments via NOT_RELEVANT_STATUSES)
    _apply_spine_status_exports(fd, d, t1_tp, t2_tp, EXPORT_CAT2_UNCERTAINTY, overwrite=True)

    # Category 4: dendrite-level blacklist (independent of spine_status)
    _apply_dendrite_exclusion_export(fd, d, t1_tp, t2_tp)


def finalize_spine_status(fd: FovData) -> None:
    for tp, df in fd.spines.items():
        for sid in df.index.astype(str):
            key = (tp, sid)
            if key not in fd.spine_status:
                fd.spine_status[key] = STATUS_UNMATCHED


def linked_dendrite_ids_at_tp(fd: FovData, tp: str) -> Set[str]:
    all_ids = set(fd.dendrite_summary[tp].index.astype(str))
    return all_ids - fd.excluded_dendrites.get(tp, set())


def assign_link_ids(fd: FovData, animal_id: str) -> pd.DataFrame:
    rows: List[dict] = []
    for tp in TP_ORDER:
        for did in linked_dendrite_ids_at_tp(fd, tp):
            fd.uf._add((tp, did))
    comps = fd.uf.components()
    comp_index = 0
    for _, nodes in sorted(comps.items(), key=lambda x: str(x[0])):
        active = [(tp, did) for tp, did in nodes if did in linked_dendrite_ids_at_tp(fd, tp)]
        if not active:
            continue
        comp_index += 1
        link_id = f"{animal_id}_fov{fd.fov}_D{comp_index:03d}"
        for tp, did in active:
            fd.link_id_by_dendrite[(tp, did)] = link_id
            rows.append({"animal_id": animal_id, "fov": fd.fov, "link_id": link_id, "timepoint": tp, "dendrite_id": did})
    return pd.DataFrame(rows)


def pivot_dendrite_links_wide(links_long: pd.DataFrame, tp_order: List[str]) -> pd.DataFrame:
    if links_long.empty:
        return links_long
    rows = []
    for (animal, fov, link_id), grp in links_long.groupby(["animal_id", "fov", "link_id"]):
        rec = {"animal_id": animal, "fov": fov, "link_id": link_id}
        for tp in tp_order:
            sub = grp[grp["timepoint"] == tp]
            rec[f"dendrite_id_{tp}"] = ",".join(sorted(sub["dendrite_id"].astype(str).unique())) if len(sub) else ""
        rows.append(rec)
    return pd.DataFrame(rows)
