from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# Chronological timepoint order (keys == on-disk folder names under respan root).
TP_ORDER: List[str] = [
    "pre-droplet",
    "mid-droplet",
    "end-droplet",
    "end-lever",
    "return to droplet",
]

KNOWN_TIMEPOINTS: Dict[str, dict] = {
    "pre-droplet": {"order": 1, "folder": "pre-droplet", "fov_case": "lower"},
    "mid-droplet": {"order": 2, "folder": "mid-droplet", "fov_case": "upper"},
    "end-droplet": {"order": 3, "folder": "end-droplet", "fov_case": "lower"},
    "end-lever": {"order": 4, "folder": "end-lever", "fov_case": "lower"},
    "return to droplet": {"order": 5, "folder": "return to droplet", "fov_case": "lower"},
}

TIMEPOINTS: Dict[str, dict] = dict(KNOWN_TIMEPOINTS)

# Baseline timepoint for fold-normalization vs pre-study mean.
BASELINE_TIMEPOINT: str = "pre-droplet"

# --- Spine status values (annotator export -> pipeline) ---
STATUS_MATCHED: str = "matched"
STATUS_NEW: str = "new"
STATUS_LOST: str = "lost"
STATUS_IGNORED: str = "ignored"
STATUS_NOT_IN_FOCUS: str = "not_in_focus"
STATUS_UNMATCHED: str = "unmatched"
ARTIFACT_STATUS: str = "artifact"

# Category 2: visual uncertainty — disqualifies landmark segments (NOT length from artifacts).
NOT_RELEVANT_STATUSES: Set[str] = {STATUS_IGNORED, STATUS_NOT_IN_FOCUS}

# Category 3: software false positives — excluded from density counts; must NOT be in NOT_RELEVANT_STATUSES.
assert ARTIFACT_STATUS not in NOT_RELEVANT_STATUSES

# Biological spines counted in rho_eff inside valid landmark segments only.
N_EFF_LANDMARK_STATUSES: Set[str] = {STATUS_MATCHED, STATUS_NEW, STATUS_LOST}

# Gross n_eff tally (legacy column); excludes uncertainty and artifacts.
N_EFF_STATUSES: Set[str] = set(N_EFF_LANDMARK_STATUSES)

# --- Annotator export CSV contract (15 CSV files + metadata.json) ---
# Each spine spec: (filename, t1|t2 side, pipeline status, id column).
# Application order in apply_export_qc is list order; later specs win on conflict.

EXPORT_MATCHED_FILE: str = "matched.csv"
EXPORT_DENDRITE_EXCLUSION_FILE: str = "excluded_non_matched_dendrite_spines.csv"

# Category 1: core biological outcomes
EXPORT_CAT1_BIOLOGICAL: Tuple[Tuple[str, str, str, str], ...] = (
    ("new.csv", "t2", STATUS_NEW, "t2_spine_id"),
    ("lost.csv", "t1", STATUS_LOST, "t1_spine_id"),
)

# Category 2: visual uncertainty (segment disqualification)
EXPORT_CAT2_UNCERTAINTY: Tuple[Tuple[str, str, str, str], ...] = (
    ("ignored_t1.csv", "t1", STATUS_IGNORED, "t1_spine_id"),
    ("ignored_t2.csv", "t2", STATUS_IGNORED, "t2_spine_id"),
    ("not_in_t1_focus.csv", "t2", STATUS_NOT_IN_FOCUS, "t2_spine_id"),
)

# Category 3: software false positives (count exclusion only; not segment disqualification)
EXPORT_CAT3_ARTIFACT: Tuple[Tuple[str, str, str, str], ...] = (
    ("removed_t1.csv", "t1", ARTIFACT_STATUS, "t1_spine_id"),
    ("removed_t2.csv", "t2", ARTIFACT_STATUS, "t2_spine_id"),
)

# Category 5: audit / temporary — never loaded by the pipeline
EXPORT_CAT5_IGNORED: frozenset[str] = frozenset(
    {
        "rejected_pairs.csv",
        "new_validation_pending.csv",
        "manual_t2_click_spines.csv",
        "manual_t1_click_spines.csv",
        "manual_click_matches.csv",
    }
)

# All CSV filenames the annotator may write (for sanity checks; metadata.json is separate).
EXPORT_ALL_CSV_NAMES: frozenset[str] = frozenset(
    {EXPORT_MATCHED_FILE, EXPORT_DENDRITE_EXCLUSION_FILE}
    | {row[0] for row in EXPORT_CAT1_BIOLOGICAL}
    | {row[0] for row in EXPORT_CAT2_UNCERTAINTY}
    | {row[0] for row in EXPORT_CAT3_ARTIFACT}
    | EXPORT_CAT5_IGNORED
)

DEFAULT_MIN_VALID_FRAC = 0.5

# Comparison folder names under results/fovX/ -> (t1_folder, t2_folder). Case- and space-sensitive.
COMPARISON_MAP: Dict[str, Tuple[str, str]] = {
    "pre-mid droplet": ("pre-droplet", "mid-droplet"),
    "pre-end droplet": ("pre-droplet", "end-droplet"),
    "pre droplet-end lever": ("pre-droplet", "end-lever"),
    "pre droplet - return to droplet": ("pre-droplet", "return to droplet"),
    "end droplet - return to droplet": ("end-droplet", "return to droplet"),
    "end droplet - end lever": ("end-droplet", "end-lever"),
}


def set_active_timepoints(found: Dict[str, dict]) -> None:
    global TIMEPOINTS, TP_ORDER
    TIMEPOINTS = found
    TP_ORDER = sorted(TIMEPOINTS.keys(), key=lambda k: TIMEPOINTS[k]["order"])


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def _add(self, x: Tuple[str, str]) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: Tuple[str, str]) -> Tuple[str, str]:
        self._add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: Tuple[str, str], b: Tuple[str, str]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        out: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return dict(out)


@dataclass
class ExportBundle:
    fov: int
    comparison: str
    t1_tp: str
    t2_tp: str
    export_dir: Path
    metadata: dict


@dataclass
class FovData:
    fov: int
    spines: Dict[str, pd.DataFrame] = field(default_factory=dict)
    dendrite_summary: Dict[str, pd.DataFrame] = field(default_factory=dict)
    exports: List[ExportBundle] = field(default_factory=list)
    spine_status: Dict[Tuple[str, str], str] = field(default_factory=dict)
    excluded_dendrites: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    uf: UnionFind = field(default_factory=UnionFind)
    link_id_by_dendrite: Dict[Tuple[str, str], str] = field(default_factory=dict)


@dataclass
class StudyLayout:
    input_root: Path
    animal_id: str
    results_dir: Path
    out_dir: Path
    timepoints: Dict[str, dict]
    tp_order: List[str]
    fovs: List[int]
    expected_comparisons: Set[str]
