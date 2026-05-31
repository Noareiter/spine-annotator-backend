from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .common import COMPARISON_MAP, KNOWN_TIMEPOINTS, StudyLayout, TIMEPOINTS, TP_ORDER, set_active_timepoints


def resolve_input_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Study input folder not found: {root}")
    return root


def infer_animal_id(input_root: Path) -> str:
    if input_root.name.lower() == "respan" and input_root.parent.name:
        return input_root.parent.name
    return input_root.name


def discover_timepoints(input_root: Path) -> Dict[str, dict]:
    found: Dict[str, dict] = {}
    for tp in TP_ORDER:
        meta = KNOWN_TIMEPOINTS[tp]
        tables = input_root / meta["folder"] / "Tables"
        if tables.is_dir() and any(tables.glob("*.csv")):
            found[tp] = meta
    if not found:
        expected = ", ".join(TP_ORDER)
        raise FileNotFoundError(
            f"No timepoint Tables/ folders found under {input_root}. Expected: {expected}"
        )
    return found


def discover_fovs(results_dir: Path) -> List[int]:
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Missing annotator results folder: {results_dir}\nExpected: <input>/results/fov1/, fov2/, ..."
        )
    fovs: List[int] = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"fov(\d+)", d.name, flags=re.IGNORECASE)
        if m:
            fovs.append(int(m.group(1)))
    return fovs


def discover_expected_comparisons(results_dir: Path, fovs: List[int]) -> Set[str]:
    names: Set[str] = set()
    for fov in fovs:
        fov_dir = results_dir / f"fov{fov}"
        if not fov_dir.is_dir():
            continue
        for comp_dir in fov_dir.iterdir():
            if comp_dir.is_dir():
                names.add(comp_dir.name)
    return names


def _infer_tp_from_metadata_path(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    parts = Path(path_str.replace("\\", "/")).parts
    for tp, meta in TIMEPOINTS.items():
        if meta["folder"] in parts:
            return tp
    return None


def resolve_comparison_tps(comp_name: str, metadata: dict) -> Optional[Tuple[str, str]]:
    if comp_name in COMPARISON_MAP:
        t1, t2 = COMPARISON_MAP[comp_name]
        if t1 in TIMEPOINTS and t2 in TIMEPOINTS:
            return t1, t2
    t1_tp = _infer_tp_from_metadata_path(str(metadata.get("t1_csv_path", "")))
    t2_tp = _infer_tp_from_metadata_path(str(metadata.get("t2_csv_path", "")))
    if t1_tp and t2_tp:
        return t1_tp, t2_tp
    return None


def discover_study_layout(
    input_root: Path,
    *,
    animal_id: Optional[str] = None,
    out_dir: Optional[Path] = None,
    fovs: Optional[List[int]] = None,
) -> StudyLayout:
    input_root = resolve_input_root(input_root)
    animal = animal_id or infer_animal_id(input_root)
    results_dir = input_root / "results"
    output = (out_dir or (Path.cwd() / "spine_summary")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    timepoints = discover_timepoints(input_root)
    set_active_timepoints(timepoints)

    fov_list = sorted(fovs) if fovs else discover_fovs(results_dir)
    if not fov_list:
        raise FileNotFoundError(f"No fov* folders under {results_dir}")

    return StudyLayout(
        input_root=input_root,
        animal_id=animal,
        results_dir=results_dir,
        out_dir=output,
        timepoints=timepoints,
        tp_order=TP_ORDER,
        fovs=fov_list,
        expected_comparisons=discover_expected_comparisons(results_dir, fov_list),
    )


def print_layout_summary(layout: StudyLayout) -> None:
    print(f"Study root:    {layout.input_root}")
    print(f"Animal ID:     {layout.animal_id}")
    print(f"Results:       {layout.results_dir}")
    print(f"Output:        {layout.out_dir}")
    print(f"Timepoints:    {', '.join(layout.tp_order)}")
    print(f"FOVs:          {', '.join(f'fov{n}' for n in layout.fovs)}")
    print(f"Comparisons:   {len(layout.expected_comparisons)} folder name(s) under results/fov*")
