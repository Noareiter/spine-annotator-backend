"""CLI and orchestration for the spine summary pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .common import DEFAULT_MIN_VALID_FRAC, FovData
from .density_registry import (
    add_density_normalization,
    build_animal_density,
    build_dendrite_density,
    build_dendrite_landmark_qc_report,
    build_fov_density,
    build_registry_long,
    build_registry_wide,
    build_spine_lineages,
    collect_fov_tp_coverage_records,
    format_analysis_coverage_text,
)
from .io_and_qc import (
    apply_export_qc,
    assign_link_ids,
    discover_exports,
    finalize_spine_status,
    load_spines_for_fov,
    pivot_dendrite_links_wide,
    safe_write_csv,
    safe_write_text,
)
from .layout import discover_study_layout, print_layout_summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build spine registry and per-dendrite density from a respan study folder."
    )
    p.add_argument(
        "--input",
        "--respan-root",
        dest="input",
        type=Path,
        default=None,
        help="Study root (default: current working directory, e.g. .../GP04/respan).",
    )
    p.add_argument(
        "--animal-id",
        type=str,
        default=None,
        help="Label for outputs (default: parent folder of 'respan', e.g. GP04).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./spine_summary under current working directory).",
    )
    p.add_argument("--fovs", type=int, nargs="*", default=None, help="Optional FOV list, e.g. 1 2 3")
    p.add_argument(
        "--min-valid-frac",
        type=float,
        default=DEFAULT_MIN_VALID_FRAC,
        help="Minimum fraction of dendrite length that must remain valid after landmark QC.",
    )
    return p.parse_args(argv)


def run_pipeline(
    input_root: Path,
    *,
    animal_id: Optional[str] = None,
    out_dir: Optional[Path] = None,
    fovs: Optional[List[int]] = None,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
) -> Path:
    """Run full pipeline programmatically; returns output directory."""
    layout = discover_study_layout(
        input_root,
        animal_id=animal_id,
        out_dir=out_dir,
        fovs=fovs,
    )
    print_layout_summary(layout)
    print(f"Min valid frac: {min_valid_frac} (landmark QC)")
    print()

    all_dendrite_links: List[pd.DataFrame] = []
    all_dendrite_density: List[pd.DataFrame] = []
    all_registry: List[pd.DataFrame] = []
    coverage_records: List[dict] = []
    manifest_rows = []
    n_expected = len(layout.expected_comparisons)

    for fov in layout.fovs:
        try:
            spines, summaries = load_spines_for_fov(layout.input_root, fov)
        except FileNotFoundError as exc:
            manifest_rows.append({"fov": fov, "status": "skip", "note": str(exc)})
            continue

        fd = FovData(fov=fov, spines=spines, dendrite_summary=summaries)
        fd.exports = discover_exports(layout.input_root, fov)
        for bundle in fd.exports:
            apply_export_qc(fd, bundle)
        finalize_spine_status(fd)
        dendrite_links = assign_link_ids(fd, layout.animal_id)
        all_dendrite_links.append(dendrite_links)

        ddf = build_dendrite_density(fd, layout.animal_id, min_valid_frac)
        coverage_records.extend(collect_fov_tp_coverage_records(fd, ddf, layout.animal_id))
        all_dendrite_density.append(add_density_normalization(ddf))

        spine_uf, _ = build_spine_lineages(fd)
        all_registry.append(build_registry_wide(fd, layout.animal_id, spine_uf))

        manifest_rows.append(
            {
                "fov": fov,
                "status": "ok",
                "n_exports": len(fd.exports),
                "n_dendrite_links": dendrite_links["link_id"].nunique() if not dendrite_links.empty else 0,
                "n_lineages": len(all_registry[-1]),
                "n_expected_comparisons": n_expected,
                "comparisons_missing": max(0, n_expected - len(fd.exports)),
            }
        )

    out = layout.out_dir
    dendrite_links_df = pd.concat(all_dendrite_links, ignore_index=True) if all_dendrite_links else pd.DataFrame()
    dendrite_links_wide = pivot_dendrite_links_wide(dendrite_links_df, layout.tp_order)
    dendrite_density_df = pd.concat(all_dendrite_density, ignore_index=True) if all_dendrite_density else pd.DataFrame()
    registry_wide = pd.concat(all_registry, ignore_index=True) if all_registry else pd.DataFrame()
    registry_long = build_registry_long(registry_wide)
    fov_density_df = build_fov_density(dendrite_density_df, layout.animal_id) if not dendrite_density_df.empty else pd.DataFrame()
    animal_density_df = build_animal_density(fov_density_df, layout.animal_id) if not fov_density_df.empty else pd.DataFrame()
    dendrite_qc_df = build_dendrite_landmark_qc_report(dendrite_density_df)
    manifest_df = pd.DataFrame(manifest_rows)

    written: List[str] = []
    for df_out, name in (
        (dendrite_links_df, "dendrite_links_reconstructed.csv"),
        (dendrite_links_wide, "dendrite_links_wide.csv"),
        (dendrite_density_df, "dendrite_density_by_timepoint.csv"),
        (dendrite_qc_df, "dendrite_landmark_qc_report.csv"),
        (fov_density_df, "fov_density_by_timepoint.csv"),
        (animal_density_df, "animal_density_by_timepoint.csv"),
        (registry_wide, "spine_registry_wide.csv"),
        (registry_long, "spine_observations_long.csv"),
        (manifest_df, "run_manifest.csv"),
    ):
        p = safe_write_csv(df_out, out / name)
        written.append(p.name)

    readme = {
        "animal_id": layout.animal_id,
        "input_root": str(layout.input_root),
        "timepoints": layout.tp_order,
        "fovs": layout.fovs,
        "min_valid_frac": min_valid_frac,
        "density_eff_rule": (
            "Landmark segmentation between consecutive matched spines; "
            "rho_eff = n_eff_landmark / L_eff_landmark_um; NaN if valid length fraction < min_valid_frac"
        ),
        "outputs": written,
    }
    readme_path = safe_write_text(out / "README_summary.json", json.dumps(readme, indent=2))
    written.append(readme_path.name)

    coverage_text = format_analysis_coverage_text(
        coverage_records,
        animal_id=layout.animal_id,
        input_root=str(layout.input_root),
        min_valid_frac=min_valid_frac,
    )
    coverage_path = safe_write_text(out / "analysis_coverage_summary.txt", coverage_text)
    written.append(coverage_path.name)

    print(f"Wrote outputs to {out}")
    print("Files:", ", ".join(written))
    if not manifest_df.empty:
        print(manifest_df.to_string(index=False))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    input_root = (args.input or Path.cwd()).resolve()
    out_dir = (args.out_dir or (Path.cwd() / "spine_summary")).resolve()
    run_pipeline(
        input_root,
        animal_id=args.animal_id,
        out_dir=out_dir,
        fovs=args.fovs,
        min_valid_frac=args.min_valid_frac,
    )


if __name__ == "__main__":
    main()
