# spine_pipeline

All spine-summary code lives in this folder.

## Run (from animal `respan` folder)

```cmd
cd /d "E:\...\GP04\respan"
python "D:\learning_project_spines\code final\step 3 - dendrite information\run.py"
```

Or from `scripts/`:

```cmd
cd /d "d:\learning_project_spines\scripts"
python -m spine_pipeline
```

## Files

| File | Role |
|------|------|
| `run.py` | Main launcher (edit `DEFAULT_*` at top) |
| `build_spine_summary.py` | Full CLI (`--input`, `--fovs`, …) |
| `build_gp04_spine_summary.py` | Deprecated shim |
| `pipeline.py` | `run_pipeline()` orchestration |
| `layout.py` | Discover timepoints / FOVs / comparisons |
| `io_and_qc.py` | Load RESPAN + exports, QC (see `common.py` export contract) |
| `density_registry.py` | Landmark density + registry tables |
| `dendrite_landmark_qc_report.csv` | Output: rejected dendrites + FOV volume fractions |
| `common.py` | Constants and dataclasses |

`scripts/run_spine_pipeline.py` and siblings are thin redirects for old paths.
