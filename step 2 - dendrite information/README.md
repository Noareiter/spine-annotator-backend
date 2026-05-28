# Step 2 — Dendrite information (spine summary pipeline)

Builds spine registry tables and per-dendrite density from RESPAN + annotator exports.

## Run

Open a terminal in your animal **`respan`** folder (where `pre-droplet/`, `results/fov1/`, etc. live):

```cmd
cd /d "E:\...\GP04\respan"
python "D:\learning_project_spines\code final\step 2 - dendrite information\run.py"
```

## Input / output

| | Path |
|---|------|
| **Input (default)** | Current working directory (`respan`) |
| **Output (default)** | `./spine_summary/` under **the folder you run from** (cwd) |

Override output:

```cmd
python ".../run.py" --out-dir ".\my_run_outputs"
```

## Package layout

| Path | Role |
|------|------|
| `run.py` | Main entry point |
| `spine_pipeline/` | All pipeline modules |
| `spine_pipeline/common.py` | Timepoints, export file contract, QC constants |

## Main outputs (in `spine_summary/`)

- `dendrite_density_by_timepoint.csv`
- `dendrite_landmark_qc_report.csv`
- `analysis_coverage_summary.txt`
- `spine_registry_wide.csv`, `spine_observations_long.csv`
- `fov_density_by_timepoint.csv`, `animal_density_by_timepoint.csv`

Legacy launchers under `scripts/` redirect here.
