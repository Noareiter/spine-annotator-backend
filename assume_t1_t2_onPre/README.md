# assume_t1_t2_onPre — baseline-bridged pairwise inference

Generate **missing non-baseline** pairwise spine annotator exports (e.g. `mid-droplet` vs `end-lever`) from existing **baseline ↔ timepoint** manual classifications.

## Logic (via `pre-droplet`)

For follow-up timepoints **T_A** (earlier) and **T_B** (later), using exports `pre-droplet ↔ T_A` and `pre-droplet ↔ T_B`:

| Outcome | Rule |
|--------|------|
| **matched.csv** | Same baseline spine matched at both T_A and T_B |
| **lost.csv** | Matched at T_A, **lost** at T_B (gone by T_B) |
| **new.csv** | **Lost** at T_A, matched at T_B (appears by T_B) |
| **unresolved_manual_review.csv** | New at T_A (no baseline bridge), ignored/removed, or ambiguous |

TIFF stacks for review are copied into `input_files/` from `matching_activity_log.txt` (or `metadata.json`) on each baseline comparison export.

## Input layout

```
<IMAGING_ROOT>/<ANIMAL_ID>/respan/results/fovN/<comparison>/<timestamp>_spine_annotator_export/
  matched.csv, new.csv, lost.csv, removed_*.csv, ignored_*.csv
  matching_activity_log.txt
```

Baseline comparison folder names (Step 3 convention):

- `pre-mid droplet`, `pre-end droplet`, `pre droplet-end lever`, `pre droplet - return to droplet`

## Usage

```cmd
python "D:\learning_project_spines\code final\assume_t1_t2_onPre\infer_baseline_bridged_pairs.py" ^
  --animal-id GP08 ^
  --imaging-root "E:\Noa\Pons - layer 5\Imaging"
```

Or point directly at results:

```cmd
python infer_baseline_bridged_pairs.py ^
  --results-root "E:\Noa\Pons - layer 5\Imaging\GP04\respan\results"
```

### Options

| Flag | Description |
|------|-------------|
| `--animal-id` | Animal folder (default: `GP04`) |
| `--imaging-root` | Parent of `GP04`, `GP08`, … |
| `--results-root` | Override `.../respan/results` path |
| `--fovs 1 2` | Subset of FOVs |
| `--pairs "mid-droplet - end-lever"` | Only named pairs |
| `--dry-run` | Print plan only |
| `--force` | Overwrite inferred exports |
| `--no-tiffs` | Skip `input_files/` |
| `--symlink-tiffs` | Symlink instead of copy |

## Output layout

```
results/fov1/mid-droplet - end-lever/2026-06-01_12-00-00_inferred_spine_annotator_export/
  input_files/
    t1_mid-droplet_fov1.tif
    t2_end-lever_fov1.tif
    README.txt
  matched.csv
  new.csv
  lost.csv
  unresolved_manual_review.csv
  metadata.json
```

Default: all **6** non-baseline chronological pairs per FOV that do not already have a **manual** export. Skips pairs where `*_inferred_*` or manual export already exists unless `--force`.

Manifest: `results/baseline_bridge_inference_manifest.json`

## Related

- Step 3 pipeline: `code final/step 3 - dendrite information/`
- Spine annotator: `code final/spine annotator/`
