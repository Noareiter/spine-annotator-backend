# Spine Annotator

Dendritic spine matching UI (FastAPI). Session autosave and exports use the project `results/` folder at the repo root.

## Layout

```
code final/spine annotator/
  run_annotator.py          # start here
  README.md
  spine_annotator_backend/
    app.py                  # API + embedded UI
    models.py
    session_store.py
    baseline_adapter.py
    crop_service.py
    io_service.py
    ...
```

## Run

From this folder:

```bash
python run_annotator.py
```

Or:

```bash
python -m spine_annotator_backend
```

Open http://127.0.0.1:8010

## Paths

- **Workspace / results:** `learning_project_spines/results/` (including `session_state/last_session/`)
- **Hybrid matching deps:** `learning_project_spines/scripts/hybrid_tracking/`, `scripts/step2-spine tracking/`

## Related

- GP04 summary pipeline: `scripts/build_gp04_spine_summary.py` (repo root)
