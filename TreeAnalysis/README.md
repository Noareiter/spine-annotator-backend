# TreeAnalysis

Place high-magnification spine fields-of-view (FOVs) into whole-cell structural
context so dendritic-spine dynamics can be analyzed by their location on the
dendritic arbor (branch, branch order, path distance from soma).

## The two coordinate problems

1. **Within a session** the 5-6 high-mag spine FOVs share one microscope stage
   origin, so their stage X/Y/Z stitch them into a single tuft mosaic
   (bookkeeping only).
2. **Mosaic -> whole-cell SWC** cannot use stage numbers: the whole-cell
   overview is a one-time, separate-session acquisition whose stage origin is
   not comparable to the FOV sessions. This step is solved by skeleton /
   branch-point registration.

## Pipeline (intended)

| Step | Module | Role |
|------|--------|------|
| 1 | `io_pvscan_xml.py` | Parse Bruker PrairieView `PVScan` XML: calibration, stage XYZ, per-frame Z |
| 2 | `io_spines.py` | Load RESPAN `detected_spines` coordinates |
| 3 | `io_swc.py` | Parse an SWC trace into a node/edge table |
| 4 | `correct_swc_z.py` | Remap SWC z slice-index -> true depth via per-animal schedule |
| 5 | `build_tree_graph.py` | Tree graph; soma root; path length & branch order per node |
| 6 | `stitch_session_fovs.py` | Stage-coordinate mosaic of one session's FOVs (+ overlap dedup) |
| 7 | `extract_branchpoints.py` | Branch-point coordinates for any skeleton |
| 8 | `register_mosaic_to_cell.py` | Branch-point matching + rigid/scale + ICP refine |
| 9 | `apply_transform.py` | Map spines into whole-cell frame |
| 10 | `assign_to_branch.py` | Snap spine -> nearest edge; emit structural coordinates |
| 11 | `qc_overlay.py` | Overlays + branch-point residual plots |
| - | `run_tree_analysis.py` | Orchestrator |

Each module is a separate, independently testable script, as requested.

## Conventions (per-animal, see `config/`)

- Coordinate units: micrometres (um) throughout.
- Lateral axes: `+X = right`; per-axis sign flags allow flipping if QC disagrees.
- Stage `positionCurrent` = FOV **center**.
- High-mag FOV z-step: uniform (Optotune ETL), read from XML.
- Whole-cell z: SNT exports z as **slice index** (z-voxel set to 1 during
  tracing); convert to true depth with a cutoff slice + before/after z-step
  schedule entered per animal.
