# Architecture & File Map

## Repository layout

### Root
* `nis_util.py` – NIS macro wrappers, `_run_macro` helper. Root-level shim for legacy code; package version lives in `autofrap/microscope/nis.py`
* `cellpose_server.py` – FastAPI Cellpose inference server, `--device auto|cuda|mps|cpu`. Runs on a remote GPU machine — the microscope PC is CPU-only (its K2200 GPU is unsupported by current PyTorch); GPU inference is much faster than CPU (verified live)
* `grid_utils.py` – pure grid geometry, `gen_grid`, `spiral_positions`. Root-level shim for legacy code; package version lives in `autofrap/core/utils/grid.py`
* `pyproject.toml`, `requirements.txt`
* `DESIGN_GOALS_AUTOFRAP.md`, `README_draft.md`, `STATUS.md`
* `docs/` – split documentation
* `legacy/wingscanner/` – pre-existing wing-scanner project (kept as-is): `automation.py`, `annotation.py`, `resources.py`, `simple_detection.py`, `start_wing_scanner.bat`, calibration JSONs in `res/`, exploratory notebooks
* `nis_ar_help_html/` – extracted NIS macro CHM (greppable HTML), source of truth for macro signatures; `nis_manual/` – NIS manual (both gitignored)

### `autofrap/`
Package with public API re-exports.

* `__init__.py` – public API re-exports for `autofrap()` / `autofrap_grid()`
* `pipeline/autofrap.py` – `autofrap()` and `autofrap_grid()` (+ CLI)
* `core/`
  * `core/detection.py` – `build_detector` composer, `load_detector_file`, runtime parameter routing
  * `core/image/segmentation/` – segmentation building blocks: `simple.py` (Otsu+watershed `SimpleSegParams`/`detect_objects`), `remote.py` (Cellpose client), `dummy.py`
  * `core/image/mask.py` – mask / label utilities (stim masks, polygons, filters)
  * `core/image/qc.py` – `save_qc_overlay`, `default_visualization`
  * `core/simple_seg.py` – deprecated shim for `core/image/segmentation/simple`
  * `core/utils/grid.py` – `gen_grid`, `spiral_positions`
* `io/nd2.py` – ND2 read helpers (`read_channel`, `stage_position`)
* `microscope/`
  * `nis.py` – NIS macro wrappers: `_run_macro`, `MacroOp` + `batch_run_macro` (batched ops), getters/setters, ROI + document management, `NDAcquisition` builder
  * `fake_nis.py` – offline stand-in for dry runs
  * `_resources.py` – resource paths (`microscope/res/`)
* `detectors/` – detector files for `--detector` (one `detection_fun` each; see the directory for the current list)
* `autofrap_bitsnpieces/` – one-off experiments, tests, bits & pieces (no per-file docs; see the directory listing)

### Test data
* `test_acquisitions/`
  * `overview/` – 2×2 overview scan 20260819
  * `autofrap_out/` – early dummy runs
  * `dry_run/` – FakeNIS dry runs with QC PNGs
  * `nuclei_20260901_110410.nd2` – real survey for validation
* `test_data/`
  * `FRAP_GMT1_ESC/` – 23 GFP-DNMT1 time series
  * `0013_ch1.tif` + cellpose masks for QC tests

### Documentation
* `docs/SESSION_HISTORY.md` – session history (detailed log of agentic coding sessions)
* `docs/NIS_REFERENCE.md` – macro → wrapper reference
* `docs/ARCHITECTURE.md` – this file

## Key data flow
`autofrap_grid` → for each position:
1. `set_position` → `run_current_nd_experiment` → survey ND2
2. `detection_fun(survey_file)` → labels, stim_mask, viz
3. `next_stimulatable_cell` with centroid matching
4. `add_polygon_roi` whole cell + stim ROI, `set_roi_type(3)`
5. `run_stimulation_experiment` → `save_current_document` → FRAP ND2
6. `cleanup_everything`: delete all ROIs + close documents (batched)

Detector contract: `survey_file -> (labels,) or (labels, stim_mask) or (labels, stim_mask, viz)`

## Notes
* Root-level `nis_util.py` / `grid_utils.py` are shims for legacy/bitsnpieces code; new code uses `autofrap.core.*`, `autofrap.io.*`, `autofrap.microscope.*`.
