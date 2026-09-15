# Architecture & File Map

## Repository layout

### Root
* `nis_util.py` – NIS macro wrappers, `_run_macro` helper. Root-level shim for legacy code; package version lives in `autofrap/microscope/nis.py`
* `cellpose_server.py` – FastAPI Cellpose inference server, `--device auto|cuda|mps|cpu`
* `grid_utils.py` – pure grid geometry, `gen_grid`, `spiral_positions`. Root-level shim for legacy code; package version lives in `autofrap/core/utils/grid.py`
* `pyproject.toml`, `requirements.txt`
* `DESIGN_GOALS_AUTOFRAP.md`, `README_draft.md`, `STATUS.md`
* `docs/` – split documentation

### `autofrap/`
Package with public API re-exports.

* `__init__.py` – public API re-exports for `autofrap()` / `autofrap_grid()`
* `pipeline/` → `pipeline/autofrap.py` – `autofrap()` and `autofrap_grid()`
* `pipeline/` → `pipeline/autofrap.py` – `autofrap()` and `autofrap_grid()`
* `core/`
  * `core/detection.py` – `build_detector` composer, runtime parameter routing
  * `core/simple_seg.py` – `SimpleSegParams`, `detect_objects`
  * `core/image/mask.py` – mask utilities
  * `core/image/qc.py` – `save_qc_overlay`
  * `core/utils/grid.py`
* `io/` → `io/nd2.py` – ND2 read helpers, `stage_position`
* `microscope/`
  * `microscope/nis.py` – NIS wrappers package version
  * `microscope/fake_nis.py`
  * `microscope/_resources.py`
* `detectors/`
  * `dummy_detector.py`
  * `cellpose_remote_detector.py`
  * `cellpose_remote_halfnucleus_modular.py` – `build_detector` assembled
  * `simple_seg_detector.py`
  * `example_detector.py`
* `autofrap_bitsnpieces/` – one-off experiments, tests, bits & pieces
  * `dry_run_pipeline.py`
  * `simple_seg_experiment.py`
  * `frap_gmt1_es_sweep.py`, `frap_gmt1_es_clusters.py`
  * tests: `test_fake_nis.py`, `test_autofrap_errors.py`, `test_centroid_matching.py`, `test_qc_overlay.py`, etc.

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
* `docs/STATUS_HISTORY.md` – full session log moved from STATUS.md
* `docs/NIS_REFERENCE.md` – macro → wrapper reference
* `docs/ARCHITECTURE.md` – this file

## Key data flow
`autofrap_grid` → for each position:
1. `set_position` → `run_current_nd_experiment` → survey ND2
2. `detection_fun(survey_file)` → labels, stim_mask, viz
3. `next_stimulatable_cell` with centroid matching
4. `add_polygon_roi` whole cell + stim ROI, `set_roi_type(3)`
5. `run_stimulation_experiment` → `save_current_document` → FRAP ND2
6. `delete_roi` both ROIs, close docs

Detector contract: `survey_file -> (labels,) or (labels, stim_mask) or (labels, stim_mask, viz)`

## Notes
* `autofrap/pipeline.py` referenced in old STATUS.md → now `autofrap/pipeline/autofrap.py`
* `autofrap/qc.py` referenced in old STATUS.md → now `autofrap/core/image/qc.py`
* Legacy shims kept at top level for bitsnpieces scripts; new code uses `autofrap.core.*`, `autofrap.io.*`, `autofrap.microscope.*`
