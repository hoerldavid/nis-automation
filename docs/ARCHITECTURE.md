# Architecture & File Map

## Repository layout

### Root
* `cellpose_server.py` – FastAPI Cellpose inference server, `--device auto|cuda|mps|cpu`. Runs on a remote GPU machine — the microscope PC is CPU-only (its K2200 GPU is unsupported by current PyTorch); GPU inference is much faster than CPU (verified live)
* `pyproject.toml`, `requirements.txt`
* `DESIGN_GOALS_AUTOFRAP.md`, `README_draft.md`, `STATUS.md`
* `docs/` – split documentation
* `legacy/wingscanner/` – pre-existing wing-scanner project (kept as-is): `automation.py`, `annotation.py`, `resources.py`, `simple_detection.py`, `start_wing_scanner.bat`, calibration JSONs in `res/`, exploratory notebooks, plus `nis_util.py` / `grid_utils.py` shims that re-export the package modules for the legacy code
* `nis_manual/` – intended location for the NIS manual: `README.md` describes how to obtain the (copyrighted, gitignored) `.chm` files from a NIS installation and extract them to greppable HTML; `nis_ar_help_html/` – the extracted macro reference HTML (both gitignored)

### `autofrap/`
Package (`__init__.py` files intentionally empty since 20261001 — import from the submodules; the old re-exports and `autofrap_grid`/`autofrap_multiposition` aliases are gone).

* `pipeline/autofrap.py` – `autofrap()` (entry point: setup + position build), `autofrap_loop_outer()` (loop over positions), `autofrap_loop_inner()` (per-FOV work; the original `autofrap()`); CLI via `python -m autofrap.pipeline`
* `pipeline/dry_run.py` – offline dry-run tool, `python -m autofrap.pipeline.dry_run`: the real CLI under FakeNIS, with detector presets and the position count capped to the number of source files by default
* `core/`
  * `core/detection.py` – `build_detector` composer, `load_detector_file`, `unpack_detection_result` / `parse_detector_args` (shared by both CLIs), runtime parameter routing
  * `core/image/segmentation/` – segmentation building blocks: `simple.py` (Otsu+watershed `SimpleSegParams`/`detect_objects`), `remote.py` (Cellpose client), `dummy.py`
  * `core/image/mask.py` – mask / label utilities (stim masks, polygons, filters, `match_imaged_centroids` = cross-cycle cell matching, `next_stimulatable_cell`, `select_next_cell` = pipeline cell selection incl. polygon viability)
  * `core/image/qc.py` – `save_qc_overlay`, `default_visualization`
  * `core/utils/grid.py` – `gen_grid`, `grid_positions`, `spiral_positions` (generator-based)
* `io/nd2.py` – ND2 read helpers (`read_channel`, `stage_position`)
* `microscope/`
  * `nis.py` – NIS macro wrappers: `_run_macro`, `MacroOp` + `batch_run_macro` (batched ops), getters/setters, ROI + document management, `NDAcquisition` builder
  * `fake_nis.py` – offline stand-in for dry runs
  * `_resources.py` – resource paths (`microscope/res/`)
* `detectors/` – detector files for `--detector` (one `detection_fun` each; see the directory for the current list) + `cli.py` (offline detector runner: `python -m autofrap.detectors --detector <file> image.nd2` — runs one detector on one image and saves a QC overlay)
* `autofrap_bitsnpieces/` – one-off experiments, plots, benchmarks (no per-file docs; see the directory listing) — the test suite lives in `tests/`, the dry-run tool in `autofrap/pipeline/dry_run.py`

### `tests/`
Stdlib `unittest` suite, mirroring the `autofrap/` layout (`core/`, `io/`, `microscope/`, `pipeline/`, `live/`). Run from the repo root: `python -m unittest discover -s tests -t .` (plain `python -m unittest discover` works too). `tests/live/` holds the microscope probes — they auto-skip off the workstation; data-dependent tests skip when their files (`01.nd2`, `test_acquisitions/`, `test_data/`) are missing, so the suite is green on every machine. `tests/pipeline/test_dry_run_cli.py` covers the CLI layer itself end-to-end (argument parsing, `--detector` file loading, exit codes) — the other pipeline tests drive the loops directly.

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
`autofrap()` → setup + position build → `autofrap_loop_outer()` → per position → `autofrap_loop_inner()` → per cycle:
1. `move_stage_with_retry` (batched set+get, position verified) → `run_current_nd_experiment` → survey ND2
2. `detection_fun(survey_file)` → labels, stim_mask, viz
3. `match_imaged_centroids` + `select_next_cell` (candidate order incl. polygon viability)
4. batched ROIs: `create_and_set_stim_roi` stim ROI (type 3 folded in) + `add_polygon_roi` whole cell
5. batched: `set_optical_configuration` → `run_stimulation_experiment` → `activate_document` + `save_current_document` → FRAP ND2
6. `cleanup_everything`: delete all ROIs + close documents (batched)

Detector contract: `survey_file -> (labels,) or (labels, stim_mask) or (labels, stim_mask, viz)`

## Notes
* Legacy code's `nis_util` / `grid_utils` imports resolve via the shims in `legacy/wingscanner/`; new code uses `autofrap.core.*`, `autofrap.io.*`, `autofrap.microscope.*`.
