# Status: autoFRAP NIS-Elements Automation

## Purpose
Automate multi-FOV, multi-cycle FRAP on Nikon microscopes via NIS Elements. Survey → detect → pick cell → create whole-cell + stimulation ROI → run stimulation → save. See `DESIGN_GOALS_AUTOFRAP.md` for the full workflow.

## Current state

**Infrastructure – verified**
* Macro execution: temp `.mac` → `nis_ar.exe -mw` → temp `.ini`. Helper `_run_macro` used by all `nis_util` wrappers.
* Live-verified wrappers: `get_position`, `get_resolution`, `get_rotation_matrix`, `get_optical_confs`, `get_nd_acq_tabs`, `get_opened_documents`/`activate_document`/`activate_opened_document`, `get_roi_ids`, `run_current_nd_experiment`, `run_stimulation_experiment`, `save_current_document`, `add_polygon_roi`, `set_roi_type`, `delete_roi`.
* FOV = xres * pixel_size / magnification. e.g. 133.1 µm at 100x/13 µm.
* **New macro batching pattern** `autofrap/microscope/nis.py`: `MacroOp` dataclass with `build(params,section)` and `parse`. Ops for `position`, `resolution`, `nd_acq_tabs` refactored; `batch_run_macro` runs multiple ops in one `nis_ar` call with unique ini sections. See `docs/NIS_REFERENCE.md` § Macro batching.
* Added new `MacroOp`s for batching and idempotent cleanup: `checkpoint`, `delete_all_rois_in_current_document`, `close_all_docs`, `add_polygon_roi` now MacroOp-based. Single-call wrappers `delete_all_rois_in_current_document`, `close_all_docs`, `checkpoint` added. `FakeNIS` patched for `batch_run_macro` and the new ops. **Needs live microscope testing** – the macros have not been run on the scope workstation yet; verify `CloseCurrentDocument` flag semantics and ROI delete loop on real NIS.
* Batching gotcha: NIS macro requires all variable declarations before any executable statements. `batch_run_macro` now hoists `int/double/char/dword/byte/word/float` declarations to the top and automatically renames variables to `section_var` to avoid collisions across ops. Every `nis_ar` call carries a roughly constant startup overhead, so batching several ops into one call measurably reduces it (verified live; concrete timings in `docs/SESSION_HISTORY.md`).

**Detection**
* `build_detector` composer with `parameter_map=None|'auto'|dict`. Runtime args via `--detector-arg`.
* Remote Cellpose: `cellpose_server.py` FastAPI, `POST /detect` np.save in/out. Client `remote_detect_objects` 60 s timeout + 1 retry.
* Border discard + relabel `distance`/`shuffle`. Stim mask default = left half of object.
* Built-in detectors: `dummy_detector.py`, `simple_seg_detector.py` (Otsu+watershed), `cellpose_remote_detector.py`, `cellpose_remote_halfnucleus_modular.py`.

**Pipeline**
* `autofrap.pipeline.autofrap` single FOV, `autofrap_grid` multi-FOV.
* Cross-cycle cell tracking: centroid matching against an accumulated “already imaged” map (`centroid_threshold='auto'` ≈ one equivalent diameter per cell), so each cell is stimulated at most once.
* Run dir naming: `<out>/<stamp>_<name>` with `--name`/`--no-timestamp`. Non-empty dir collision aborts.
* Error handling: `RecoverableError` → skip FOV, `NonRecoverableError` → abort grid. Best-effort cleanup.
* Clean Ctrl-C: `AutofrapInterruptedException` with checkpoints P1 cycle end, P2 after survey, P3 between FOVs.
* Live verified 20260901: 2×2 grid, real DAPI nuclei, cellpose `diameter=70`. Survey ~10 s, detection ~2.1 s, stimulation ~14 s, return to start.
* FakeNIS offline dry-run works: `autofrap/fake_nis.py` + `dry_run_pipeline.py`.

**QC**
* `autofrap.core.image.qc.save_qc_overlay` renders per-cycle PNG with image, FRAP mask, labels, polygons, legend.

## Open TODOs

1. **Detector tuning on real samples** – try `diameter`/`min_size` per sample, consider multi-channel input.
2. **CLI flag for `allow_interrupt_after_survey`** – parameter exists, not exposed via argparse.
3. **Stimulation ROI groups S1–S3** – `ChangeROIType(3)` → group 1. No macro API for group selection found. Low priority.
4. **Pixel ↔ stage coordinate transform for per-tile ROIs** – calibration matrix from `get_rotation_matrix`. Low priority.
5. **Final cleanup housekeeping** – stale one-offs left as-is per convention.
6. **User-facing documentation** – `README_draft.md` exists, needs refinement and move to repo root.
7. **Finish documentation rework** – STATUS.md: recent-sessions rollup, unnumbered TODOs, trim redundant milestones; `docs/ARCHITECTURE.md`: full file-map refresh; `docs/NIS_REFERENCE.md`: complete wrapper list. (Structure/rules themselves are in place per AGENTS.md.)

## Known gotchas

* FakeNIS patching works by modifying the module object (`autofrap.microscope.nis`), not by name matching. Direct function imports (`from module import function`) create independent references that bypass patching. Always access NIS operations via the module namespace (e.g., `nis_util.batch_run_macro()`, `nis_util._OP_MACRO_NAME`) to ensure FakeNIS can patch them.
* ROI persistence: ROIs from `CreatePolygonROI` are `ScopeType.Global` and session-global, picked up by any new acquisition. The exact inheritance behavior for new acquisitions is still poorly understood, but is mitigated in practice by deleting all ROIs before adding new ones each cycle and using `cleanup_everything` which reliably removes ROIs via backward iteration.
* Close `.mac` handle before `nis_ar`, or GUI reports "Can't open file for reading".
* NIS keeps lock on failed `.mac` → `PermissionError` on remove.
* `Int_SetKeyValue` only numeric; use `Int_SetKeyString` for strings.
* File paths in macros must be absolute.
* `CloseCurrentDocument(save='yes')` pops Save-As dialog and blocks macro.
* ROIs are session-global; closing document does not remove them.
* `ImageSaveAs` on `Frozen` live view silently writes nothing.
* `ND_DefineExperiment` filename is the de-facto save on/off switch.
* `GetROIInfo` color read-back always 0, but colors render correctly.

## Recent milestones
* 20260922 – Import refactoring and dry-run improvements:
  - Moved all imports to top-level in `autofrap/pipeline/autofrap.py` for consistency and Python best practices.
  - Updated all MacroOp references to use `nis_util._OP_*` pattern instead of direct imports.
  - Fixed `_inner_loop_stimulation` to extract ROI IDs from dict returned by `batch_run_macro`.
  - Enhanced FakeNIS: added `get_opened_documents` to PATCHED_FUNCTIONS and implemented fake method.
  - Updated `dry_run_pipeline.py`: changed SURVEY_GLOB to use `test_acquisitions/autofrap_out/*survey.nd2`, added `dummy` preset using `dummy_detector.py`.
  - Verified all three presets (dummy, simple_seg, cellpose) work correctly with FakeNIS.
* 20260918 – Live microscope validation and bug fixes:
  - Fixed `_OP_DELETE_ALL_ROIS_IN_CURRENT_DOCUMENT` macro to iterate backwards to avoid skipping ROIs on deletion; stimulation ROI no longer persists after `cleanup_everything`.
  - Fixed `_parse_nd_acq_tabs` fallback usage `sec_cfg.get(tab, fallback='0')` → `sec_cfg.get(tab, '0')` for `SectionProxy` compatibility.
  - Fixed `setup_microscope` double-parse of `batch_run_macro` results and incorrect result keys `position_0`/`resolution_0` → `position_1`/`resolution_2`.
  - Fixed `_inner_loop_stimulation` double-parse of `add_polygon_roi` results from `batch_run_macro`.
  - Moved inline `_OP_CLOSE_CURRENT_DOCUMENT` MacroOp to `autofrap/microscope/nis.py`, updated `close_current_document` wrapper to use the op, and imported it in `autofrap/pipeline/autofrap.py`.
  - Added safeguard activation of unsaved `ND Acquisition` document after stimulation completes and before FRAP save to reduce accidental user interaction with survey ROIs.
  - Live tested 2×2 grid with `simple_seg_detector.py` and `488 CSU-W1 FRAP`, then 1-FOV until-done run with remote Cellpose server `10.163.69.12:8000`, 7 cells stimulated over 8 cycles with clean cleanup.
* 202610? – Centralize cleanup and untangle autofrap_loop_inner:
  - Added `cleanup_everything(nis_exe)` with n_open==0 guard and 3× TimeoutError retry using batched delete ROIs + close current document.
  - Updated `cleanup_run` and `autofrap_loop_inner` finally block to use `cleanup_everything`.
  - Removed explicit per-cycle block 7 document/ROI teardown from inner loop.
  - Extracted survey acquisition + document sanity check to `_inner_loop_do_survey`.
  - Extracted matching / cell selection / polygon building / QC overlay to `_inner_loop_select_cell_and_qc` with stop check left before the call.
  - Extracted ROI creation, optical config switch, stimulation run and FRAP save to `_inner_loop_stimulation`.
  - Inner loop now a readable survey → detect → select+QC → stimulate pipeline.
* 20260917 – Refactor autoFRAP pipeline to explicit setup / cleanup blocks and batched macro calls:
  - Added `setup_microscope` with batched `nd_acq_tabs + position + resolution` reads and 0/2/4 s retry.
  - Introduced `autofrap_loop_inner`, `autofrap_loop_outer`, `cleanup_run`, and outer `autofrap` orchestrator; position generation moved outside the loop.
  - Added `move_stage_with_retry` helper with 0/2/4 s retry and `TimeoutError` handling.
  - Batched ROI creation: `delete_all_rois_in_current_document + add_polygon_roi x2` via `batch_run_macro` with one retry on timeout; `set_roi_type` kept separate.
  - Replaced per-ROI `delete_roi` with `delete_all_rois_in_current_document` in cycle cleanup and finally block; removed reopen-on-failure guards.
  - Updated `FakeNIS` to return MacroOp-compatible dicts for `add_polygon_roi` and patched `batch_run_macro` usage via `nis_util`.
  - Dry-run pipeline verified with Cellpose remote server; intensity filter threshold lowered to 450 for test images.
  - Updated `autofrap/__init__.py` alias for backwards compatibility.
* 20260916 – Small fixes at microscope + detector examples: live 2×2 and spiral runs with Cellpose remote on real sample; fixed `pip install -e .` multiple-top-level-packages error via explicit `[tool.setuptools.packages.find]` in `pyproject.toml`; fixed import bug in `autofrap/pipeline/autofrap.py` (`NonRecoverableError` import) and added `--frap-oc` CLI flag; fixed `build_positions` spiral FOV scalar handling; added intensity filter `filter_intensity_inside` to `cellpose_remote_halfnucleus_modular.py` (channel 0 mean > 550) and created detector variants `cellpose_remote_cluster_modular.py` (cluster stim mask, channel 2) and `cellpose_remote_randomcircle_modular.py` (random circle stim mask). Live verified spiral 5-position, 2-cycle runs with `488 CSU-W1 FRAP` optical config.
* 20260916 – Multi-channel building blocks + default RGB visualization: added `channel` selection to `remote_detect_objects`, `cluster_stim_mask`, `detect_objects`/`segment_nuclei_otsu_watershed` in `autofrap/core/simple_seg.py`. Added `default_visualization` for (c,y,x) → RGB composite with per-channel percentile normalization. Updated `autofrap/detectors/cellpose_remote_halfnucleus_modular.py` to use `default_visualization` and explicit `load_channel`/`det_channel` routing via `parameter_map='auto'`. Verified with dry-run pipeline using `load_channel='all'` and `det_channel=0`. Closed TODO #8 “Update building blocks to accept (C,Y,X) images”.
* 20260916 – `read_channel` load-all support: added `channel='all'` option to `autofrap/io/nd2.py`. Default remains `channel=0` → 2-D `(Y,X)`. `channel='all'` returns `(C,Y,X)`, promoting single-channel files to `(1,Y,X)`. Building blocks still expect 2-D labels/masks; updating them to accept `(C,Y,X)` with explicit channel selection is now TODO #8.
* 20260916 – Cellpose server URL per-run: detectors now accept `server_url` via `**detector_kwargs` / `--detector-arg server_url=...` with fallback to `CELLPOSE_SERVER_URL` env var → `DEFAULT_CELLPOSE_SERVER_URL`. Updated `autofrap/detectors/cellpose_remote_detector.py` and `cellpose_remote_halfnucleus_modular.py`, tested via dry-run with modular detector. TODO #8 closed.
* 20260916 – Spiral position count CLI + refactor: added `--max-positions/--num-positions` flag, applied as hard cap to both grid and spiral visit orders; extracted `build_positions` and `parse_cli_args` helpers from `__main__` in `autofrap/pipeline/autofrap.py`. Updated `README_draft.md` usage example. TODO #8 closed.
* 20260915 – Mask utilities bbox-local refactor: `half_object_stim_mask`, `random_circle_stim_mask`, `largest_region_per_label`, `most_central_region_per_label`, `mask_to_polygon` rewritten to operate per-object bbox with EDT-based centre selection for random circles. TODO #5 closed.
* 20260915 – Package reorg cleanup completed: compatibility shims `autofrap/detection.py`, `autofrap/mask_utils.py`, `autofrap/nd2_helpers.py`, `autofrap/fake_nis.py` removed after bitsnpieces imports migrated to `autofrap.core.*`, `autofrap.io.*`, `autofrap.microscope.*`. TODO #8 closed.
* 20260914 – Package reorg step 1, simple-seg detector refactored into core, README draft + TODO updates.
* 20260910 – FakeNIS dry-run, cellpose server `--device`, clean Ctrl-C stop, experiment name/run dir guards, detector runtime parameters.
* 20260909 – ROI persistence probes, session-global ROIs discovery, opened-document wrappers live verified, ND template pre-flight check.
* 20260901 – Live 2×2 grid with real cellpose, stage position from ND2 metadata verified.

Full session history: `docs/SESSION_HISTORY.md`
NIS macro reference: `docs/NIS_REFERENCE.md`
Architecture / file map: `docs/ARCHITECTURE.md`
