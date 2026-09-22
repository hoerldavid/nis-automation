# Status: autoFRAP NIS-Elements Automation

## Purpose
Automate multi-FOV, multi-cycle FRAP on Nikon microscopes via NIS Elements. Survey → detect → pick cell → create whole-cell + stimulation ROI → run stimulation → save. See `DESIGN_GOALS_AUTOFRAP.md` for the full workflow.

## Current state

**Infrastructure – verified**
* Macro execution: temp `.mac` → `nis_ar.exe -mw` → temp `.ini`. Helper `_run_macro` used by all `nis_util` wrappers.
* Live-verified wrappers: `get_position`, `get_resolution`, `get_rotation_matrix`, `get_optical_confs`, `get_nd_acq_tabs`, `get_opened_documents`/`activate_document`/`activate_opened_document`, `get_roi_ids`, `run_current_nd_experiment`, `run_stimulation_experiment`, `save_current_document`, `add_polygon_roi`, `set_roi_type`, `delete_roi`.
* FOV = xres * pixel_size / magnification. e.g. 133.1 µm at 100x/13 µm.
* **New macro batching pattern** `autofrap/microscope/nis.py`: `MacroOp` dataclass with `build(params,section)` and `parse`. Ops for `position`, `resolution`, `nd_acq_tabs` refactored; `batch_run_macro` runs multiple ops in one `nis_ar` call with unique ini sections. See `docs/NIS_REFERENCE.md` § Macro batching.
* More `MacroOp`s for batching and idempotent cleanup: `checkpoint`, `delete_all_rois_in_current_document`, `close_all_docs`; `add_polygon_roi` and `_OP_CLOSE_CURRENT_DOCUMENT` are MacroOp-based. `FakeNIS` patched for `batch_run_macro` and the new ops. Live-verified 20260918: `delete_all_rois_in_current_document` (backward-iteration delete loop) + `close_current_document`; `checkpoint` / `close_all_docs` not yet run on the scope.
* Batching gotcha: NIS macro requires all variable declarations before any executable statements. `batch_run_macro` now hoists `int/double/char/dword/byte/word/float` declarations to the top and automatically renames variables to `section_var` to avoid collisions across ops. Every `nis_ar` call carries a roughly constant startup overhead, so batching several ops into one call measurably reduces it (verified live; concrete timings in `docs/SESSION_HISTORY.md`).

**Detection**
* `build_detector` composer with `parameter_map=None|'auto'|dict`. Runtime args via `--detector-arg`.
* Remote Cellpose: `cellpose_server.py` FastAPI, `POST /detect` np.save in/out. Client `remote_detect_objects` 60 s timeout + 1 retry.
* Border discard + relabel `distance`/`shuffle`. Stim mask default = left half of object.
* Detector files live in `autofrap/detectors/` (dummy, simple-seg, remote Cellpose variants — see the directory for the current list).

**Pipeline**
* Entry point `autofrap()` (setup + position build) → `autofrap_loop_outer()` over positions → `autofrap_loop_inner()` per FOV (the original `autofrap()`). `autofrap_grid` / `autofrap_multiposition` are aliases for `autofrap_loop_outer`.
* Cross-cycle cell tracking: centroid matching against an accumulated “already imaged” map (`centroid_threshold='auto'` ≈ one equivalent diameter per cell), so each cell is stimulated at most once.
* Run dir naming: `<out>/<stamp>_<name>` with `--name`/`--no-timestamp`. Non-empty dir collision aborts.
* Error handling: `RecoverableError` → skip FOV, `NonRecoverableError` → abort grid. Best-effort cleanup.
* Clean Ctrl-C: `AutofrapInterruptedException` with checkpoints P1 cycle end, P2 after survey, P3 between FOVs.
* Live verified 20260901: 2×2 grid, real DAPI nuclei, cellpose `diameter=70`. Survey ~10 s, detection ~2.1 s, stimulation ~14 s, return to start.
* FakeNIS offline dry-run works: `autofrap/fake_nis.py` + `dry_run_pipeline.py`.

**QC**
* `autofrap.core.image.qc.save_qc_overlay` renders per-cycle PNG with image, FRAP mask, labels, polygons, legend.

## Open TODOs

* **Detector tuning on real samples** – try `diameter`/`min_size` per sample, consider multi-channel input.
* **CLI flag for `allow_interrupt_after_survey`** – parameter exists, not exposed via argparse.
* **Stimulation ROI groups S1–S3** – `ChangeROIType(3)` → group 1. No macro API for group selection found. Low priority.
* **Pixel ↔ stage coordinate transform for per-tile ROIs** – calibration matrix from `get_rotation_matrix`. Low priority.
* **Final cleanup housekeeping** – stale one-offs left as-is per convention.
* **User-facing documentation** – `README_draft.md` exists, needs refinement and move to repo root.

## Known gotchas

* FakeNIS patching works by modifying the module object (`autofrap.microscope.nis`), not by name matching. Direct function imports (`from module import function`) create independent references that bypass patching. Always access NIS operations via the module namespace (e.g., `nis_util.batch_run_macro()`, `nis_util._OP_MACRO_NAME`) to ensure FakeNIS can patch them.
* ROIs are session-global (`ScopeType.Global`) and picked up by any new acquisition — the pipeline deletes all ROIs before/after each cycle via `cleanup_everything` (details: `docs/NIS_REFERENCE.md` § Important ROI behavior).
* Other NIS-specific gotchas (macro file handles, `Int_SetKeyValue`, absolute paths, `CloseCurrentDocument` dialog, `ND_DefineExperiment` save switch, `Frozen` live view, ...): `docs/NIS_REFERENCE.md` § Gotchas.

## Recent sessions
* 20260922 – Documentation rework: one fact, one home — `docs/SESSION_HISTORY.md` becomes a pure session log (fossil sections merged/removed), AGENTS.md codifies the documentation structure.
* 20260922 – Import refactoring, FakeNIS improvements, dry-run enhancements (top-level imports, `_OP_*` access pattern, dry-run presets verified).
* 20260918 – Live microscope validation + bug fixes (batched-op double-parses, backward ROI deletion, ND Acquisition doc activation safeguard).
* 20260918 – Centralized cleanup (`cleanup_everything`) and untangled `autofrap_loop_inner` into survey / select+QC / stimulation helpers.
* 20260917 – Pipeline refactored to explicit setup/cleanup blocks with batched macro calls (`setup_microscope`, `move_stage_with_retry`).

Full session history: `docs/SESSION_HISTORY.md`
NIS macro reference: `docs/NIS_REFERENCE.md`
Architecture / file map: `docs/ARCHITECTURE.md`
