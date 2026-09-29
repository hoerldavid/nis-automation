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
* Error handling: any FOV-level failure skips the FOV; the grid aborts after `--max-consecutive-failures` (default 3) consecutive FOV failures (a completed FOV resets the counter). `AbortRunError` (configuration/resource) aborts immediately. Best-effort per-cycle cleanup. Full policy + the timeout assumption: `autofrap/pipeline/autofrap.py` module docstring.
* Clean Ctrl-C: `AutofrapInterruptedException` with checkpoints P1 cycle end, P2 after survey, P3 between FOVs.
* Live verified 20260901: 2×2 grid, real DAPI nuclei, cellpose `diameter=70`. Survey ~10 s, detection ~2.1 s, stimulation ~14 s, return to start.
* FakeNIS offline dry-run works: `autofrap/microscope/fake_nis.py` (failure table for error injection) + `autofrap/autofrap_bitsnpieces/dry_run_pipeline.py` (thin wrapper around the real CLI) + `autofrap/autofrap_bitsnpieces/test_offline_pipeline.py` (offline assertion suite: fake sanity, pre-flight, failure policy, clean stop).

**QC**
* `autofrap.core.image.qc.save_qc_overlay` renders per-cycle PNG with image, FRAP mask, labels, polygons, legend.

## Open TODOs

* **Detector tuning on real samples** – try `diameter`/`min_size` per sample, consider multi-channel input.
* **CLI flag for `allow_interrupt_after_survey`** – parameter exists, not exposed via argparse.
* **Stimulation ROI groups S1–S3** – `ChangeROIType(3)` → group 1. No macro API for group selection found. Low priority.
* **Pixel ↔ stage coordinate transform for per-tile ROIs** – legacy wing-scanner calibrations retro-analyzed (20260929): one common linear part (scale × flip × rotation) + corner-anchored origin rule (`NIS_REFERENCE.md` § Pixel ↔ stage transform). Open until the planned live test on the current unit confirms the anchor rule (Δ = FOV/2, first tile at (left, top)) via nd2 stage-position metadata and matches the fitted M against `get_rotation_matrix` (per-unit values). Low priority.
* **bitsnpieces cleanup** – stale fossils removed (20260928); the remaining one-offs all run against current code (some need the scope, the cellpose server, or data that lives on other machines — see their docstrings).
* **User-facing documentation** – `README_draft.md` exists, needs refinement and move to repo root (and a link to `WRITING_DETECTOR.md`, written 20260929).

## Known gotchas

* FakeNIS patching works by modifying the module object (`autofrap.microscope.nis`), not by name matching. Direct function imports (`from module import function`) create independent references that bypass patching. Always access NIS operations via the module namespace (e.g., `nis_util.batch_run_macro()`, `nis_util._OP_MACRO_NAME`) to ensure FakeNIS can patch them.
* ROIs are session-global (`ScopeType.Global`) and picked up by any new acquisition — the pipeline deletes all ROIs before/after each cycle via `cleanup_everything` (details: `docs/NIS_REFERENCE.md` § Important ROI behavior).
* Other NIS-specific gotchas (macro file handles, `Int_SetKeyValue`, absolute paths, `CloseCurrentDocument` dialog, `ND_DefineExperiment` save switch, `Frozen` live view, ...): `docs/NIS_REFERENCE.md` § Gotchas.

## Recent sessions
* 20260929 – Detector writing guide `WRITING_DETECTOR.md` (user-facing, repo root): `build_detector` modular approach, building-block tables, explicit `parameter_map` for `--detector-arg`. Along the way: fixed non-executable `python -m autofrap.pipeline` (module-level `main` + `__main__.py`) and refreshed `example_detector.py`. Follow-up: drop `parameter_map='auto'`, convert built-in modular detectors to explicit maps.
* 20260929 – Pixel ↔ stage transform: retro-analysis of the legacy wing-scanner overview calibrations (offline, no code). All six calibs decompose into one common linear part + a corner-anchored origin (first stitch tile centered on the bbox (left, top) corner, offset ≈ half tile) — the manual GUI calibration may be replaceable by `get_rotation_matrix` + FOV + nd2 stage positions. Findings + planned live test: `docs/NIS_REFERENCE.md` § Pixel ↔ stage transform; TODO left open until live-verified on the current unit.
* 20260928 – Consecutive-failure error policy replaces the Recoverable/NonRecoverable routing; generic `run_with_retries` helper added (`autofrap/core/utils/retry.py`). Policy + timeout assumption documented in the pipeline module docstring. Follow-up: test scripts consolidated (`dry_run_pipeline.py` is now a thin wrapper around the real CLI, new `test_offline_pipeline.py` assertion suite, FakeNIS failure table replaces the old knobs, `Frozen` special-casing removed from the fake document state machine); `--out` default is now cwd-relative `autofrap_out`, abspassed before NIS (NIS macros resolve relative paths against the executable's directory). Then a full `autofrap_bitsnpieces` sweep: 7 stale fossils deleted (the `nis_util_old.py` snapshot pair, the superseded `overview_scan.py`, `test_cellpose.py`, the ephemeral-data `test_nd2_stage_position.py`, the calmutils pair), and 11 scripts fixed to the current layout (root `nis_util`/`grid_utils` imports, one-level-shallow `ROOT`s, the new default-visualization contract) — all 10 offline-runnable tests green.
* 20260922 – Documentation rework: one fact, one home — `docs/SESSION_HISTORY.md` becomes a pure session log (fossil sections merged/removed), AGENTS.md codifies the documentation structure.
* 20260922 – Import refactoring, FakeNIS improvements, dry-run enhancements (top-level imports, `_OP_*` access pattern, dry-run presets verified).
* 20260918 – Live microscope validation + bug fixes (batched-op double-parses, backward ROI deletion, ND Acquisition doc activation safeguard).
* 20260918 – Centralized cleanup (`cleanup_everything`) and untangled `autofrap_loop_inner` into survey / select+QC / stimulation helpers.

Full session history: `docs/SESSION_HISTORY.md`
NIS macro reference: `docs/NIS_REFERENCE.md`
Architecture / file map: `docs/ARCHITECTURE.md`
