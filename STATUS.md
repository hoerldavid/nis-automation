# Status: autoFRAP NIS-Elements Automation

## Purpose
Automate multi-FOV, multi-cycle FRAP on Nikon microscopes via NIS Elements. Survey → detect → pick cell → create whole-cell + stimulation ROI → run stimulation → save. See `DESIGN_GOALS_AUTOFRAP.md` for the full workflow.

## Current state

**Infrastructure – verified**
* Macro execution: temp `.mac` → `nis_ar.exe -mw` → temp `.ini`. Helper `_run_macro` used by all `nis_util` wrappers.
* Live-verified wrappers: `get_position`, `get_resolution`, `get_rotation_matrix`, `get_optical_confs`, `get_nd_acq_tabs`, `get_opened_documents`/`activate_document`/`activate_opened_document`, `get_roi_ids`, `run_current_nd_experiment`, `run_stimulation_experiment`, `save_current_document`, `add_polygon_roi`, `set_roi_type`, `delete_roi`.
* FOV = xres * pixel_size / magnification. e.g. 133.1 µm at 100x/13 µm.

**Detection**
* `build_detector` composer with `parameter_map=None|'auto'|dict`. Runtime args via `--detector-arg`.
* Remote Cellpose: `cellpose_server.py` FastAPI, `POST /detect` np.save in/out. Client `remote_detect_objects` 60 s timeout + 1 retry.
* Border discard + relabel `distance`/`shuffle`. Stim mask default = left half of object.
* Built-in detectors: `dummy_detector.py`, `simple_seg_detector.py` (Otsu+watershed), `cellpose_remote_detector.py`, `cellpose_remote_halfnucleus_modular.py`.

**Pipeline**
* `autofrap.pipeline.autofrap` single FOV, `autofrap_grid` multi-FOV.
* Run dir naming: `<out>/<stamp>_<name>` with `--name`/`--no-timestamp`. Non-empty dir collision aborts.
* Error handling: `RecoverableError` → skip FOV, `NonRecoverableError` → abort grid. Best-effort cleanup.
* Clean Ctrl-C: `AutofrapInterruptedException` with checkpoints P1 cycle end, P2 after survey, P3 between FOVs.
* Live verified 20260901: 2×2 grid, real DAPI nuclei, cellpose `diameter=70`. Survey ~10 s, detection ~2.1 s, stimulation ~14 s, return to start.
* FakeNIS offline dry-run works: `autofrap/fake_nis.py` + `dry_run_pipeline.py`.

**QC**
* `autofrap.core.image.qc.save_qc_overlay` renders per-cycle PNG with image, FRAP mask, labels, polygons, legend.

## Open TODOs

1. **Understand ROI persistence** – ROIs from `CreatePolygonROI` are `ScopeType.Global`, session-global, picked up by any new acquisition. End-of-cycle `delete_roi` mandatory. Exact new-acquisition inheritance still open. No wiring changes until understood.
2. **Detector tuning on real samples** – try `diameter`/`min_size` per sample, consider multi-channel input.
3. **CLI flag for `allow_interrupt_after_survey`** – parameter exists, not exposed via argparse.
4. **Stimulation ROI groups S1–S3** – `ChangeROIType(3)` → group 1. No macro API for group selection found. Low priority.
5. **Pixel ↔ stage coordinate transform for per-tile ROIs** – calibration matrix from `get_rotation_matrix`. Low priority.
6. **Final cleanup housekeeping** – stale one-offs left as-is per convention.
7. **User-facing documentation** – `README_draft.md` exists, needs refinement and move to repo root.

## Known gotchas

* Close `.mac` handle before `nis_ar`, or GUI reports “Can't open file for reading”.
* NIS keeps lock on failed `.mac` → `PermissionError` on remove.
* `Int_SetKeyValue` only numeric; use `Int_SetKeyString` for strings.
* File paths in macros must be absolute.
* `CloseCurrentDocument(save='yes')` pops Save-As dialog and blocks macro.
* ROIs are session-global; closing document does not remove them.
* `ImageSaveAs` on `Frozen` live view silently writes nothing.
* `ND_DefineExperiment` filename is the de-facto save on/off switch.
* `GetROIInfo` color read-back always 0, but colors render correctly.

## Recent milestones
* 20260916 – Cellpose server URL per-run: detectors now accept `server_url` via `**detector_kwargs` / `--detector-arg server_url=...` with fallback to `CELLPOSE_SERVER_URL` env var → `DEFAULT_CELLPOSE_SERVER_URL`. Updated `autofrap/detectors/cellpose_remote_detector.py` and `cellpose_remote_halfnucleus_modular.py`, tested via dry-run with modular detector. TODO #8 closed.
* 20260916 – Spiral position count CLI + refactor: added `--max-positions/--num-positions` flag, applied as hard cap to both grid and spiral visit orders; extracted `build_positions` and `parse_cli_args` helpers from `__main__` in `autofrap/pipeline/autofrap.py`. Updated `README_draft.md` usage example. TODO #8 closed.
* 20260915 – Mask utilities bbox-local refactor: `half_object_stim_mask`, `random_circle_stim_mask`, `largest_region_per_label`, `most_central_region_per_label`, `mask_to_polygon` rewritten to operate per-object bbox with EDT-based centre selection for random circles. TODO #5 closed.
* 20260915 – Package reorg cleanup completed: compatibility shims `autofrap/detection.py`, `autofrap/mask_utils.py`, `autofrap/nd2_helpers.py`, `autofrap/fake_nis.py` removed after bitsnpieces imports migrated to `autofrap.core.*`, `autofrap.io.*`, `autofrap.microscope.*`. TODO #8 closed.
* 20260914 – Package reorg step 1, simple-seg detector refactored into core, README draft + TODO updates.
* 20260910 – FakeNIS dry-run, cellpose server `--device`, clean Ctrl-C stop, experiment name/run dir guards, detector runtime parameters.
* 20260909 – ROI persistence probes, session-global ROIs discovery, opened-document wrappers live verified, ND template pre-flight check.
* 20260901 – Live 2×2 grid with real cellpose, stage position from ND2 metadata verified.

Full session history: `docs/STATUS_HISTORY.md`
NIS macro reference: `docs/NIS_REFERENCE.md`
Architecture / file map: `docs/ARCHITECTURE.md`
