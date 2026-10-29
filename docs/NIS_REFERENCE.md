# NIS Elements Macro Reference

## Macro execution pattern
All wrappers in `autofrap/microscope/nis.py` follow:
1. Write temp `.mac` file
2. `nis_ar.exe -mw <file>` attaches to running GUI, blocks until done
3. Return values via `Int_SetKeyValue` / `Int_SetKeyString` into temp `.ini`

Macro function signatures: the NIS macro function reference (CHM from a NIS installation, extracted to greppable HTML — see `nis_manual/README.md` how to obtain/extract it; extracted copy at `nis_ar_help_html/`) is the source of truth.

### Macro batching
To reduce `nis_ar` overhead, `autofrap/microscope/nis.py` now uses a `MacroOp` pattern:
* `MacroOp(name, build(params, section), parse=None)`
* `build` returns a NIS snippet with `Int_SetKeyValue("__INI_PATH__", section, ...)`
* `parse` converts the ini section to a Python value; setters have `parse=None`
* `batch_run_macro(path, [(op, params), ...], timeout)` builds one macro with unique sections per op and runs it once.

Every `nis_ar -mw` call carries a roughly constant startup overhead, so batching several ops into a single call reduces it (verified live on the workstation; concrete timings in `docs/SESSION_HISTORY.md`).

Current ops (single-call wrapper in parentheses):
* getters: `OP_POSITION` (`get_position`), `OP_RESOLUTION` (`get_resolution`), `OP_ND_ACQ_TABS` (`get_nd_acq_tabs`)
* stage / config: `OP_SET_POSITION` (`set_position`), `OP_SET_OPTICAL_CONFIGURATION` (`set_optical_configuration`)
* acquisition: `OP_RUN_STIMULATION_EXPERIMENT` (`run_stimulation_experiment`)
* ROIs: `OP_ADD_POLYGON_ROI` (`add_polygon_roi`), `OP_CREATE_AND_SET_STIM_ROI` (`add_polygon_roi_and_set_stim_type`; create + `ChangeROIType(3)` in one op), `OP_DELETE_ALL_ROIS_IN_CURRENT_DOCUMENT` (`delete_all_rois_in_current_document`)
* documents: `OP_ACTIVATE_DOCUMENT` (`activate_document`), `OP_SAVE_CURRENT_DOCUMENT` (`save_current_document`), `OP_CLOSE_CURRENT_DOCUMENT` (`close_current_document`), `OP_CLOSE_ALL_DOCS` (`close_all_docs`)
* misc: `OP_CHECKPOINT` (`checkpoint`)

Example:
```python
from autofrap.microscope.nis import batch_run_macro, OP_POSITION, OP_RESOLUTION
res = batch_run_macro(nis_exe, [(OP_POSITION, {}), (OP_RESOLUTION, {})])
# res == {'position_0': (x,y,z0,z1), 'resolution_1': (xres,yres,siz,mag)}
```

Implementation notes
* NIS macro language requires all variable declarations to appear before any executable statements. When multiple `MacroOp`s are concatenated, `batch_run_macro` hoists all `int/double/char/dword/byte/word/float` declarations to the top and automatically renames variables to `section_var` to avoid collisions across ops.
* The macro language has `dword` but no `unsigned int`; keep variable declarations simple (complex declarations have failed to compile).

Gotchas
* Close `.mac` handle before calling `nis_ar`, else “Can't open file for reading”
* NIS locks failed `.mac` files → `PermissionError` on remove
* `Int_SetKeyValue` only numeric. Use `Int_SetKeyString` for strings
* File paths in macro calls must be absolute
* `CloseCurrentDocument(save='yes')` pops Save-As dialog and blocks macro
* `CameraGet_Cam0Flip` / `CameraGet_Cam0Rotate180` do not exist in this API (the macro aborts at compile); `get_cam_rotation` uses `CameraGet_Rotate` / `Camera_RotateGet` instead
* `nis_ar -mw` exits non-zero even on success — observed constant 0xFFFFFF36 (= -202) on every call, successful ones included (live, 20261019). Don't use the exit code as a success signal; a failed macro shows up as a timeout or a missing ini section/key

## Key wrappers

### Inspection / getters
* `get_camera_format` → (live, capture) format strings
* `get_position` → (x, y, z0, z1-or-None) µm
* `get_resolution` → (xres, yres, pixel_size, magnification)
* `get_camera_roi` → (enabled, (left, top, right, bottom) px); setting via `ROISet(l,t,r,b)` + `ROIEnable(0|1)` — not wrapped yet
* `get_rotation_matrix` → (a11, a12, a21, a22) camera→stage
* `get_cam_rotation` → (rotation, rotation2) deg
* `get_optical_confs` → list of conf names, e.g. `FRAPPA` index 0
* `get_nd_acq_tabs` → {tab: bool} active ND Acquisition tabs; queries the current experiment definition, no document needed
* `get_opened_documents` → list of open document names, full path for saved files, GUI title otherwise
* `activate_document(name)` / `activate_opened_document(path/title)` → make already-open doc current, no disk reload
* `get_current_document` → name of the current document via `Get_Filename(5, buf)`; for unsaved documents this is the document title (e.g. `ND Acquisition`), not a path
* `get_roi_ids` → list of visible ROI IDs via `GetROICount` + `GetROIIdFromIndex`

### Acquisition
* `run_current_nd_experiment(outfile, open_after=True)` → runs current ND experiment
  * `ND_DefineExperiment(-1,-1,-1,-1,-1,"<path>","",...)` keeps GUI dimensions; the filename is the de-facto save on/off switch and must be a **full path** — empty runs the experiment but saves nothing
  * `open_after=True` keeps result open; `False` closes result → dialog if unsaved. With a save destination the result document is closed after saving anyway (current doc reverts to the live view)
* `run_stimulation_experiment()` → runs current stimulation ND experiment
* `save_current_document(outfile)` → `ImageSaveAs(path, 15, 0)` ImType 15 = all layers, ImCompr 0 = lossless
* `close_current_document(save='discard'|'save'|'ask')` → `CloseCurrentDocument(2)` = discard without dialog
* `open_image(path)` → `ImageOpen`, makes the file the current document
* Gotcha: `ImageSaveAs` on the frozen live view (current doc `Frozen`) silently writes nothing (single live observation, 20260901 — no controlled A/B probe yet) — grab a single-frame ND acquisition instead

### Stimulation (GUI-template-driven)
* No programmatic define function exists: `_ND_CreateSequentialStimulationExp()` / `_ND_CreateSimultaneousStimulationExp()` only open the GUI window — the stimulation experiment must be pre-configured in the GUI
* No macro sets the stimulation output file (the `ND_DefineExperiment` filename trick does not apply to stimulation) → run with GUI save unset, then `save_current_document`
* The optical configuration **must be FRAPPA** before the stimulation run (select via `set_optical_configuration`), or the data is useless
* Sequential phase API (not wrapped; for later programmatic configuration): `ND_StimulationResetPhases()`, `ND_StimulationAppendPhase(type, interval_ms, duration_ms)` (−1 wait, 0 acquisition, 1 stimulation, 2 bleaching), `ND_StimulationCommand(macro)`, `ND_StimulationPoint(enabled, x, y)`; simultaneous variants: `ND_StimulationSimultaneousAcquisition(duration)`, `ND_StimulationSimultaneousStimulation(wait, interval, duration, manualStart)`; `StimulationDeviceSetActive(name)` (FRAPPA = `CLxStimulationDeviceFrappa`)
* **No getter for stimulation phases** (live-confirmed 20261027): `ND_GetTimeLapsePhaseCount()` / `ND_GetTimePhaseSchedule(i, ...)` read the ND Acquisition Time-tab phase table — a *different* table from the sequential stimulation phases (ground truth: after appending 3 known stimulation phases, the getters still reported the timelapse tab's phase). The manual's stimulation API is setters/runners only, and the `#STRINGS` binary inventory contains no undocumented stimulation getter either. Consequence: phase timing must be tracked on the Python side from our own `ND_StimulationAppendPhase` configuration
* `ND_GetLambdaChannel(i, ...)` **crashes NIS Elements on every call — do not use** (live-confirmed 20260904 + 20261027): the signature matches the manual verbatim (9 params), but the call kills the NIS process even when a channel-0 result arrives (the ini writes complete before the process dies; the next macro call then spawns a fresh NIS instance, which times out). Applied ND state makes no difference. Bug in this NIS build's getter implementation — channel ground truth is our own `ND_AppendLambdaChannel` configuration
* Not wrapped yet: `MatchCameraROI("CLxStimulationDeviceFrappa")` (match camera FOV to stimulation device extent), `A1ApplyStimulationSettings()` (push a changed stimulation ROI to hardware mid-experiment)

### Stage
* `set_position(pos_xy, pos_z=None, pos_piezo=None, relative_xy=False, relative_z=False, relative_piezo=False)` → `StgMoveXY` blocks until arrival (relative flags give a Δ instead of an absolute position). `settle_s` removed.

### ROIs
* `add_polygon_roi(points, color)` → `CreatePolygonROI`, returns ROI ID
* `add_polygon_roi_and_set_stim_type(points, color)` → same + `ChangeROIType(id, 3)` in the same macro call
* `set_roi_type(roi_id, type)` → 0 standard, 1 background, 2 reference, 3 stimulation
  * Type 3 label prefix `S1:<n>` – group 1 of 3
  * Return values unreliable (1 and 0 both observed) — verify via `get_roi_count` / the GUI
* `delete_roi(roi_id)` → removes visible ROI. Always returns 0.
* `get_roi_info(roi_id)` → bbox, center, Feret, rotation, color. Color read-back always 0.

Pixel coordinates: (0,0) top-left, x right, y down — no flip needed (verified live: read-back bboxes matched the computed polygons exactly).

Important ROI behavior
* ROIs created with `CreatePolygonROI` are `ScopeType.Global` – session-global, not per-document, not stage-keyed
* Closing document does not remove them; any new acquisition picks them up
* `DeleteROI` only works with attached document current
* ROIs added after a doc's save are unsaved changes: `ActivateDocument` pops Save/Discard/Cancel
* ND experiment save embeds ROIs present at acquisition time
* NIS stores closed polygons without the repeated closing vertex (one point fewer than the pixel-space polygon sent)

### Housekeeping / other
* `get_roi_count` → number of visible ROIs (`GetROICount`)
* `delete_all_rois_in_current_document()` → batched `DeleteROI` loop (backward iteration), one macro call
* `close_all_docs(save_flag=2)` → close all open documents (default: discard)
* `checkpoint(key='ok', value=1)` → writes an ini key via a macro (liveness / progress marker)
* `do_autofocus(step_coarse, step_fine, focus_criterion, focus_with_piezo)` → two-pass adaptive stage focus (`StgFocusAdaptiveTwoPasses`) + freeze
* `do_large_image_scan(save_path, left, right, top, bottom, ...)` → `Stg_LargeImageScanArea` stitched scan
* `backup_optical_configurations(backup_path)` → `BackupOptConf` exports all optical configurations as XML
* `export_nd2_to_tiff(nd2_file, out_dir, combine_t/yx/z/c)` → converts an ND2 to TIFFs via NIS
* `NDAcquisition(outfile)` → from-scratch ND experiment builder (points, channels, z-range → compiled macro)

### FOV
`FOV = xres * pixel_size / magnification`

## Pixel ↔ stage transform (single images)

For a single acquired image, the pixel → stage map is affine — a similarity (uniform scale + one axis flip + rotation) plus a translation. The linear part is a property of the imaging chain (camera port + optical configuration) and is identical for any stage position; only the anchor is per-image:

```
stage(px) = p_acq + s · R^T · F · (px − c)
```

Components:

* `px` — pixel coordinates `(col, row)`; (0,0) is top-left, x grows right, y grows down (the same convention as ROI polygons, see § ROIs).
* `s` — effective pixel size in µm/px: `pixel_size / magnification` from `get_resolution` (equivalently `fov / xres` via `get_fov_from_res`). Depends on objective, binning and camera — re-read per optical configuration.
* `R` — the camera→stage calibration matrix from `get_rotation_matrix` (`Get_CalibrationAngleMatrix`), **applied transposed**: the NIS manual documents the application as the row-vector formula `x' = x*A11 + y*A12`, `y' = x*A21 + y*A22`, i.e. `(x', y') = (x, y) · A`; in column-vector (numpy) form that is `A^T @ p`, not `A @ p`. Applying it untransposed rotates the map by the wrong angle — the misplacement grows linearly with distance from the anchor pixel. The matrix is per camera (`ActiveCamera` argument) and per microscope unit — re-read it when the camera or optical configuration changes. The manual documents it as the "rotation and flip" of the optical path: `det(R) = −1` means the optical path itself contains a mirror (the flip situation then changes — re-verify rather than assume `F`).
* `F` — the axis flip between pixel coordinates and the camera frame (pixel y grows down, camera-frame y grows up): `diag(1, −1)`. **There is no programmatic way to obtain this flip**: no macro getter exists — the Camera Light Path "Flip" setting (Acquire → Camera Light Path → Camera Rotation and Flipping) is a calibration-time GUI checkbox, not runtime-readable, and `CameraGet_Cam0Flip` does not exist in this API (see § Gotchas). It was established by a one-time live measurement on this unit (details: `docs/SESSION_HISTORY.md`, 20261019). The manual notes that NIS flips flipped camera data internally "so that all ports give the same, non-flipped data", which indicates the remaining flip is the fixed pixel/camera axis convention rather than a per-port setting — but that is documentation, not a getter. If the camera, port or light-path settings ever change, check the GUI settings and re-verify the flip.
* `c` — the anchor pixel: the image center, `(xres/2, yres/2)`. If a camera ROI is active, the saved image is the ROI — the natural assumption is that `c` is then the ROI center in saved-image pixel coordinates (`get_camera_roi` returns the rectangle; `get_resolution` already reports the ROI size when enabled), but **the anchor with an active camera ROI is untested** — verify before relying on it.
* `p_acq` — the stage position in µm (the `StgGetPosXY` / `StgMoveXY` frame) **at the moment the image was acquired**. Anchor rule (live-verified, 20261019): the stage position corresponds to the image-center pixel — the image center sits at stage coordinate `p_acq`. In the pipeline this is simply the commanded target of the verified move (`move_stage_with_retry` checks the reached position; the survey is acquired immediately after), so no nd2 metadata is needed. The position stored in the saved nd2 is equivalent but goes stale if the stage moved between acquisition and save; a `get_position` read taken later belongs to a different image.

Notes:

* **The closed-form formula itself is directly live-verified (20261027)**: predicted stage coords for 7 pixels across the FOV vs. the GUI right-click readout (whole-µm precision), agreement at two different stage positions including after a stage move — all within the 1 µm display precision. Works with the live view as the anchor context (continuously acquired at the current position; `get_position` is a valid `p_acq` as long as the stage does not move). Re-verification probe script: `autofrap/autofrap_bitsnpieces/check_pixel_stage_transform.py`. On this setup `det(R) = +1` with R ≈ −I (~180° rotation); the flip comes solely from `F`.
* The result is in the `StgMoveXY` frame and feeds straight back into `set_position` — the programmatic inverse of the GUI's "Move this point to center".
* On this setup stage x decreases left-to-right and stage y increases top-to-bottom.
* Stitched large images (`Stg_LargeImageScanArea`) are **not covered** here: their anchor (first-tile corner anchoring; what position(s) the stitched nd2 stores) is unverified on this unit — see `docs/SESSION_HISTORY.md` (20260929 legacy retro-analysis, 20261019 live-test attempt).

## To do X, use macro Y

* Save survey image after ND run → `save_current_document`
* Make an already-open ND2 current → `activate_opened_document`
* Create whole-cell ROI → `add_polygon_roi` + `set_roi_type(0)`
* Create stimulation ROI → `add_polygon_roi_and_set_stim_type` (one macro call), or `add_polygon_roi` + `set_roi_type(3)`
* Remove all ROIs → `delete_all_rois_in_current_document` (batched), or `get_roi_ids` → `delete_roi` for each
* Move stage → `set_position`
* Check survey template is single image → `get_nd_acq_tabs` → Time/XY/Large Image must be inactive
* Switch optical configuration → `set_optical_configuration` (FRAPPA required before stimulation)

Full session notes in `docs/SESSION_HISTORY.md`.
