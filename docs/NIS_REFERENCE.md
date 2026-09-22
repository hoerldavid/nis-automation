# NIS Elements Macro Reference

## Macro execution pattern
All wrappers in `nis_util.py` follow:
1. Write temp `.mac` file
2. `nis_ar.exe -mw <file>` attaches to running GUI, blocks until done
3. Return values via `Int_SetKeyValue` / `Int_SetKeyString` into temp `.ini`

Macro function signatures: `nis_ar_help_html/` (the NIS macro CHM extracted to greppable HTML) is the source of truth.

### Macro batching
To reduce `nis_ar` overhead, `autofrap/microscope/nis.py` now uses a `MacroOp` pattern:
* `MacroOp(name, build(params, section), parse=None)`
* `build` returns a NIS snippet with `Int_SetKeyValue("__INI_PATH__", section, ...)`
* `parse` converts the ini section to a Python value; setters have `parse=None`
* `batch_run_macro(path, [(op, params), ...], timeout)` builds one macro with unique sections per op and runs it once.

Every `nis_ar -mw` call carries a roughly constant startup overhead, so batching several ops into a single call reduces it (verified live on the workstation; concrete timings in `docs/SESSION_HISTORY.md`).

Current ops:
* `_OP_POSITION` → `get_position`
* `_OP_RESOLUTION` → `get_resolution`
* `_OP_ND_ACQ_TABS` → `get_nd_acq_tabs`

Example:
```python
from autofrap.microscope.nis import batch_run_macro, _OP_POSITION, _OP_RESOLUTION
res = batch_run_macro(nis_exe, [(_OP_POSITION, {}), (_OP_RESOLUTION, {})])
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
* Gotcha: `ImageSaveAs` on the frozen live view (current doc `Frozen`) silently writes nothing — grab a single-frame ND acquisition instead

### Stimulation (GUI-template-driven)
* No programmatic define function exists: `_ND_CreateSequentialStimulationExp()` / `_ND_CreateSimultaneousStimulationExp()` only open the GUI window — the stimulation experiment must be pre-configured in the GUI
* No macro sets the stimulation output file (the `ND_DefineExperiment` filename trick does not apply to stimulation) → run with GUI save unset, then `save_current_document`
* The optical configuration **must be FRAPPA** before the stimulation run (select via `set_optical_configuration`), or the data is useless
* Sequential phase API (not wrapped; for later programmatic configuration): `ND_StimulationResetPhases()`, `ND_StimulationAppendPhase(type, interval_ms, duration_ms)` (−1 wait, 0 acquisition, 1 stimulation, 2 bleaching), `ND_StimulationCommand(macro)`, `ND_StimulationPoint(enabled, x, y)`; simultaneous variants: `ND_StimulationSimultaneousAcquisition(duration)`, `ND_StimulationSimultaneousStimulation(wait, interval, duration, manualStart)`; `StimulationDeviceSetActive(name)` (FRAPPA = `CLxStimulationDeviceFrappa`)
* Not wrapped yet: `MatchCameraROI("CLxStimulationDeviceFrappa")` (match camera FOV to stimulation device extent), `A1ApplyStimulationSettings()` (push a changed stimulation ROI to hardware mid-experiment)

### Stage
* `set_position(pos_xy, pos_z=None, pos_piezo=None, relative_xy=False, relative_z=False, relative_piezo=False)` → `StgMoveXY` blocks until arrival (relative flags give a Δ instead of an absolute position). `settle_s` removed.

### ROIs
* `add_polygon_roi(points, color)` → `CreatePolygonROI`, returns ROI ID
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

## To do X, use macro Y

* Save survey image after ND run → `save_current_document`
* Make an already-open ND2 current → `activate_opened_document`
* Create whole-cell ROI → `add_polygon_roi` + `set_roi_type(0)`
* Create stimulation ROI → `add_polygon_roi` + `set_roi_type(3)`
* Remove all ROIs → `delete_all_rois_in_current_document` (batched), or `get_roi_ids` → `delete_roi` for each
* Move stage → `set_position`
* Check survey template is single image → `get_nd_acq_tabs` → Time/XY/Large Image must be inactive
* Switch optical configuration → `set_optical_configuration` (FRAPPA required before stimulation)

Full session notes in `docs/SESSION_HISTORY.md`.
