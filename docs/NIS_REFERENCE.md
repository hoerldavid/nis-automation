# NIS Elements Macro Reference

## Macro execution pattern
All wrappers in `nis_util.py` follow:
1. Write temp `.mac` file
2. `nis_ar.exe -mw <file>` attaches to running GUI, blocks until done
3. Return values via `Int_SetKeyValue` / `Int_SetKeyString` into temp `.ini`

Gotchas
* Close `.mac` handle before calling `nis_ar`, else “Can't open file for reading”
* NIS locks failed `.mac` files → `PermissionError` on remove
* `Int_SetKeyValue` only numeric. Use `Int_SetKeyString` for strings
* File paths in macro calls must be absolute
* `CloseCurrentDocument(save='yes')` pops Save-As dialog and blocks macro

## Key wrappers

### Inspection / getters
* `get_camera_format` → (live, capture) format strings
* `get_position` → (x, y, z0, z1-or-None) µm
* `get_resolution` → (xres, yres, pixel_size, magnification)
* `get_camera_roi` → (enabled, (left, top, right, bottom) px)
* `get_rotation_matrix` → (a11, a12, a21, a22) camera→stage
* `get_cam_rotation` → (rotation, rotation2) deg
* `get_optical_confs` → list of conf names, e.g. `FRAPPA` index 0
* `get_nd_acq_tabs` → {tab: bool} active ND Acquisition tabs
* `get_opened_documents` → list of open document names, full path for saved files, GUI title otherwise
* `activate_document(name)` / `activate_opened_document(path/title)` → make already-open doc current, no disk reload
* `get_roi_ids` → list of visible ROI IDs via `GetROICount` + `GetROIIdFromIndex`

### Acquisition
* `run_current_nd_experiment(outfile, open_after=True)` → runs current ND experiment
  * `ND_DefineExperiment(-1,-1,-1,-1,-1,"<path>","",...)` keeps GUI dimensions, filename = save switch
  * `open_after=True` keeps result open; `False` closes result → dialog if unsaved
* `run_stimulation_experiment()` → runs current stimulation ND experiment
* `save_current_document(outfile)` → `ImageSaveAs(path, 15, 0)` ImType 15 = all layers, ImCompr 0 = lossless
* `close_current_document(save='discard'|'save'|'ask')` → `CloseCurrentDocument(2)` = discard without dialog

### Stage
* `set_position(x,y,z=None,pos_piezo=None)` → `StgMoveXY` blocks until arrival. `settle_s` removed.

### ROIs
* `add_polygon_roi(points, color)` → `CreatePolygonROI`, returns ROI ID
* `set_roi_type(roi_id, type)` → 0 standard, 1 background, 2 reference, 3 stimulation
  * Type 3 label prefix `S1:<n>` – group 1 of 3
* `delete_roi(roi_id)` → removes visible ROI. Always returns 0.
* `get_roi_info(roi_id)` → bbox, center, Feret, rotation, color. Color read-back always 0.

Important ROI behavior
* ROIs created with `CreatePolygonROI` are `ScopeType.Global` – session-global, not per-document, not stage-keyed
* Closing document does not remove them; any new acquisition picks them up
* `DeleteROI` only works with attached document current
* ROIs added after a doc's save are unsaved changes: `ActivateDocument` pops Save/Discard/Cancel
* ND experiment save embeds ROIs present at acquisition time

### FOV
`FOV = xres * pixel_size / magnification`

## To do X, use macro Y

* Save survey image after ND run → `save_current_document`
* Make an already-open ND2 current → `activate_opened_document`
* Create whole-cell ROI → `add_polygon_roi` + `set_roi_type(0)`
* Create stimulation ROI → `add_polygon_roi` + `set_roi_type(3)`
* Remove all ROIs from session → `get_roi_ids` → `delete_roi` for each
* Move stage → `set_position`
* Check survey template is single image → `get_nd_acq_tabs` → Time/XY/Large Image must be inactive

Full session notes in `docs/STATUS_HISTORY.md`.
