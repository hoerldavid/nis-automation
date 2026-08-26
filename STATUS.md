# Status: NIS-Elements Automation Pipeline

_Last updated: after live-testing the new detection contract (cells + stimulation
mask, `b0ad1675`) — 2-cycle auto-FRAP run verified, TODO #4 (duration vs ROI area)
resolved. Before that: after the reorganization — generated code moved into
`autofrap/`, all acquired test data into `test_acquisitions/` (see File map)._

## Infrastructure (solid, verified)

- **Macro execution pattern**: temp `.mac` file → `nis_ar.exe -mw file` (attaches to
  running GUI, blocks until done) → return values via `Int_SetKeyValue` into a temp `.ini`.
  All `nis_util` functions follow this.
  - Gotcha: close the `.mac` handle (`ntf.close()`) **before** calling `nis_ar`, or the
    GUI reports "Can't open file for reading".
  - Gotcha: NIS keeps a lock on `.mac` files of *failed* macros → `os.remove` can raise
    `PermissionError` (handled with try/except in tests).
  - Gotcha: `Int_SetKeyValue(file, sec, key, value)` only takes a **numeric** value
    (despite the help page claiming `char *`) — passing a string literal aborts the
    macro in the GUI. Use `Int_SetKeyString(file, sec, key, buf)` for strings.
  - `Get_Filename(5, buf)` → full path of the currently opened image; for unsaved
    documents it returns the document title (e.g. `"ND Acquisition"`).
- **Help extracted**: `nis_ar_help_html/` — 3,701 HTML pages from the NIS macro CHM.
  Browsable and grep-able; source of truth for macro function signatures.
- **Dependencies**: `requirements.txt` gained `nd2`, `numpy`, `scikit-image`,
  `calmutils` (label remapping between cycles via
  `calmutils.segmentation.merge_label_slices`).
- **Bug fixed**: `get_cam_rotation` called `CameraGet_Cam0Flip` / `CameraGet_Cam0Rotate180`,
  which **do not exist** in this API → whole macro aborted at compile → empty ini →
  `KeyError: 'res'`. Now uses `CameraGet_Rotate` / `Camera_RotateGet`. Flip/180° info is
  part of the camera→stage calibration matrix (`get_rotation_matrix`).
- **Refactor** (TODO #5): all the temp-`.mac` → `nis_ar -mw` → temp-`.ini` read-back
  boilerplate is now in one shared helper, `_run_macro(path_to_nis, body, ini=False)`
  (returns the parsed `ConfigParser` when `ini=True`); the ini path in macro bodies is
  written as the `__INI_PATH__` placeholder (module constant `INI_PLACEHOLDER`) and
  substituted in the helper. Every wrapper is now ~5 lines of macro body + read-back.
  Verified byte-for-byte: all 33 generated macro bodies (every wrapper, incl. edge
  arg combos) are **identical** to what the pre-refactor code produced (checked with a
  one-off old-vs-new harness using deterministic fake temp files + no-op
  `subprocess.call`; the pre-refactor module was snapshotted for the comparison).
  Also made the temp-file cleanup tolerate `PermissionError` (NIS keeps a lock on the
  `.mac` of a *failed* macro — previously documented gotcha, now handled in one place).
- **Latent bugs fixed in the refactor** (neither was reachable in the current code
  paths, so behavior on the microscope is unchanged):
  - `get_position`: `tuple(map(float, res))` raised `TypeError` when no piezo is
    present (`z1` missing from the ini → `float(None)`); it now correctly returns
    `z1=None` as documented.
  - `set_position(pos_piezo=...)`: the piezo branch formatted `pos_z` instead of
    `pos_piezo` → `StgMovePiezoZ(None,0)` (macro compile error) whenever `pos_z` was
    None; now uses `pos_piezo`.

## Inspection (all `get_*` verified live)

| function | returns |
|---|---|
| `get_camera_format` | (live, capture) format strings |
| `get_position` | (x, y, z0, z1-or-None) µm |
| `get_resolution` | (xres, yres, pixel_size, magnification) |
| `get_camera_roi` | (enabled: bool, (left, top, right, bottom) px) camera ROI rectangle |
| `get_rotation_matrix` | (a11, a12, a21, a22) camera→stage "rotation and flip" |
| `get_cam_rotation` | (rotation, rotation2) deg |
| `get_optical_confs` | list of 16 names (incl. `FRAPPA`; was 13) |

**FOV formula**: `FOV = xres × pixel_size / magnification` (`get_fov_from_res`) —
e.g. **665.6 µm** at 20x / 13 µm pixels; 20260826 read-out (1024, 1024, 13.0 µm, 100x)
→ **133.1 µm**.

**Camera ROI** (chip sub-region for read-out speed): `get_camera_roi` reads back the
stored rectangle (right/bottom exclusive) plus the enabled flag — `ROIGet` returns the
rectangle even when the ROI is off, the flag distinguishes stored from active; when
active, `get_resolution` reports the ROI size so the FOV formula keeps working. Setting:
`ROISet(l,t,r,b)` + `ROIEnable(0|1)` (or the `_ROIDefine()` GUI dialog) — not wrapped
yet. 20260826 live probe: a stored central 512×512 (256,256,768,768) is present but
disabled (full 1024 read-out) — leftover from earlier manual testing by user and
colleagues.

## Part 1 — Grid scan (done, tested)

- `grid_positions(nis, nx, ny, spacing)` — grid centered on the **current stage position**;
  spacing in FOV units: 1 = touching FOVs, <1 = overlap, >1 = gap. Any nx/ny, odd or even.
- `run_current_nd_experiment(nis, outfile, open_after=True, progress_bar=True)`
  — runs the **GUI-configured** ND experiment as-is (`ND_RunExperiment` = "starts the
  current ND experiment").
  - Key validated finding: `ND_DefineExperiment(-1,-1,-1,-1,-1,"<path>","",...)` keeps all
    GUI dimensions (−1 = "keep current") and the **Filename parameter is the de facto
    save on/off switch**: it must be a **full path** (folder + filename — a bare
    filename runs but saves nothing); empty = no save (result stays open in the GUI);
    non-empty = saves there even with the GUI save checkbox off.
  - `open_after` semantics: `True` keeps the result open as the current document;
    `False` makes NIS **close** the result after the run — with no save destination
    this triggers a save/discard/cancel dialog (verified live). Hence default True;
    only use False when saving.
    Note: with a save destination the result document is closed after saving anyway
    (current doc reverts to the live view) — the survey file must be reopened via
    `open_image` before ROIs can be added to it (this is what `autofrap.py` does).
  - Closing documents programmatically: `CloseCurrentDocument(Save)` with
    0 = ask (dialog), 1 = save, **2 = discard without any user interaction**
    (verified live). Wrapped as `close_current_document(nis, save='discard')`.
- `overview_scan.py` — per position: `StgMoveXY` (blocking) → settle → capture → next;
  returns to the start position; returns `[(index, x, y, outfile, saved), ...]`.
  - 2×2 test: 4/4 saved (~7.4 s each), recorded metadata positions match commanded within
    ~2 µm, all channels intact.

## Part 2 — Detection (dummy done, contract updated)

- `detection.detect(nd2_file, channel=0)` → `(labels, stimulation_mask)` tuple;
  labels: 2D label map, same (y, x) shape as the image, 0 = background, 1..N = objects;
  stimulation_mask: 2D binary mask (same shape), True = areas eligible for photostimulation.
- Dummy detector: circle (label 1, center ⅓/⅓, r = min(h,w)/16) + rectangle
  (label 2, lower-right quadrant, 1/16 of the image per side). Both objects are
  fully included in the stimulation mask (`stim_mask = labels > 0`). Objects are
  kept small on purpose: FRAP bleaching is a laser scan, so stimulation time
  scales with ROI area; the two sizes differ by ~3x (12853 px vs 4096 px at
  1024×1024) on purpose, so a two-cycle run doubles as a duration-scaling test
  (see TODO #4). Real detector swaps in behind the same interface.
- `detection.label_to_polygon(labels, label_id, tolerance=2.0)` → simplified polygon
  vertices (x, y) pixels via `find_contours` + `approximate_polygon` (Douglas–Peucker).
  At 1024×1024: circle → 17 vertices, rectangle → 5.
- `detection.detect_stim_mask(labels, stimulation_mask, cell_id)` → binary mask
  where `(labels == cell_id) & stimulation_mask` (intersection of cell with stim mask).
- `detection.detect_polygon_stim_mask(labels, stimulation_mask, cell_id, tolerance=2.0)`
  → polygon for a cell's stimulation-eligible region only (same Douglas–Peucker
  simplification as `label_to_polygon`). Used by `autofrap.py` to draw the
  stimulation ROI.
- `read_channel` uses `nd2.ND2File.asarray()[channel]` (`read_frame` indexes time, not channels).

## ROIs in NIS (done, verified live)

- `open_image(nis, path)` — `ImageOpen`, makes the tile the current document (ROIs attach
  to the current image).
- `add_polygon_roi(nis, points, color)` — builds `double pts[2N]` + `CreatePolygonROI`,
  returns the ROI ID (read back via ini). `ROI_COLORS` dict: green/red/cyan/yellow/...
- `get_roi_info(nis, roi_id)` — `GetROIInfo` read-back (bbox, center, Feret, rotation, color).
- Live test: read-back bboxes matched the computed polygons exactly → pixel coordinate
  convention confirmed: (0,0) top-left, x right, y down; **no flip needed**.
- **Stimulation-relevant finds**:
  - `ChangeROIType(roi_id, type)` — 0 standard, 1 background, 2 reference, **3 stimulation**.
    **Verified live**: type 3 turns the ROI into a visible stimulation ROI;
    type 1 **hides the ROI** (hidden ROIs can reappear later, e.g. after closing a
    stimulation document / switching OC) — do not use.
    Return values are unreliable (1 and 0 both observed for plausible outcomes);
    verify via `GetROICount()` / the GUI instead.
  - `DeleteROI(roi_id)` — removes a *visible* ROI (verified live: count 3 → 1).
    Always returns 0 (also for unknown/hidden IDs — don't trust it).
  - `GetROICount()` / `GetROIIdFromIndex(n)` — enumerate **visible** ROIs only
    (track IDs at creation time; hidden ROIs are invisible to all of this).
  - `GetROIInfo` color read-back is always 0 (unreliable). Macro language: has
    `dword` but **no `unsigned int`**; `int l1, t1, r1, ...` failed to compile while
    the proven `int l, t, r, b, cx, cy, minf, maxf;` works — keep declarations simple.
  - `MatchCameraROI("CLxStimulationDeviceFrappa")` — match camera FOV to stimulation device extent.
  - `A1ApplyStimulationSettings()` — push a changed stimulation ROI to hardware mid-experiment.

## Part 3 — FRAP / photostimulation (in progress)

- **API map** (ND → Stimulation):
  - `_ND_CreateSequentialStimulationExp()` / `_ND_CreateSimultaneousStimulationExp()` —
    display the GUI window only; **no programmatic define function exists**.
  - `ND_RunSequentialStimulationExp()` / `ND_RunSimultaneousStimulationExp()` — "starts the
    **current** … Stimulation ND experiment" (same current-experiment mechanism, but see below).
  - Sequential phase API: `ND_StimulationResetPhases()`,
    `ND_StimulationAppendPhase(type, interval_ms, duration_ms)` (−1 wait, 0 acquisition,
    1 stimulation, 2 bleaching), `ND_StimulationCommand(macro)` (per-phase command),
    `ND_StimulationPoint(enabled, x, y)` (confocal point, pixels).
  - Simultaneous: `ND_StimulationSimultaneousAcquisition(duration)`,
    `ND_StimulationSimultaneousStimulation(wait, interval, duration, manualStart)`.
  - `StimulationDeviceSetActive(name)` — FRAPPA = `CLxStimulationDeviceFrappa`.
- **Destination file: no macro exists.** Global manual search found no stimulation output
  path setter. The `ND_DefineExperiment` filename trick does **not** apply to stimulation
  experiments (proven: Define with explicit filename + run → 33.7 s experiment ran, result
  open in GUI, nothing saved).
- **Decided workflow**: experimenters keep the GUI save **unset** → run experiment via macro
  → result stays open as the current document → `save_current_document(nis, outfile)`
  (wraps `ImageSaveAs(path, 15, 0)`; ImType 15 = all layers, ImCompr 0 = lossless,
  format from extension).
  Verified live: unsaved ND acquisition saved to `.nd2` in 1.4 s, GUI document rebinds
  to the file (see `get_current_document`), reads back via `nd2` with identical structure
  to an `ND_DefineExperiment` save (3 ch × 1024×1024 uint16).
- **Full manual FRAP loop verified end-to-end**:
  detect → `add_polygon_roi` → `ChangeROIType(id, 3)` → `SelectOptConf("FRAPPA")`
  → `ND_RunSequentialStimulationExp()` (42.5 s) → `save_current_document`
  → `test_stim.nd2` (12 time points × 1024×1024). Note: the optical conf **must** be
  FRAPPA before the stimulation run (earlier run without it produced no useful data).
- **Automated loop verified end-to-end** (`autofrap.py`): 2-cycle test with the dummy
  detector. Per cycle: survey via `run_current_nd_experiment` (~8 s) → detect →
  `next_stimulatable_cell` picks the smallest unstimulated label with nonzero stim
  pixels (skips cells with no stim-eligible area) → polygon from
  `detect_polygon_stim_mask` (ROI clipped to `(cell == id) & stim_mask`) →
  `set_roi_type(3)` → FRAPPA → `run_stimulation_experiment` (duration tracks ROI
  area, see TODO #4) → `save_current_document` → `delete_roi` → close both
  documents. Re-verified end-to-end on 20260826 after the detection-contract change
  (circle → 25.2 s, rectangle → 18.2 s; 40 frames × 1024×1024 each; StimulationROI
  saved in both FRAP files with the right polygon and center).
  Verified: correct object each cycle (circle, then rectangle); each FRAP file contains
  the ROI with `InterpType.StimulationROI` in the nd2 `rois` metadata (deletion happens
  *after* saving, so it stays in the file for downstream analysis); GUI left clean.
- **`next_cell()` kept** for backward compatibility; `next_stimulatable_cell()` is the
  preferred function when a stimulation mask is available (integrates the mask check
  into the search loop — cells with zero stim pixels are skipped automatically).
- **New detection helpers** (`detect_stim_mask`, `detect_polygon_stim_mask`) are thin
  wrappers around the existing `find_contours`/`approximate_polygon` pipeline, applied
  to the intersection mask `(labels == cell_id) & stimulation_mask`.

## TODO (next days)

1. ~~Fix fallout from the folder move~~ — **import fix done**: `autofrap/autofrap.py`
   and `autofrap/autofrap_bitsnpieces/overview_scan.py` now prepend the repo root to
   `sys.path` (verified: top-level imports resolve under real script-run semantics).
   AppleDouble `._*.py` junk deleted. **Deferred**: stale hardcoded output dirs
   (`overview_scan.py` → `<root>/overview/`, `autofrap.py` → `autofrap/autofrap_out/`)
   — everything at this point is ephemeral testing, so re-running recreating the old
   folders is acceptable for now; point them at `test_acquisitions/` (or make them CLI
   args) once real data starts accumulating.
2. Real detector to replace the dummy (interface: `image -> (labels, stimulation_mask)`).
3. ~~Investigate: stimulation run duration did not scale with ROI area~~ —
   **resolved (20260826, 2-cycle run with dummy areas differing ~3x)**: the earlier
   anomaly was leftovers from previous tests on the microscope (a ROI still in
   memory, etc.). Test result: circle (12853 px) → 25.2 s, rectangle (4096 px) →
   18.2 s. Fits `T ≈ 15 s fixed overhead + 0.8 ms/px × area`: the area-dependent
   part (≈10.3 s vs ≈3.3 s) scales exactly with the area ratio (3.1x).
4. ~~Refactor `nis_util.py`: shared helper for the temp-`.mac` → `nis_ar -mw` →
   temp-`.ini` boilerplate~~ — **done**: `_run_macro(path_to_nis, body, ini=False)`
   + `__INI_PATH__` placeholder (see Infrastructure notes). Verified byte-for-byte
   against the pre-refactor code for all 33 macro bodies; also fixed two latent
   bugs found along the way (`get_position` without piezo, `set_position(pos_piezo=)`).
   Note: `get_optical_confs`'s macro had `sprintf(&buf, "conf%i", "i" )` — a
   string literal where the loop int was expected (worked only because the NIS
   macro compiler is permissive about it). Fixed to `i` on 20260826 (snapshot
   `nis_util_old.py` updated to match, equivalence check still 33/33),
   re-verified live: all 16 names still returned.
   **Live-verified 20260826** at the microscope: all read-only `get_*` wrappers +
   `set_position` XY round-trip (±2 µm, back within 0.1 µm) + piezo round-trip
   (±1 µm, exact) pass — `autofrap/autofrap_bitsnpieces/test_nis_util_live.py`.
5. Cleanup: test data is now organized in `test_acquisitions/` and kept (incl.
   `test_stim.nd2` and the `autofrap_out/` FRAP files, which contain saved
   stimulation ROIs). Remaining junk at the root: `__pycache__/`,
   `.ipynb_checkpoints/`, `.virtual_documents/`, `pi-session-*.html` (pi session
   logs) — the `._*.py` AppleDouble files in `autofrap/autofrap_bitsnpieces/`
   (leftovers from the macOS move) have been deleted.
6. Pixel ↔ stage coordinate transform for per-tile ROIs (calibration matrix from
   `get_rotation_matrix` + pixel size; `get_roi_info` center as a shortcut) — needed
   to move the stage to a detected object before stimulating. **Low priority**:
   centering the object before stimulating was considered, but the current
   approach (no stage move, ROI drawn directly on the survey image) works fine.

## File map

Layout after the reorganization: generated pipeline code under `autofrap/`
(test-like bits under `autofrap/autofrap_bitsnpieces/`), all acquired test data
under `test_acquisitions/`, and the NIS macro wrappers at the root (`nis_util.py`).

### Generated pipeline code

| file | purpose |
|---|---|
| `nis_util.py` | NIS macro wrappers on top of the shared `_run_macro` helper: `get_*`, `get_current_document`, `save_current_document`, `close_current_document`, `set_position`, `run_current_nd_experiment`, `run_stimulation_experiment`, `grid_positions`, `open_image`, `add_polygon_roi`, `set_roi_type`, `delete_roi`, `get_roi_count`, `get_roi_info`, `NDAcquisition` (from-scratch builder), `export_nd2_to_tiff`, ... |
| `autofrap/autofrap.py` | Part 3: auto-FRAP loop (survey → detect → stimulate next unused cell → repeat) |
| `autofrap/detection.py` | Part 2: `detect` (returns `(labels, stimulation_mask)` tuple),
`dummy_detect_objects`, `read_channel`, `label_to_polygon`, `detect_stim_mask`,
`detect_polygon_stim_mask` |
| `autofrap/autofrap_bitsnpieces/overview_scan.py` | Part 1: grid scan script (move + capture per position) |
| `autofrap/autofrap_bitsnpieces/inspect_microscope.ipynb` | notebook walking through the `get_*` functions + FOV |
| `autofrap/autofrap_bitsnpieces/stimulation_loop.py` | colleague's sketch (source of the label-remapping logic, now ported into `autofrap.py`) |
| `autofrap/autofrap_bitsnpieces/test_merge_label_slices.py` | colleague's sketch: `merge_label_slices` test |
| `autofrap/autofrap_bitsnpieces/split_mask_along_axis_equal_area.py` | mask utility: split a label mask into two equal-area halves along an axis (skimage) |
| `autofrap/autofrap_bitsnpieces/test_stim_save.py` | one-off test script (save-current-document probe; cleanup candidate) |
| `autofrap/autofrap_bitsnpieces/test_nis_util_refactor.py` | equivalence check for the `_run_macro` refactor: all generated `.mac` bodies byte-identical to the frozen pre-refactor snapshot + round-trip / cleanup tests (run: `python autofrap/autofrap_bitsnpieces/test_nis_util_refactor.py`) |
| `autofrap/autofrap_bitsnpieces/test_nis_util_live.py` | live smoke test for the refactored wrappers (run at the microscope): all read-only `get_*` + `set_position` XY/piezo round-trip |
| `autofrap/autofrap_bitsnpieces/nis_util_old.py` | **frozen snapshot** of the pre-refactor `nis_util.py` (input of the equivalence check). The check served its purpose (33/33 byte-identical + live re-verification); the snapshot is no longer kept in sync with deliberate macro-body changes and will be removed (with `test_nis_util_refactor.py`) once the code stabilizes |

### Test data (`test_acquisitions/`)

| path | contents |
|---|---|
| `test_acquisitions/overview/` | 2×2 grid-scan test, 4 files (20260819) |
| `test_acquisitions/autofrap_out/<stamp>/` | auto-FRAP runs: `cNN_survey.nd2` + `cNN_frap.nd2` per cycle (three 2-cycle runs, 20260824); FRAP files contain the saved stimulation ROI in the nd2 `rois` metadata |
| `test_acquisitions/test_*.nd2` | earlier one-off artifacts: `test_current`, `test_current3`, `test_save_on`, `test_saveas`, `test_stim` |

### Pre-existing project files (not part of this work)

| file | purpose |
|---|---|
| `annotation.py`, `automation.py`, `resources.py`, `start_wing_scanner.bat`, `res/` | wing-scanner project code + calibration JSONs / `dummy.nd2` |
| `simple_detection.py` | pre-existing wing detection (bbox-based, skimage) |
| notebooks: `NIS_Macro_Acquisition.ipynb`, `simple_overview.ipynb`, `create_overview_calibration.ipynb`, `manual_roi_annotation_test.ipynb`, `simple_intelligent_acquisition.ipynb`, `transform_test.ipynb`, `wing_keypoint_detect.ipynb` | pre-existing / exploratory notebooks |
| `nis_ar_help_html/` | extracted NIS macro reference (HTML) |

Junk at the root: `__pycache__/`, `.ipynb_checkpoints/`, `.virtual_documents/`,
`pi-session-*.html` (pi session logs).
