# Status: NIS-Elements Automation Pipeline

_Last updated: **`nis_util.py` TODO cleanup** (no microscope needed): dead
color-camera-crop macro snippet + commented-out `set_camera` stub removed;
`gen_grid` moved to `grid_utils.py` (refs in `automation.py` +
`NIS_Macro_Acquisition.ipynb` updated); `_quote` inlined into `_run_macro`;
`get_fov_from_res` unpacks its input; `grid_positions` moved to
`autofrap/autofrap.py` as a pure `grid_positions(position, fov, nx, ny,
spacing)` (callers `autofrap_grid` + `overview_scan.py` updated, one fewer
stage read per grid run); `ROI_COLORS` now hex literals. Verified:
`test_nis_util_refactor.py` still 33/33 byte-identical + all round-trip
tests pass; moved functions numerically identical to the pre-cleanup code
on 200 random arg combos each. The three remaining `nis_util.py` TODOs are
live checks (TODO #12). Before that: **multi-FOV grid runner** (`autofrap_grid` in
`autofrap/autofrap.py`) — loops the verified single-FOV `autofrap()` over
stage positions from `grid_positions`, one sub-directory per FOV, settle
after each move, return to start, per-FOV failure isolation. **Live
verified 20260901**: 2×2 grid (spacing 1.0), one cell per FOV, real
cellpose — 4/4 FOVs (21/17/18/13 objects, different fields per FOV),
survey 9.2–9.9 s, detection ~2.1 s, stimulation 12.6–13.3 s, returned to
start; 8 files verified (2-ch surveys + 12-frame FRAPs, each FRAP with
`StandardROI` + `StimulationROI`); open issue: reading the stage position
from the *survey* nd2 metadata (TODO #11). Before that: real detector in
the loop — cellpose (cpdino-vitb) on a
remote V100 GPU server (`cellpose_server.py`, FastAPI, np.save wire format),
client in `detection.py`; `detect()` gained `detector` / `relabel` params,
border-touching objects are discarded (`clear_border`), stim mask moved into
`default_stimulation_mask`; `autofrap()` takes a `detection_fun` partial and
defaults to the remote cellpose detector. **Verified 20260901 at the
microscope: 2-cycle run on real DAPI-stained nuclei** (26→22 / 27→23 objects
after border discard, different nucleus per cycle, both ROI types saved,
14.1 / 13.9 s stimulations). Before that: auto-FRAP loop now saves two ROIs
per cell (whole cell + stimulation half) and the dummy stim mask is the left
half of each object — verified 20260826; camera ROI detection added
(`get_camera_roi`). Before that: after live-testing the new detection contract
(cells + stimulation mask, `b0ad1675`) — 2-cycle auto-FRAP run verified,
TODO #3 (duration vs ROI area) resolved. Before that: after the
reorganization — generated code moved into `autofrap/`, all acquired test
data into `test_acquisitions/` (see File map)._ 

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
  `calmutils.segmentation.merge_label_slices`), `fastremap`
  (fast label-remap kernels; already present transitively via calmutils,
  now an explicit dep for `detection.shuffle_labels`), `requests`
  (cellpose client). **GPU note (20260901)**: torch on the microscope PC is
  CPU-only — the Quadro K2200 (Kepler, sm_35, 2015 driver 353.53) is
  unsupported by any PyTorch that runs on Python 3.14, and no bigger GPU fits
  (no PSU power connectors); detection therefore runs on a **remote Linux GPU
  server** (10.163.69.12, Tesla V100-PCIE-32GB) via `cellpose_server.py`
  (server machine: `pip install fastapi uvicorn cellpose` + CUDA torch;
  weights cached locally).
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
- **TODO cleanup in `nis_util.py`** (workstation-free part, after the 20260901
  grid run): removed the dead color-camera-crop macro snippet + commented-out
  `set_camera` stub (`is_color_camera` kept — still used by the old wing-scanner
  `automation.py`); `gen_grid` moved to root-level `grid_utils.py` (pure
  geometry, not NIS-specific; refs updated in `automation.py` +
  `NIS_Macro_Acquisition.ipynb`); `_quote` removed (its two uses in
  `_run_macro` inlined as an f-string — the `nis_ar` command line is
  byte-identical); `get_fov_from_res` unpacks its `(xres, yres, pixel_size,
  magnification)` input; `grid_positions` moved to `autofrap/autofrap.py` as a
  pure `grid_positions(position, fov, nx, ny, spacing)` — `autofrap_grid` and
  `overview_scan.py` now fetch position/FOV themselves (and `autofrap_grid`
  saves one `nis_ar` round trip: the position is read once, not twice);
  `ROI_COLORS` uses hex literals (same int values → macro bodies unchanged;
  comment fixed RGB→BGR, matching the live-verified green read-back).
  Verified: `test_nis_util_refactor.py` still 33/33 byte-identical macro
  bodies + all round-trip/behaviour tests pass; the moved functions give
  identical results to the pre-cleanup code on 200 random arg combos each.

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

## Part 2 — Detection (done: cellpose client/server; dummy kept for testing)

- `detection.detect(nd2_file, channel=0, detector='dummy'|'cellpose-remote',
  server_url=None, relabel='distance'|'shuffle'|None, **detector_kwargs)`
  → `(labels, stimulation_mask)` tuple; labels: 2D label map, same (y, x)
  shape as the image, 0 = background, 1..N = objects; stimulation_mask: 2D
  binary mask (same shape), True = areas eligible for photostimulation.
  Pipeline: read channel → detector (labels only) → **border discard** →
  **relabel** → stim mask.
- **Cellpose detector (the real one — TODO #2 done 20260901)**:
  `remote_detect_objects(image, server_url, timeout=1800, **eval_kwargs)`
  POSTs the channel as raw `np.save` bytes (`application/x-numpy`,
  `allow_pickle=False` both ends; ~2 MB for 1024² uint16, no compression)
  to `cellpose_server.py` and receives the label map back the same way.
  Server: FastAPI, model loaded once at startup (`cpdino-vitb`, the smallest
  CP4 generalist — SAM/DINOv3 backbones; explicit
  `device=torch.device('cuda'|'cpu')`), requests serialized by a lock;
  `POST /detect` also takes the main `model.eval()` knobs as query params
  (`diameter`, `min_size`, `cellprob_threshold`, `flow_threshold`,
  `max_size_fraction` — defaults are cellpose's own), `GET /health` reports
  `cuda` + GPU name. No auth, plain HTTP: lab network only (or SSH-tunnel it).
  - **Timing** (1024² DAPI): 293.6 s on the PC's CPU → **2.1–2.6 s on the
    V100** (~109×). Segmentation identical CPU vs GPU (15 objects, per-label
    areas match within ±2 px float noise).
  - **Gotchas found while wiring**: current FastAPI reads a bare `bytes`
    endpoint annotation as a *query* param (422) → use `Body(...)`;
    scikit-image 0.26 moved `relabel_sequential` from `skimage.measure` to
    `skimage.segmentation` and it now returns 3 values
    `(labels, forward_map, inverse_map)`; `ImageSaveAs` on a frozen live view
    (current doc `"Frozen"`) silently writes nothing — grab a single-frame ND
    acquisition instead (see Part 3 note).
- **Border discard** (in `detect`, after detection, before relabelling):
  `clear_border(labels)` — removes the **whole label** if it touches the
  image border (partially imaged cells must not be stimulated) — then
  `relabel_sequential` closes the numbering gaps (gap-free 1..N). 20260901
  run: 26→22 and 27→23 objects in a dense FOV.
- **Relabelling modes** (`relabel` param, after border discard):
  `'distance'` (default) — `relabel_by_distance`, 1..N by increasing centroid
  distance to image center (optical axis first); `'shuffle'` —
  `shuffle_labels`, random permutation (no raster-order bias); `None` — as-is.
- **Stim mask**: `default_stimulation_mask(labels)` — the **left half** of
  each object (equal-area split via `split_mask_along_axis_equal_area`,
  ported from
  `autofrap/autofrap_bitsnpieces/split_mask_along_axis_equal_area.py`),
  mimicking a real FRAP experiment in which part of the cell is bleached and
  diffusion from the rest is recorded. Computed **in `detect()`** from the
  final labels; the detector functions themselves return labels only.
- Dummy detector (`detector='dummy'`, no server needed): circle (label 1,
  center ⅓/⅓, r = min(h,w)/16) + rectangle (label 2, lower-right quadrant,
  1/16 of the image per side); objects kept small on purpose so runs double
  as a stimulation-duration-vs-area check.
- `detection.shuffle_labels(labels, seed=None)` → label map with 1..N randomly
  permuted (background stays 0), via `fastremap.remap` with a permutation dict —
  detectors number objects in raster order (top-left first), which would bias
  order-dependent processing when only part of the FOV is imaged.
  (scikit-image has no shuffle — only `relabel_sequential` — hence fastremap.)
- `detection.relabel_by_distance(labels, reference=None)` → relabels 1..N by
  increasing centroid distance to a reference point (centroids from skimage
  `regionprops`; remap via `fastremap.remap`). Works for 2D (y, x) and 3D
  (z, y, x) label maps (numpy-based, one coordinate per axis). Default
  reference = image center (optical axis: least distortion/vignetting →
  process first). Background stays 0.
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
- **Automated loop verified end-to-end** (`autofrap.py`): per cycle: survey via
  `run_current_nd_experiment` (~10 s) → detect → `next_stimulatable_cell` picks the
  smallest unstimulated label with nonzero stim pixels (skips cells with no
  stim-eligible area) → **two ROIs** on the survey image: the whole cell
  (`label_to_polygon`, standard type — saved for downstream analysis) and the
  stimulation region (`detect_polygon_stim_mask`, clipped to
  `(cell == id) & stim_mask`), the latter set to stimulation mode
  (`set_roi_type(3)`) → FRAPPA → `run_stimulation_experiment` (duration tracks ROI
  area, see TODO #3) → `save_current_document` → delete both ROIs → close both
  documents. Verified 20260826 (halved dummy ROIs): circle half → 20.9 s,
  rectangle half → 17.0 s (model T ≈ 15 s + 0.8 ms/px × area predicts 20.1/16.6);
  40 frames × 1024×1024 each. Each FRAP file contains both the standard whole-cell
  polygon **and** the `InterpType.StimulationROI` half (correct polygon and center;
  NIS stores closed polygons without the repeated closing vertex, so one point
  fewer than the pixel-space polygon). Deletion happens *after* saving, so both ROIs
  stay in the file for downstream analysis; GUI left clean.
- **20260901: loop verified with the real cellpose detector** (2 cycles,
  DAPI-stained nuclei, GUI: 2 ch × 1024², 12 FRAP frames): per cycle survey
  ~10 s → V100 detection ~2.1 s → stimulation 14.1 / 13.9 s. c01 and c02
  stimulated **different nuclei** (cross-cycle remapping worked); each FRAP
  file contains the whole-cell `StandardROI` + the `StimulationROI` whose
  center is offset to the **left** of the cell center (the default left-half
  convention survives all the way to NIS). `autofrap()` now takes
  `detection_fun` (a `partial` of `detection.detect`) and **defaults to the
  remote cellpose detector** (`CELLPOSE_SERVER_URL`, `SURVEY_CHANNEL=0`);
  pass `partial(detection.detect, detector='dummy')` to test without the
  server. Note: saving the **frozen live view** (`ImageSaveAs`, current doc
  `"Frozen"`) silently produces no file — grab a single-frame ND acquisition
  instead (used to acquire `nuclei_20260901_110410.nd2`).
- **Known limitation — cross-cycle label-id drift (TODO #13)**: the
  `stimulated` set is expressed in cycle-1 label numbering and relies on
  `merge_label_slices` keeping that numbering stable. It re-baselines its
  first input via `relabel_sequential`, so if a cell *vanishes* in an
  intermediate cycle, the gap it leaves shifts the ids of all objects above
  it (in scan order) in the next merge — `stimulated` then points at the
  wrong cells: a previously FRAPed cell can be FRAPed twice, and an
  un-FRAPed cell can be skipped (id collision). Verified with a synthetic
  3-cycle trace (A=1, X=2; X vanishes in c2 → new Y gets id 3 and is FRAPed;
  c3 re-baseline shifts Y 3→2 ∉ `stimulated` → Y FRAPed again, while
  reappearing X collides with Y's old id and is skipped). **Needs ≥3
  cycles/FOV** (cycle 1 always FRAPs the smallest id, which can never
  shift), so it cannot fire in the intended 1–2-cycles/FOV experiments;
  consequence is bounded (one cell double-bleached at worst). Fix options if
  longer multi-cycle runs are ever needed: exclude FRAPed cells by centroid
  (id-independent, preferred) or match without re-baselining (calmutils'
  private `_correct_next_plane`).
- **`next_cell()` kept** for backward compatibility; `next_stimulatable_cell()` is the
  preferred function when a stimulation mask is available (integrates the mask check
  into the search loop — cells with zero stim pixels are skipped automatically).
- **New detection helpers** (`detect_stim_mask`, `detect_polygon_stim_mask`) are thin
  wrappers around the existing `find_contours`/`approximate_polygon` pipeline, applied
  to the intersection mask `(labels == cell_id) & stimulation_mask`.
- **Code cleanup**: the main loop in `autofrap.py` now uses f-strings instead of
  old-style `%` formatting (no logic changes).

## TODO (next days)

1. **Multi-FOV: loop `autofrap()` over stage positions** — run the
   verified single-FOV loop on a grid of positions (grid first; a
   center-out spiral ordering is a possible later upgrade — calmutils has
   no spiral function yet, only `centered_tiles` with `snake_rows`
   serpentine order). **Implemented** (`autofrap_grid` in
   `autofrap/autofrap.py`): `grid_positions` for the coordinate set, one
   sub-directory per FOV (`<out_dir>/<run_stamp>/fov<i>/`, plain number —
   exact stage positions live in the nd2 metadata) so the
   per-cycle `<stamp>_cNN_survey/frap.nd2` names keep working unchanged,
   `settle_s` after each stage move, return to start after the last FOV,
   a failing FOV is logged and the run continues; accepts a precomputed
   `positions` list so any visit ordering (e.g. spiral) can be passed in
   later. **Live-verified 20260901 at the microscope**: 2×2 grid, spacing
   1.0, `max_cycles=1`, default cellpose `detection_fun` — 4/4 FOVs,
   one cell each (21/17/18/13 objects detected, i.e. genuinely different
   fields), survey 9.2–9.9 s, V100 detection ~2.1 s, stimulation
   12.6–13.3 s, returned to start; all 8 files present (2-ch × 1024²
   surveys + 12-frame FRAPs), each FRAP file contains the whole-cell
   `StandardROI` and the `StimulationROI`; FRAP-file stage positions match
   commanded within ≤1.3 µm. Run: `test_acquisitions/autofrap_grid/
   20260901_160216/` (log `grid_live_test.log` in `test_acquisitions/
   autofrap_grid/`), runner
   `autofrap/autofrap_bitsnpieces/test_autofrap_grid_live.py`.
2. ~~Fix fallout from the folder move~~ — **import fix done**: `autofrap/autofrap.py`
   and `autofrap/autofrap_bitsnpieces/overview_scan.py` now prepend the repo root to
   `sys.path` (verified: top-level imports resolve under real script-run semantics).
   AppleDouble `._*.py` junk deleted. **Deferred**: stale hardcoded output dirs
   (`overview_scan.py` → `<root>/overview/`, `autofrap.py` → `autofrap/autofrap_out/`)
   — everything at this point is ephemeral testing, so re-running recreating the old
   folders is acceptable for now; point them at `test_acquisitions/` (or make them CLI
   args) once real data starts accumulating.
3. ~~Real detector to replace the dummy (interface: `image -> (labels,
   stimulation_mask)`)~~ — **done (20260901)**: cellpose cpdino-vitb on the
   V100 server via `cellpose_server.py` / `remote_detect_objects`; wired into
   `autofrap()` as the default `detection_fun`; 2-cycle run on real nuclei
   verified (see Part 2 / Part 3).
4. ~~Investigate: stimulation run duration did not scale with ROI area~~ —
   **resolved (20260826, 2-cycle run with dummy areas differing ~3x)**: the earlier
   anomaly was leftovers from previous tests on the microscope (a ROI still in
   memory, etc.). Test result: circle (12853 px) → 25.2 s, rectangle (4096 px) →
   18.2 s. Fits `T ≈ 15 s fixed overhead + 0.8 ms/px × area`: the area-dependent
   part (≈10.3 s vs ≈3.3 s) scales exactly with the area ratio (3.1x).
5. ~~Refactor `nis_util.py`: shared helper for the temp-`.mac` → `nis_ar -mw` →
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
6. Cleanup: test data is now organized in `test_acquisitions/` and kept (incl.
   `test_stim.nd2` and the `autofrap_out/` FRAP files, which contain saved
   stimulation ROIs). Remaining junk at the root: `__pycache__/`,
   `.ipynb_checkpoints/`, `.virtual_documents/`, `pi-session-*.html` (pi session
   logs) — the `._*.py` AppleDouble files in `autofrap/autofrap_bitsnpieces/`
   (leftovers from the macOS move) have been deleted.
7. Pixel ↔ stage coordinate transform for per-tile ROIs (calibration matrix from
   `get_rotation_matrix` + pixel size; `get_roi_info` center as a shortcut) — needed
   to move the stage to a detected object before stimulating. **Low priority**:
   centering the object before stimulating was considered, but the current
   approach (no stage move, ROI drawn directly on the survey image) works fine.
8. Per-cycle QC artifact: save a label-overlay PNG next to each `cNN_survey.nd2`
   (labels + drawn ROIs) so detection can be spot-checked without opening NIS.
9. Client timeout: `remote_detect_objects` still has `timeout=1800` (CPU-era
   leftover); ~60 s is right for the V100 — also decide the fail-fast behavior
   when the GPU server is unreachable mid-run.
10. Detector tuning on real samples: try `diameter` / `min_size` per sample
   (e.g. `min_size` to drop dust/debris); consider multi-channel input
   (channel 1 of the survey is currently unused).
11. **Stage position in nd2 metadata: pick the right XY device block** —
    surfaced by the 20260901 grid run: the `ImageMetadataSeqLV` chunk
    contains several microscope-state blocks (coarse stage + `XYDriver`
    piezo, each with `XYUseN` / `XYKeyN` / `XYPositionXN` entries); a naive
    first-match extraction gave a fixed, wrong position for the *survey*
    files while the *FRAP* files (saved via `ImageSaveAs`) carried the
    correct one (≤1.3 µm). Needed to recover per-FOV stage coordinates
    from survey files alone now that filenames are plain `fovNN`. Likely
    fix: read the block whose `XYUseN` flag is set / match `XYKeyN` to the
    stage. Calibration target: `test_acquisitions/overview/` part-1 files
    (commanded coords in the filenames).
12. **Pending live checks in `nis_util.py`** (need the microscope workstation;
    kept as TODO comments in the code): `close_current_document(save='yes')` —
    what happens if the unsaved document is closed with the save flag;
    `add_polygon_roi` — verify colors other than 'green' render correctly in
    NIS; `set_roi_type` — what does type 2 (reference) do.
13. **Cross-cycle label-id drift in `autofrap()`** (needs ≥3 cycles/FOV, so
    not an issue for the intended 1–2 cycles/FOV; see Part 3 note): if a cell
    vanishes between cycles, `merge_label_slices`' re-baselining shifts the
    ids of objects above the gap and the `stimulated` set points at the wrong
    cells (a cell can be FRAPed twice). For longer multi-cycle runs: exclude
    FRAPed cells by centroid (preferred, id-independent) instead of label id.

## File map

Layout after the reorganization: generated pipeline code under `autofrap/`
(test-like bits under `autofrap/autofrap_bitsnpieces/`), all acquired test data
under `test_acquisitions/`, and the NIS macro wrappers + the cellpose server
at the root (`nis_util.py`, `cellpose_server.py`).

### Generated pipeline code

| file | purpose |
|---|---|
| `nis_util.py` | NIS macro wrappers on top of the shared `_run_macro` helper: `get_*`, `get_current_document`, `save_current_document`, `close_current_document`, `set_position`, `run_current_nd_experiment`, `run_stimulation_experiment`, `open_image`, `add_polygon_roi`, `set_roi_type`, `delete_roi`, `get_roi_count`, `get_roi_info`, `NDAcquisition` (from-scratch builder), `export_nd2_to_tiff`, ... |
| `grid_utils.py` | pure grid geometry for tiled acquisitions: `gen_grid` (moved out of `nis_util.py` — not NIS-specific); used by the old wing-scanner `automation.py` + `NIS_Macro_Acquisition.ipynb` |
| `cellpose_server.py` | cellpose inference server for the GPU machine (runs on 10.163.69.12, V100): FastAPI, `POST /detect` (np.save bytes in/out + `model.eval()` query params), `GET /health` (cuda/device); model loaded once at startup with explicit `device`; setup + run instructions in its docstring |
| `autofrap/autofrap.py` | Part 3: auto-FRAP loop (survey → detect → stimulate next unused cell → repeat); takes `detection_fun` (a `partial` of `detection.detect`), defaults to the remote cellpose detector (`CELLPOSE_SERVER_URL`, `SURVEY_CHANNEL`); `grid_positions(position, fov, nx, ny, spacing)` (pure grid math, moved here from `nis_util.py`); `autofrap_grid` runs the loop over that stage grid or a custom `positions` list, one sub-dir per FOV, return to start |
| `autofrap/detection.py` | Part 2: `detect` (`(labels, stimulation_mask)`; params `detector` / `server_url` / `relabel`; border discard via `clear_border` + `relabel_sequential`), `remote_detect_objects` (cellpose client), `default_stimulation_mask` (left half of each object), `dummy_detect_objects` (labels only), `shuffle_labels`, `relabel_by_distance`, `read_channel`, `label_to_polygon`, `detect_stim_mask`, `detect_polygon_stim_mask`, `split_mask_along_axis_equal_area` (ported from bitsnpieces) |
| `autofrap/autofrap_bitsnpieces/overview_scan.py` | Part 1: grid scan script (move + capture per position) |
| `autofrap/autofrap_bitsnpieces/inspect_microscope.ipynb` | notebook walking through the `get_*` functions + FOV |
| `autofrap/autofrap_bitsnpieces/stimulation_loop.py` | colleague's sketch (source of the label-remapping logic, now ported into `autofrap.py`) |
| `autofrap/autofrap_bitsnpieces/test_merge_label_slices.py` | colleague's sketch: `merge_label_slices` test |
| `autofrap/autofrap_bitsnpieces/split_mask_along_axis_equal_area.py` | mask utility: split a label mask into two equal-area halves along an axis (skimage); ported into `detection.py` (the pipeline copy is canonical) |
| `autofrap/autofrap_bitsnpieces/test_stim_save.py` | one-off test script (save-current-document probe; cleanup candidate) |
| `autofrap/autofrap_bitsnpieces/test_cellpose.py` | one-off cellpose test: local model or `--server URL` (remote-client A/B), reports objects + saves image/label previews (run: `python autofrap/autofrap_bitsnpieces/test_cellpose.py <nd2> [channel] [--server URL]`) |
| `autofrap/autofrap_bitsnpieces/test_nis_util_refactor.py` | equivalence check for the `_run_macro` refactor: all generated `.mac` bodies byte-identical to the frozen pre-refactor snapshot + round-trip / cleanup tests (run: `python autofrap/autofrap_bitsnpieces/test_nis_util_refactor.py`) |
| `autofrap/autofrap_bitsnpieces/test_nis_util_live.py` | live smoke test for the refactored wrappers (run at the microscope): all read-only `get_*` + `set_position` XY/piezo round-trip |
| `autofrap/autofrap_bitsnpieces/test_autofrap_grid_live.py` | live `autofrap_grid` test (run at the microscope): 2×2 grid, `max_cycles=1`, default cellpose detector; output `test_acquisitions/autofrap_grid/` |
| `autofrap/autofrap_bitsnpieces/nis_util_old.py` | **frozen snapshot** of the pre-refactor `nis_util.py` (input of the equivalence check). The check served its purpose (33/33 byte-identical + live re-verification); the snapshot is no longer kept in sync with deliberate macro-body changes and will be removed (with `test_nis_util_refactor.py`) once the code stabilizes |

### Test data (`test_acquisitions/`)

| path | contents |
|---|---|
| `test_acquisitions/overview/` | 2×2 grid-scan test, 4 files (20260819) |
| `test_acquisitions/autofrap_out/<stamp>/` | auto-FRAP runs (dummy detector): `cNN_survey.nd2` + `cNN_frap.nd2` per cycle (three 2-cycle runs, 20260824); FRAP files contain the saved stimulation ROI in the nd2 `rois` metadata. Newer runs went to the stale `autofrap/autofrap_out/` dir instead (TODO #1): 20260826 ×4, **20260901** ×1 (cellpose, 2 cycles, real nuclei) |
| `test_acquisitions/test_*.nd2` | earlier one-off artifacts: `test_current`, `test_current3`, `test_save_on`, `test_saveas`, `test_stim` |
| `test_acquisitions/nuclei_20260901_110410.nd2` (+ `_cellpose_image/labels.png` previews) | 2-ch single-frame survey of DAPI-stained nuclei (1024², ch0 = DAPI); first real detection test image + cellpose segmentation previews |

### Pre-existing project files (not part of this work)

| file | purpose |
|---|---|
| `annotation.py`, `automation.py`, `resources.py`, `start_wing_scanner.bat`, `res/` | wing-scanner project code + calibration JSONs / `dummy.nd2` |
| `simple_detection.py` | pre-existing wing detection (bbox-based, skimage) |
| notebooks: `NIS_Macro_Acquisition.ipynb`, `simple_overview.ipynb`, `create_overview_calibration.ipynb`, `manual_roi_annotation_test.ipynb`, `simple_intelligent_acquisition.ipynb`, `transform_test.ipynb`, `wing_keypoint_detect.ipynb` | pre-existing / exploratory notebooks |
| `nis_ar_help_html/` | extracted NIS macro reference (HTML) |

Junk at the root: `__pycache__/`, `.ipynb_checkpoints/`, `.virtual_documents/`,
`pi-session-*.html` (pi session logs).
