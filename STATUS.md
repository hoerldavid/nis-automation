# Status: NIS-Elements Automation Pipeline

_Last updated: **detector (BYOD) refactor — `detect()` is now a minimal composer (WIP, see TODO #18)** (no microscope): `detection.detect(load_fun, detector_fun, relabel='distance', clear_border=True, stim_mask_fun=None)` returns a `survey_file -> (labels[, stim_mask])` callable — the experiment-specific parts (which channel(s) to load, which model, which areas are FRAP-eligible) are callables passed in, and `detect()` only applies the stable housekeeping (`clear_border` + gap-free renumber, `relabel_by_distance`/`shuffle_labels`) and contract checks (2D int labels matching the image's (y, x); 2D mask; the one-region-per-cell warning). `stim_mask_fun(labels, image)` has a fixed 2-arg signature (e.g. FRAP an organelle detected in the image within each cell); `default_stimulation_mask` (left half) is no longer applied automatically — it is a `stim_mask_fun` building block (adapter `autofrap._default_stim_mask` drops the image arg). The `detector='dummy'|'cellpose-remote'` name dispatch is gone: `autofrap()`'s default and the CLI now pass `partial(detection.remote_detect_objects, server_url=...)` / `detection.dummy_detect_objects` straight in, and `load_fun` is `partial(nd2_helpers.read_channel, channel=...)` — multi-channel loading (e.g. detect in ch0, filter by a ch1 marker) is a `np.stack` lambda in `load_fun`. A detector that also returns a viz bypasses `detect()` and passes a fully custom `survey_file -> (labels[, mask[, viz]])` to `autofrap()` (module docstring documents both tiers + examples). Breaking for the one-off `test_autofrap_errors.py` (left as-is per convention). Tested in-container with stub `load_fun`s (2D + (y, x, c), all contract checks). Previous: **output naming** (`file_prefix`, flat run dir by default, `_cycle<NN>_` tags — see TODO #8 tail) and **TODO #8 done** (no microscope): per-cycle QC overlay hooked into `autofrap()` — `<stamp>_cNN_survey_qc.png` saved before each stimulation run, warn-and-continue; `detection_fun` return is now a 1–3-tuple `(labels[, stimulation_mask[, viz]])` (no mask → whole-cell FRAP; no viz → blank black canvas in `qc.save_qc_overlay(image=None, ...)`), see TODO #8. **sprintf revert**: `get_optical_confs` back to the documented `"i"` form (TODO #17: live re-check). Previous: **ND Acquisition dialog settings are queryable — TODO #14 first step done** (at microscope): the dialog's *current experiment definition* can be queried read-only with **no document open** (settings persist across NIS restarts): tab active states via `ND_IsAcqTabChecked` (names, case sensitive: **Time, XY, Z, Lambda, Large Image** — the multichannel tab displays as λ; verified with all tabs ticked → all TRUE), loop params via `ND_GetTimeLapsePhaseCount` / `ND_GetTimePhaseSchedule` / `ND_MP_GetCount` / `ND_GetZSeriesExp` (params persist even when the tab is unticked — tab state and params are separate questions). `ND_GetExperimentLoopSize` is **not** usable (queries the open *document*; -9 with none open). New in `nis_util.py`: `get_nd_acq_tabs` (verified) + `_nis_running` guard in `_run_macro` — **`nis_ar -mw` spawns a fresh NIS instance when none is running, which closes again after the macro** (an interrupted probe spawned exactly such a throwaway instance); `_run_macro` now refuses to do that and raises instead. **CAUTION**: `ND_GetLambdaChannel(0, ...)` **crashed NIS Elements** (twice; a `done` marker written after the call survived once — cause not identified, user: dialog settings persistent, 2 channels defined); channel count/names stay out of the pre-flight for now. Macro-language gotcha: `sprintf(buf, fmt, args)` is **not** C-variadic — `args` is a comma-separated string of variable names to substitute; literal strings need `strcpy()`. Probe: `autofrap_bitsnpieces/test_nd_exp_getter_live.py`. Next: integrate the tab check into the `autofrap()` pre-flight (TODO #14). Before that: **QC overlay renderer `autofrap/qc.py` (TODO #8, part 1 —
rendering only, not yet hooked into `autofrap()`)** (no microscope
needed): `save_qc_overlay(image, labels, path, stimulation_mask=None,
cell_id=None, cell_poly=None, stim_poly=None, caption=None, dpi=100)`
renders one PNG, layers bottom→top with **explicit zorder**: image →
orange FRAP-mask fill (30 %) → label + mask contours → solid polygons as
sent to NIS (cyan = selected cell, magenta = stim ROI) → label IDs (10 pt,
selected one bold 16 pt cyan matching its ROI; the selected cell gets no
extra outline — the polygons already mark it) → legend (cells / FRAP mask
/ selected cell / stim ROI, 12 pt, only for layers that are present) +
caption. Image: 2D → grayscale with 1–99.5 % percentile clipping, or
RGB(A) `(y, x, 3/4)` → as-is (a multi-channel visualization assembled by
the detector is its responsibility to provide). Pixel-center coordinates
throughout (same convention `find_contours` / `mask_to_polygon` / NIS
use). Gotcha found while testing: matplotlib by default draws Text
(zorder 3) above *every* imshow (zorder 0) regardless of call order, so
the layering was only accidentally right — all zorders are now explicit.
Style iterated with the user (solid cyan/magenta instead of dashed
yellow/red, bigger text, legend kept). Tested with the copied
`0013_ch1.tif` + cellpose masks (43 objects) via
`autofrap_bitsnpieces/test_qc_overlay.py`: three real-data variants
(0013_qc_selected/noselect/bigcell.png) + a synthetic 300×300 case with
per-layer pixel checks (fill color, polygon positions, RGB input;
geometry sits bottom-right because the fixed-font-size legend is
proportionally huge on small canvases). Test artifacts + inputs are
`.gitignore`d. Remaining (TODO #8 part 2): hook into `autofrap()` —
save `<stamp>_cNN_survey_qc.png` per cycle, warn-and-continue on failure,
image from `detect()`'s optional third return so `autofrap()` never
needs to know the channel. Before that: **TODO #12 live checks done + 'type 1 hides ROI' note struck
(20260904, microscope)**: on an unsaved ND-acquisition document —
`close_current_document(save='yes')` pops the GUI Save-As dialog and
**blocks** the macro call until it is answered (cancel keeps the document open)
→ unattended code keeps using 'discard'; `add_polygon_roi` colors render
correctly ('red'/'cyan'/'yellow' checked; `GetROIInfo` color read-back still
always 0); `set_roi_type`: types 1–3 all keep the ROI visible, label prefixed
'B:<n>' (background) / 'R:<n>' (reference) / 'S1:<n>' (stimulation, group 1 of
3) — the earlier 'type 1 hides the ROI' claim is **not reproducible** and was
struck (the disappearing ROIs of an earlier session had another cause, e.g.
closing a document / switching OC). New low-priority TODO #16: stimulation
groups S1–S3 — how to change the group (only S1 is needed now). TODO comments
resolved in `nis_util.py` (docstring notes kept short; details here). Before
that: **TODO list refresh (no microscope needed)**: junk at
the root (`__pycache__/`, `pi-session-*.html` pi session logs, ...) is
already covered by `.gitignore` (it was created during the reorg) — TODO
#6 closed accordingly, the files stay on disk (per user: don't delete).
Two new TODOs from the design-doc cross-reference: #14 startup check of
the survey ND template (design goal 1a) and #15 spiral visit ordering
(design goal 2 — `autofrap_grid` already accepts a custom `positions`
list, only the spiral generator is missing). Before that: **detector contract made explicit: at most one connected
FRAP region per cell (design decision, no microscope needed)**: per the
updated `DESIGN_GOALS_AUTOFRAP.md`, picking *which* FRAP region a cell
gets (largest / most centered / ...) is the detector's job, not the
microscope side's. The contract (step 6): the stimulation mask holds
"not more than one connected region per cell"; cells without a region
are skipped (step 7 — already the behavior of
`next_stimulatable_cell`'s stim-pixel check). Code: `detect()` now
**warns** (`_warn_multi_region`; a violation degrades instead of
aborting the grid, `mask_to_polygon` still returns the largest region)
when a cell's stimulation mask has >1 connected region — 4-connectivity,
the same convention `find_contours` uses for boundaries; the half-cell
default is compliant by construction, but e.g. the left half of a
C-shaped object can in principle be two pieces, so the warning is
reachable even with the current detector. `mask_to_polygon` docstring:
"largest contour" = outer boundary, holes ignored, largest-region
fallback on contract violation. Module + `detect()` docstrings state
the contract. No behavior change for compliant detectors; offline tests
still pass. Before that: **`autofrap.py` `__main__` is now a CLI for `autofrap_grid`
(TODO #2)** (no microscope needed): argparse, the arguments mirror the
`autofrap_grid` parameters 1:1 (out, nis, nx, ny, spacing, max-cycles,
until-done, settle, no-return, detector) so a future notebook "parameters"
cell can call `autofrap_grid(...)` with the same values; `--out` defaults to
`<root>/test_acquisitions/autofrap_grid/` (the stale `autofrap/autofrap_out/`
and hardcoded Windows paths are gone; single FOV = `--nx 1 --ny 1`,
`--detector dummy` for server-less testing). `overview_scan.py` left as-is
(test-only, not production). Verified: `--help` + a run without NIS fails
cleanly with the domain `NonRecoverableError`; all offline tests still pass.
Before that: **error handling: Recoverable/NonRecoverable exceptions +
TODO #9 (timeout/retry)** (no microscope needed): `autofrap()` now translates
every failure into one of two new classes in `autofrap.py` —
`RecoverableError` (per-FOV: detection 5xx on this image, no polygon, ROI
creation failed) and `NonRecoverableError` (run-level: NIS macro aborted /
empty ini read-back, survey or FRAP file not saved — NIS fails silently,
so both are now checked after the fact, document not opened, detection server
unreachable after one client-side retry, OS error) — and best-effort deletes
its own ROIs / closes its documents in a `finally` before re-raising (a failed
cycle no longer leaves type-3 stim ROIs or open documents for the next FOV).
`autofrap_grid()` switches on the two classes: Recoverable skips the FOV and
continues (previous behavior), NonRecoverable aborts the run (remaining
positions unvisited, logged); return-to-start is best-effort in a `finally`
so it happens even after an abort or unexpected error; a failed stage move
aborts. `remote_detect_objects`: timeout 1800 s → **60 s** + one retry with
2 s backoff on connection/timeout/HTTP errors (TODO #9; V100 answers in
~2 s). Also fixed: the stim-ROI failure message printed the cell-ROI id.
Verified offline: new `test_autofrap_errors.py` (fake `nis_util`, 14
scenarios incl. grid continue/abort policy + client retry) 0 failures;
`test_nis_util_refactor.py` + `test_nd2_stage_position.py` + dummy detect
round-trip still pass. Before that: **stage position from nd2 metadata works (TODO #11 closed)** (no
microscope needed, on the copied `test_acquisitions/autofrap_grid/` data):
`autofrap/nd2_helpers.py` — new module for nd2 reads, `stage_position(nd2_file)`
returns the (x, y, z) stage position in µm via the public
`frame_metadata(0).channels[0].position.stagePositionUm` (NIS writes the coarse
stage into the per-frame `dXPos`/`dYPos`/`dZPos`; one value per file). The
previous session's "pick the right XY device block" was a red herring: the raw
`pDeviceSetting` XY slots are **not** the stage — slot 0 is unused and holds a
stale value (the "fixed, wrong" survey position), and the only in-use slot is
`XYDrive` (the Ti XY piezo, position ~0). `read_channel` moved from
`detection.py` into `nd2_helpers.py` (refs updated). Verified
(`test_nd2_stage_position.py`, 0 failures, 5 µm tol): all 8 files of the
20260901_160216 grid run match commanded within ≤1.5 µm (survey **and** FRAP),
all 4 overview (20260819) files within ≤3.1 µm (commanded coords in filenames);
12-frame FRAP files carry the same position in every frame. `detect(dummy)`
round-trip + `test_nis_util_refactor.py` still pass. Before that: **`nis_util.py` TODO cleanup** (no microscope needed): dead
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
| `get_nd_acq_tabs` | {tab: bool} — active ND Acquisition tabs (Time/XY/Z/Lambda/Large Image); queries the current experiment definition, no document needed |

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
    1 = save: on an *unsaved* document (no file) this pops the Save-As dialog
    and blocks the macro call until answered; cancel keeps the document open
    (20260904).
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
  `remote_detect_objects(image, server_url, timeout=60, retries=1,
  **eval_kwargs)` (one retry with 2 s backoff for connection/timeout/HTTP
  errors)
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
- `detection.cell_mask(labels, cell_id, stimulation_mask=None)` → binary mask
  of one cell: `labels == cell_id`, or intersected with the stim mask when one
  is given (same optional-mask pattern as `next_stimulatable_cell`).
- `detection.mask_to_polygon(mask, tolerance=2.0)` → simplified polygon
  vertices (x, y) pixels from a binary mask via `find_contours` +
  `approximate_polygon` (Douglas–Peucker, largest/outermost contour).
  At 1024×1024: dummy circle → 17 vertices, rectangle → 5. `autofrap.py`
  draws both ROIs from these two: whole cell = `mask_to_polygon(cell_mask(...))`,
  stimulation ROI = `mask_to_polygon(cell_mask(..., stimulation_mask))`.
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
    types 1–3 all keep the ROI visible, the label gets a prefix: 'B:<n>'
    (background), 'R:<n>' (reference), 'S1:<n>' (stimulation, group 1 of 3)
    (20260904). An earlier session claimed type 1 *hides* the ROI — **not
    reproducible**, note struck (the disappearing ROIs observed then must have
    had another cause, e.g. closing a document / switching OC).
    Return values are unreliable (1 and 0 both observed for plausible outcomes);
    verify via `GetROICount()` / the GUI instead.
  - `DeleteROI(roi_id)` — removes a *visible* ROI (verified live: count 3 → 1).
    Always returns 0 (also for unknown IDs — don't trust it).
  - `GetROICount()` / `GetROIIdFromIndex(n)` — enumerate **visible** ROIs only
    (track IDs at creation time).
  - `GetROIInfo` color read-back is always 0 (unreliable) — but the colors
    themselves *are* set correctly (20260904: 'red'/'cyan' ROIs render as
    requested; only the read-back is broken). Macro language: has
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
  (via `mask_to_polygon(cell_mask(...))`, standard type — saved for downstream
  analysis) and the stimulation region (`mask_to_polygon(cell_mask(...,
  stimulation_mask))`, i.e. clipped to `(cell == id) & stim_mask`), the latter
  set to stimulation mode
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
- `next_cell()` merged into `next_stimulatable_cell(labels, stimulated,
  stimulation_mask=None)`: the stimulation mask is optional — the "has
  stimulation-eligible pixels" check is only applied when a mask is given.
  (The unused `skip` parameter — retry-after-failed-stimulation, which never
  happens; a failing FOV just moves to the next grid position — was dropped
  at the same time. The `next_cell` in the `stimulation_loop.py` sketch is a
  separate local copy.)
- **ROI helpers compacted**: `label_to_polygon` / `detect_stim_mask` /
  `detect_polygon_stim_mask` merged into two primitives — `cell_mask`
  (whole cell, or cell ∩ stim mask) and `mask_to_polygon` (largest contour →
  Douglas–Peucker polygon); see Part 2 for the signatures.
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
   folders is acceptable for now; point them at `test_acquisitions/` (or make them
   CLI args) once real data starts accumulating. **Done (20260902)**: `autofrap.py`
   `__main__` is now a CLI for `autofrap_grid` with `--out` defaulting to
   `test_acquisitions/autofrap_grid/` (see top entry). `overview_scan.py` keeps
   its hardcoded path — it was only ever a test script, not production.
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
   Note (corrected after checking the NIS manual): `get_optical_confs`'s
   macro used `sprintf(&buf, "conf%i", "i" )` — per the sprintf() documentation
   in the NIS manual this is the *documented* form (the third argument is a
   comma-separated string of variable names, not C-variadic values). It was
   briefly "fixed" to bare `i` on 20260826, which also worked live (all 16
   names returned — the macro compiler is permissive), but has since been
   reverted to the documentation-conformant `"i"` in both `nis_util.py` and
   the `nis_util_old.py` snapshot (TODO #17: re-verify at the microscope).
   **Live-verified 20260826** at the microscope: all read-only `get_*` wrappers +
   `set_position` XY round-trip (±2 µm, back within 0.1 µm) + piezo round-trip
   (±1 µm, exact) pass — `autofrap/autofrap_bitsnpieces/test_nis_util_live.py`.
6. ~~Cleanup~~ — **done (20260902)**: test data is organized in
   `test_acquisitions/` and kept (incl. `test_stim.nd2` and the
   `autofrap_out/` FRAP files, which contain saved stimulation ROIs); the
   `._*.py` AppleDouble files (leftovers from the macOS move) have been
   deleted. The remaining root junk (`__pycache__/`, `pi-session-*.html`
   pi session logs, ...) is git-ignored (`.gitignore` already covered it)
   and left on disk.
7. Pixel ↔ stage coordinate transform for per-tile ROIs (calibration matrix from
   `get_rotation_matrix` + pixel size; `get_roi_info` center as a shortcut) — needed
   to move the stage to a detected object before stimulating. **Low priority**:
   centering the object before stimulating was considered, but the current
   approach (no stage move, ROI drawn directly on the survey image) works fine.
8. ~~Per-cycle QC artifact~~ — **done** (renderer `autofrap/qc.py`
   `save_qc_overlay` + hook in `autofrap()`): each cycle saves
   `<stamp>_cNN_survey_qc.png` next to the survey, *before* the
   stimulation run (so it survives a NIS failure), warn-and-continue
   on rendering errors. Along the way, the `detection_fun` return
   contract became a 1–3-tuple: `(labels[, stimulation_mask[, viz]])`
   — only labels required; `stimulation_mask` absent/None → whole cell
   is FRAPed (downstream `next_stimulatable_cell` / `cell_mask` were
   already None-tolerant); `viz` (2D or RGB(A), detector-assembled)
   feeds the overlay, absent → blank black canvas (`save_qc_overlay`
   now accepts `image=None`; autofrap() deliberately does *not* fall
   back to reading a survey channel — it doesn't know which channel(s)
   the detector used). Position 2 is always the mask, position 3
   always the viz (no dtype sniffing, no dict). Tested with synthetic
   label maps (2D / RGB / blank); first live artifacts at the next
   run. **File naming** (along the way): `autofrap()` takes a
   `file_prefix` (default: timestamp, standalone runs; `autofrap_grid`
   passes `fov<NN>` per position). Grid default is now **one run
   directory per grid run** — `<run_stamp>/fov03_cycle01_survey.nd2`
   (the prefix keeps files self-describing and lets the QC PNGs be
   browsed side by side); per-FOV sub-directories are opt-in via
   `autofrap_grid(..., fov_subdirs=True)`. The cycle tag is
   `_cycle<NN>_` via a `CYCLE_PREFIX` constant instead of `_c<NN>_`
   (the bare `c` reads like color channel).
9. ~~Client timeout: `remote_detect_objects` still has `timeout=1800` (CPU-era
   leftover); ~60 s is right for the V100 — also decide the fail-fast behavior
   when the GPU server is unreachable mid-run.~~ — **done (20260902)**:
   timeout 60 s + one retry (2 s backoff) for connection/timeout/HTTP errors;
   fail-fast decided via the new exception classes: server unreachable
   (after retry) → `NonRecoverableError` → grid aborts; per-image 5xx →
   `RecoverableError` → FOV skipped, run continues. See `autofrap.py` +
   `test_autofrap_errors.py`.
10. Detector tuning on real samples: try `diameter` / `min_size` per sample
   (e.g. `min_size` to drop dust/debris); consider multi-channel input
   (channel 1 of the survey is currently unused).
11. ~~**Stage position in nd2 metadata: pick the right XY device block**~~ —
    **done (20260902)**: the stage position is the per-frame `dXPos`/`dYPos`/
    `dZPos` (public: `frame_metadata(0).channels[0].position.stagePositionUm`),
    correct in survey *and* FRAP files alike — wrapped as
    `nd2_helpers.stage_position()`. The "XY device blocks" of the previous
    attempt are the raw `pDeviceSetting` slots: slot 0 unused (stale value —
    the "fixed, wrong" survey position), the only in-use slot is `XYDrive`
    (Ti XY piezo, position ~0). Verified against commanded coords: 8/8 grid-run
    files ≤1.5 µm, 4/4 overview files ≤3.1 µm
    (`test_nd2_stage_position.py`).
12. ~~**Pending live checks in `nis_util.py`**~~ — **done (20260904, at the
    microscope, on an unsaved ND-acquisition document)**:
    `close_current_document(save='yes')` pops the GUI Save-As dialog and blocks
    the macro call until answered (cancel keeps the document open);
    `add_polygon_roi` colors render correctly ('red'/'cyan'/'yellow';
    `GetROIInfo` color read-back still 0); `set_roi_type`: types 1–3 keep the
    ROI visible, label prefixed 'B:'/'R:'/'S1:' (background/reference/stimulation)
    — the re-check also struck the 'type 1 hides the ROI' note (not
    reproducible). TODO comments resolved in `nis_util.py`.
13. **Cross-cycle label-id drift in `autofrap()`** (needs ≥3 cycles/FOV, so
    not an issue for the intended 1–2 cycles/FOV; see Part 3 note): if a cell
    vanishes between cycles, `merge_label_slices`' re-baselining shifts the
    ids of objects above the gap and the `stimulated` set points at the wrong
    cells (a cell can be FRAPed twice). For longer multi-cycle runs: exclude
    FRAPed cells by centroid (preferred, id-independent) instead of label id.
14. **Startup check of the survey ND template (design goal 1a)**: when
    autoFRAP starts, make a reasonable effort to verify the
    GUI-configured survey ND-Acquisition is sane — single image, one or more
    channels; Z-stacks may be allowed in the future, but timeseries and
    multi-position acquisitions don't make sense. Not implemented yet:
    `autofrap()` currently just runs the template as-is.
    **API found (20260904, doc grep + live probes)**: the ND Acquisition
    dialog's *current experiment definition* is queryable read-only with no
    document open — tab *active* state: `ND_IsAcqTabChecked(tab)` (names,
    case sensitive: Time, XY, Z, Lambda, Large Image), loop params:
    `ND_GetTimeLapsePhaseCount()` +
    `ND_GetTimePhaseSchedule(i, &interval, &duration, &loopcnt)`,
    `ND_MP_GetCount()`, `ND_GetZSeriesExp(...)` (incl. Z count + device).
    Loop params persist even when a tab is not checked — so both the tab
    state *and* the params matter for the check. **CAUTION**: probing
    crashed NIS — a macro with `ND_GetLambdaChannel(0, &name, &oc, &color,
    &before, &after, &aftype, &afarg1, &afarg2);` followed by
    `Int_SetKeyValue(...,"done",1)` and `Int_SetKeyString(...,"name",name)`
    wrote 'done'=1, then crashed NIS Elements (never wrote 'name'); cause
    not identified (user: dialog settings are persistent and 2 channels
    were defined, so it is not "no channels") — probe
    `autofrap_bitsnpieces/test_nd_exp_getter_live.py`; retry the channel
    query in small steps once NIS is back. Also: NIS `sprintf(buf, fmt,
    args)` is not C-variadic — `args` is a comma-separated string of
    variable names to substitute; literal strings go through `strcpy()`.
    **Done**: tab getter wrapped as `get_nd_acq_tabs` (names verified with
    all tabs ticked); `_run_macro` gained a `_nis_running` guard (see top
    entry). Remaining: wire the check into `autofrap()` startup (stop if
    Time/XY/Large Image active; Lambda must be active — leave channel
    correctness to the user), and revisit channel count/names separately.
15. **Spiral visit ordering (design goal 2)**: the grid is done; a
    center-out square spiral around the start position is the proposed
    upgrade. `autofrap_grid` already accepts a precomputed `positions`
    list, so only a spiral generator is missing (calmutils has no spiral
    function yet, only `centered_tiles` with `snake_rows` serpentine order).
16. **Stimulation ROI groups S1–S3 (low priority)**: `ChangeROIType(3)` puts
    the ROI into stimulation group 1 (label 'S1:<n>'); there are 3 groups
    (S1–S3). How to select/change the group is unknown — no group parameter in
    the macro API (`ChangeROIType` takes type 0–3 only; a grep of
    `nis_ar_help_html/` found no group function; `Stimulate(dur, StimMask,
    StimFinish)`'s `StimMask` selects lasers, not groups). Only S1 is needed
    now (one stimulation ROI per FOV); investigate if multiple stimulation
    ROIs per FOV ever become necessary.
17. **Re-verify `get_optical_confs()` at the microscope**: the macro was
    reverted from bare `i` to the documented `sprintf(&buf, "conf%i", "i")`
    form (see TODO #5 note) — quick check via the read-only `get_*` live test
    script (`autofrap_bitsnpieces/test_nis_util_live.py`); expect all 16 conf
    names back.
18. **Flexible detectors / bring-your-own (BYOD)** — **WIP (20260905)**:
    `detect()` refactored into a minimal composer (see top entry):
    `detect(load_fun, detector_fun, relabel='distance', clear_border=True,
    stim_mask_fun=None)` → `survey_file -> (labels[, stim_mask])`;
    `stim_mask_fun(labels, image)` (fixed 2-arg: e.g. organelle-based
    FRAP regions); no more `detector=` name dispatch; per-experiment
    channel loading (multi-channel `load_fun`) and mask policies are
    just callables. **Remaining**: viz is currently all-or-nothing —
    a detector that returns one must bypass `detect()` with a fully
    custom `detection_fun`; a first-class viz hook (e.g. a
    `viz_fun(labels, image)` part) is still to be designed. Also, how
    user-supplied functions are actually integrated in practice
    (notebook-only vs. CLI, sharing parts, ...) needs a bit more
    thought.

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
| `autofrap/autofrap.py` | Part 3: auto-FRAP loop (survey → detect → stimulate next unused cell → repeat); takes `detection_fun` (`survey_file -> (labels[, stim_mask[, viz]])`), default composed with `detection.detect` (remote cellpose detector, `CELLPOSE_SERVER_URL`, `SURVEY_CHANNEL`, left-half mask; `_default_stim_mask` adapter); `grid_positions(position, fov, nx, ny, spacing)` (pure grid math, moved here from `nis_util.py`); `autofrap_grid` runs the loop over that stage grid or a custom `positions` list, **one flat run dir by default** (`fov<NN>_` file prefix; `fov_subdirs=True` opt-in), return to start. **Error handling**: `AutofrapError` / `RecoverableError` (per-FOV, grid continues) / `NonRecoverableError` (grid aborts); `autofrap()` translates low-level exceptions at the points where their meaning is known (incl. post-save file checks for NIS's silent failures) and cleans up its ROIs/documents best-effort on failure; `autofrap_grid` catches the two classes, best-effort return-to-start in `finally`. **CLI** (`python autofrap/autofrap.py`):
argparse wrapper for `autofrap_grid`, args mirror the function parameters 1:1
(for a future notebook parameters cell); `--out` →
`test_acquisitions/autofrap_grid/`, `--detector dummy|cellpose-remote` |
| `autofrap/nd2_helpers.py` | ND2 read helpers: `read_channel` (moved from `detection.py`), `stage_position` (per-frame `dXPos`/`dYPos`/`dZPos` → public `stagePositionUm`; the raw `pDeviceSetting` XY slots are *not* the stage — see TODO #11) |
| `autofrap/detection.py` | Part 2: `detect` (minimal composer: `detect(load_fun, detector_fun, relabel='distance', clear_border=True, stim_mask_fun=None)` → `survey_file -> (labels[, stim_mask])`; optional housekeeping: `clear_border` + gap-free renumber, `relabel_by_distance`/`shuffle_labels`; contract checks: 2D int labels matching the image, 2D mask, at most one connected stim region per cell — `detect` warns on violation, `mask_to_polygon` falls back to the largest region), `remote_detect_objects` (cellpose client), `default_stimulation_mask` (left half of each object — a `stim_mask_fun(labels, image)` building block, not applied automatically), `dummy_detect_objects` (labels only), `shuffle_labels`, `relabel_by_distance`, `cell_mask`, `mask_to_polygon`, `split_mask_along_axis_equal_area` (ported from bitsnpieces) |
| `autofrap/qc.py` | QC overlay renderer (TODO #8): `save_qc_overlay` — image (2D grayscale percentile-clipped, or RGB(A) as-is) + FRAP-mask fill + label contours + NIS polygons (cyan selected cell / magenta stim) + label IDs + legend + caption; explicit zorder, headless Agg (see top entry) |
| `autofrap/autofrap_bitsnpieces/overview_scan.py` | Part 1: grid scan script (move + capture per position) |
| `autofrap/autofrap_bitsnpieces/inspect_microscope.ipynb` | notebook walking through the `get_*` functions + FOV |
| `autofrap/autofrap_bitsnpieces/stimulation_loop.py` | colleague's sketch (source of the label-remapping logic, now ported into `autofrap.py`) |
| `autofrap/autofrap_bitsnpieces/test_merge_label_slices.py` | colleague's sketch: `merge_label_slices` test |
| `autofrap/autofrap_bitsnpieces/split_mask_along_axis_equal_area.py` | mask utility: split a label mask into two equal-area halves along an axis (skimage); ported into `detection.py` (the pipeline copy is canonical) |
| `autofrap/autofrap_bitsnpieces/test_stim_save.py` | one-off test script (save-current-document probe; cleanup candidate) |
| `autofrap/autofrap_bitsnpieces/test_cellpose.py` | one-off cellpose test: local model or `--server URL` (remote-client A/B), reports objects + saves image/label previews (run: `python autofrap/autofrap_bitsnpieces/test_cellpose.py <nd2> [channel] [--server URL]`) |
| `autofrap/autofrap_bitsnpieces/test_nd2_stage_position.py` | offline check of `nd2_helpers.stage_position` vs commanded coords: 20260901_160216 grid run (positions parsed from `grid_live_test.log`) + 20260819 overview run (coords in filenames), 5 µm tolerance (run: `python autofrap/autofrap_bitsnpieces/test_nd2_stage_position.py`) |
| `autofrap/autofrap_bitsnpieces/test_qc_overlay.py` | one-off visual + synthetic pixel test of `qc.save_qc_overlay` (run: `python autofrap/autofrap_bitsnpieces/test_qc_overlay.py`; uses the copied `0013_ch1.tif` + cellpose masks at the repo root, writes git-ignored 0013_qc_*.png) |
| `autofrap/autofrap_bitsnpieces/test_autofrap_errors.py` | offline test of the error handling: fake `nis_util` layer, 14 scenarios — each failure mode checked for its exception class (missing survey/FRAP file, macro abort, server down / 5xx / corrupt file, ROI failure + cleanup, broken open), grid continue/abort policy, best-effort return-to-start, `remote_detect_objects` retry (run: `python autofrap/autofrap_bitsnpieces/test_autofrap_errors.py`) |
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
