# Status: NIS-Elements Automation Pipeline

_Last updated: **Detector runtime parameters** (20260714, no microscope): `build_detector` now accepts a `parameter_map` setting that controls how extra keyword arguments (passed via `autofrap_grid --detector-arg key=value`, repeatable) are routed to its sub-functions. Three modes: `None` (default — extra args silently ignored, backward compatible), `'auto'` (each sub-function receives the args it can accept: those matching its named parameters, or all of them if it accepts `**kwargs`), and `dict` (explicit mapping `{<build_detector_arg_name>: {<runtime_key>: <internal_name>}}` for renaming and selective routing — resolves name collisions like `channel` meaning different things in `load_fun` vs `detector_fun`). The `_detect` closure returned by `build_detector` now accepts `**runtime_kwargs` and routes them via a `_route` helper; `_make_viz` likewise. `autofrap()` and `autofrap_grid()` accept `**detector_kwargs` and forward them to `detection_fun(survey_file, **kwargs)`. Standalone `detection_fun` files work unchanged (empty `**{}` = no extra args). Before that: **TODO #26 done: `settle_s` removed (20260909, at the microscope)**: live-verified `StgMoveXY` **blocks until arrival** — three timed relative moves (+1.5 mm, −3 mm, +1.5 mm back), immediate `get_position` read-back within ≤0.6 µm of commanded after every move (an async move would have been caught mid-travel); call wall time ~constant 1.7–2.0 s (the `nis_ar`/macro overhead, not the mechanical move); re-run with the GUI stage speed set to "extra slow" gave identical timings — that setting does not affect macro-driven moves. Per user: the `settle_s` parameter of `autofrap_grid` is removed entirely (no residual settle — the stage is at rest when the macro returns): both `time.sleep` calls, the docstring entry and the `--settle` CLI flag are gone; `bitsnpieces` one-offs (`overview_scan.py` keeps its own local `settle_s`, `test_autofrap_errors.py` — stale — still passes `settle_s=0.0`) left as-is per convention. Before that: **ROI-persistence probes, round 2 (20260909 PM, at the microscope)**: minimal A/B/C/D live probes (`test_acquisitions/roi_persistence_probe/<stamp>/`, ~9 s acqs, two small triangle ROIs). New established facts: (1) **new acquisitions do NOT close previous documents** — A/B/C stay open in parallel (NIS auto-suffixes colliding manual acqs, e.g. `imgC001.nd2`); (2) ROIs added *after* a doc's save are that doc's **unsaved changes**: `ActivateDocument` on a modified doc pops a save query **naming that doc** (repro 3/3, via `activate_opened_document` *and* raw `ActivateDocument` — `GetOpenedDocumentList` ruled out: no `*` in its list, the asterisk is a GUI-only decoration): **Save** → written to file, switch completes; **Discard** → changes reverted (post-save ROIs vanish from that doc), switch completes; **Cancel** → **switch aborted** (current doc unchanged, changes kept); switching to an unmodified doc is dialog-free — this also explains the old #32 "unexplained save dialog"; (3) the ND-experiment save **embeds ROIs present at acquisition time**: a doc that acquired inherited ROIs is unmodified (no dialog on switch) — its file already contains them (survey files show 0 ROIs because ROIs are added *after* the survey save); (4) **`DeleteROI` is per-document** — deleting on one doc never removes ROIs from another (A kept its ROI through all deletes done on B); (5) **the new-acquisition inheritance source is still not pinned down**: a controlled test refuted "copy from current doc" (C current with view `[1]` after deleting roi 2 on C → next acq got `[1,2]`, roi 2 still alive on B), while an earlier run refuted "union of all existing" (C had `[1,2]` but the next acq got `[]` — in that run the current doc B had just been emptied by the deletes, i.e. the pipeline's delete-on-survey pattern, which behaves cleanly and matches the user's manual follow-up too). Per user: not going further down this rabbit hole for now — `autofrap()` already deletes both ROIs **on the survey image** (after FRAP save/close, survey doc current, `pipeline.py` step 7), which keeps the state clean (multi-cycle runs verified); no pipeline change. For later offline inspection: `ND2File.rois` exists; probe files with ROIs embedded on disk: `20260909_165534/imgA.nd2` (1 ROI, via activation Save), `20260909_165534/imgB.nd2` (2), `20260909_171653/imgC.nd2` + `imgD.nd2` (2). GUI left in the user's manual-acquisition state at session end. Before that: **TODO #25 done + session-global ROIs discovery (20260909, at the microscope)**: `autofrap()` step 7 now makes the still-open survey document current with `activate_opened_document` (no disk re-load; `get_current_document` verify, `open_image` only as fallback on mismatch) instead of re-opening it from disk; the `finally` cleanup uses the same pattern. **Major ROI gotcha found while verifying**: ROIs made with `CreatePolygonROI` are `ScopeType.Global` — they are **session-global, not per-document and not stage-position-keyed**: closing the document (even with discard) does *not* remove them, and **any** new acquisition at **any** position picks them up again (probe: the same 4 stale ROIs, same ids, at the same position *and* 100 µm away). The end-of-cycle `delete_roi` calls are therefore mandatory — dropping them (closing alone) made a 2-cycle run pollute cycle 2's survey *and* FRAP file with cycle 1's ROIs (4 ROIs in the c2 FRAP). Misleading probe along the way: close + re-open of the *saved* file shows 0 ROIs (re-attachment only happens for **new acquisitions**), so close-only cleanup looked fine until the multi-cycle run. Contaminated session cleaned via new wrapper `get_roi_ids` (`GetROICount` + `GetROIIdFromIndex`) → delete all → fresh acquisition shows 0. Verified: 2-cycle dummy run → each FRAP exactly 2 ROIs (Standard + Stimulation), surveys 0, GUI and session clean afterwards. Follow-up probes (cross-document ROI deletion) were **inconclusive** and surfaced further odd GUI behavior — consolidated in new **TODO #32 (Understand ROI persistence)**: `DeleteROI` from a *non-attached* document did *not* remove the global ROIs (a fresh acquisition afterwards still showed the same ids) — deletion only works with the attached document current, which is exactly what the pipeline does, so it is unaffected; additionally two one-off `nis_ar -mw` hangs (no pattern, recovered on retry) and one unexplained save dialog on `CloseCurrentDocument(QUERYSAVE_NO)` (user cancelled; repro unknown). Per user: no further ROI-handling changes until #32 is understood; NIS GUI state left as found. Second probe gotcha: file paths in macro calls must be **absolute** — relative paths resolve against the temp `.mac`'s directory, and `ImageOpen` then fails *silently* (current document unchanged). Before that: **TODO #24 closed (20260909, at the microscope)**: the proactive `open_image` after the ND run is gone — the survey document is already current (`open_after=True`, live-verified), the `get_current_document` check is now a real verification, and `open_image` survives only as a fallback on mismatch (re-verify → abort if it still fails; same verify/fallback pattern as planned for #25). Confirmed with a 1×1 dummy run (full cycle, no fallback needed, clean GUI state). Before that: **smoke runs after the offline reorg + cleanup-gap fix (20260909, at the microscope, no sample)**: (a) dummy-detector 2×2 grid (`--detector autofrap/detectors/dummy_detector.py`, 1 cycle/FOV) — 4/4 FOVs, survey ~10 s (first 23 s, one-off), stim 12.1–12.7 s, returned to start; output verified: 12 files (4 survey + 4 FRAP + 4 QC PNG, flat `fovNN_cycleNN_` naming), every FRAP file carries `StandardROI` + `StimulationROI`, stage position within ≤0.5 µm of commanded; (b) cellpose-detector run with the server **down** (error-logic check): survey acquired, detection failed after the client retry → `NonRecoverableError` → grid aborted at FOV 1, remaining FOVs unvisited, return-to-start in `finally` — exactly the designed behavior. **Found + fixed a cleanup gap along the way**: `autofrap()`'s `finally` only cleaned up when ROIs existed — on a detection failure the survey document left open by the ND run (`open_after=True`) lingered (confirmed live: it was the current document after the aborted run). The `finally` now also closes the survey document when no ROIs were created (best effort). Re-ran the server-down case: no leftover documents, GUI back to pre-run state. **This is also live evidence for TODO #24**: right after the ND run the survey is the current document, so the post-run `open_image` is a no-op. (Also fixed a stale docstring path in `dummy_detector.py`.) Before that: **opened-document wrappers + live verification (20260909, at the microscope)**: new `nis_util` wrappers from the ImageDocument→Activate help pages — `get_opened_documents` (`GetOpenedDocumentList(docs, 24, MAX_FILE_NAME=260)`: packed fixed-width name slots; returns the count, or a **negative** = required capacity when the buffer is too small), `activate_document` (`ActivateDocument(name)` — makes an already-open document current, no disk re-load), `activate_opened_document` (matches a path/title against the list — full-name then unique base name, pure logic in `_match_opened_document` — and activates the **exact NIS spelling** of the matched entry, sidestepping the full-path-vs-title question of `ActivateDocument`). **Live-verified**: the list returns full paths for saved files and GUI titles for non-file documents (the frozen live view appears as `Frozen`); `ActivateDocument` works for both and `get_current_document` follows the switch (incl. to `Frozen`); base-name matching works; switch/restore round-trip clean. Follow-up probe (the test's base-name step had only verified the *Python-side* match, since the survey was already current): `ActivateDocument` also accepts a **unique base name** — switching Frozen → survey with just the `.nd2` filename worked and `get_current_document` confirmed it; so raw `activate_document` is safe even without the list round-trip (ambiguity with two same-named docs is still guarded in `activate_opened_document`). One-off: one `nis_ar -mw` call then blocked for minutes with no error and recovered on retry (transient GUI busy state) — note: `_run_macro` has **no subprocess timeout** on `nis_ar -mw`, so such a hang would block an unattended run indefinitely (watch for recurrence; a timeout + retry is the fix if it ever does). Tests: `test_opened_documents_match.py` (8 offline cases) + `test_opened_documents_live.py` (all passed at the scope). This is the building block for TODO #24/#25 (wiring into `autofrap()` is the remaining step). Before that: **TODO #17** (20260909, at the microscope): `get_optical_confs()` re-verified — the documented `sprintf(&buf, "conf%i", "i")` form works; all names read back, list now 17 (new `405 CSU-W1 FRAP`; `FRAPPA` index 0). Before that: **TODO #14** (no microscope): ND Acquisition template pre-flight check in `autofrap_grid()` — raises `NonRecoverableError` if Time/XY/Large Image tabs are active; allows Lambda (multi-channel), Z (future support), and nothing active (single image with current laser). Pure logic in `_check_nd_acq_template()` (11 tests + 1 wrapper test).
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
`autofrap/pipeline.py` as a pure `grid_positions(position, fov, nx, ny,
spacing)` (callers `autofrap_grid` + `overview_scan.py` updated, one fewer
stage read per grid run); `ROI_COLORS` now hex literals. Verified:
`test_nis_util_refactor.py` still 33/33 byte-identical + all round-trip
tests pass; moved functions numerically identical to the pre-cleanup code
on 200 random arg combos each. The three remaining `nis_util.py` TODOs are
live checks (TODO #12). Before that: **multi-FOV grid runner** (`autofrap_grid` in
`autofrap/pipeline.py`) — loops the verified single-FOV `autofrap()` over
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
  - Gotcha: file paths in macro calls must be **absolute** (20260909) — a
    relative path resolves against the temp `.mac`'s directory (OS temp dir),
    not the Python CWD; `ImageOpen` with a relative path then fails
    *silently* (macro "succeeds", current document unchanged).
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
| `get_optical_confs` | list of 17 names (incl. `FRAPPA`; was 16, before 13) |
| `get_nd_acq_tabs` | {tab: bool} — active ND Acquisition tabs (Time/XY/Z/Lambda/Large Image); queries the current experiment definition, no document needed |
| `get_opened_documents` | list of open document names — full path for saved files, GUI title otherwise (e.g. `Frozen` for the frozen live view) |
| `activate_document` / `activate_opened_document` | make an already-open document current (`ActivateDocument`, no disk re-load); the latter matches a path/title against the list and activates the exact NIS spelling |
| `get_roi_ids` | list of IDs of all visible ROIs on the current image (`GetROICount` + `GetROIIdFromIndex`) |

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
    (track IDs at creation time); wrapped as `get_roi_ids` (20260909).
  - **ROIs are session-global (20260909, `ScopeType.Global`)**: ROIs created
    with `CreatePolygonROI` are not per-document and not stage-position-
    keyed — closing the document (even with discard) does not remove them,
    and *any* new acquisition at *any* position picks them up again
    (same ids at the same position and 100 µm away); `DeleteROI` removes
    them from the session. Re-opening a *saved* file does not re-attach
    them (re-attachment only happens for new acquisitions) — so a
    close/reopen probe under-tests this; a multi-cycle run exposes it
    (stale ROIs land in the next cycle's survey *and* FRAP files).
    End-of-cycle `delete_roi` calls are mandatory for this reason.
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
  remote cellpose detector** (`cellpose_remote_detector.py`);
  pass `partial(detection.detect, detector='dummy')` to test without the
  server. Note: saving the **frozen live view** (`ImageSaveAs`, current doc
  `"Frozen"`) silently produces no file — grab a single-frame ND acquisition
  instead (used to acquire `nuclei_20260901_110410.nd2`).
- **Cross-cycle matching** (TODO #13 resolved): replaced
  `merge_label_slices` (cycle-1 ID numbering + re-baselining) with
  centroid-based matching against an accumulated "already-imaged" map
  (`imaged_centroids`). Each new cycle matches detected objects to the
  map via `regionprops` centroids (y, x numpy order);
  ``centroid_threshold`` defaults to ``'auto'`` which uses each cell's
  ``equivalent_diameter_area`` as the matching radius — objects whose
  centroid lies within one equivalent-diameter of a previously
  stimulated cell are considered the same cell. Matched IDs form a
  skip set passed to `next_stimulatable_cell`. No label remapping, no
  ID shifts, no re-baselining — id-independent from cycle 1 onward.
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

32. **Understand ROI persistence** — *no wiring changes before this is
    understood* (probes 20260909, at the microscope). Established:
    ROIs from `CreatePolygonROI` are `ScopeType.Global`; closing a
    document (even with discard) does not remove them; **any** new
    acquisition (any stage position) picks them up again (same ids);
    they do *not* show on documents opened via `ImageOpen` (count 0
    there) — only on new acquisitions; `DeleteROI` from a
    *non-attached* document does **not** remove them (fresh acquisition
    afterwards still showed them) — deletion only works with the
    attached document current (which is exactly what the pipeline does;
    all verified runs ended session-clean). Open questions: what
    "attached" means exactly (position? channel? something else?),
    whether other creation paths (GUI, `CreateRectangleROI`, …) behave
    the same, whether the ROIs ever expire on their own, and one
    unexplained save dialog on `CloseCurrentDocument(QUERYSAVE_NO)`
    (user cancelled; repro unknown). Also observed: two one-off
    `nis_ar -mw` hangs with no pattern (both recovered on retry —
    `_run_macro` still has no subprocess timeout). The pipeline works
    correctly with the current 'activate attached document → delete →
    close' pattern; do not change ROI handling until this is understood. Round 2 (20260909 PM, see top entry): the `ActivateDocument` save query on a modified document is now understood/reproducible (Save/Discard complete the switch, Discard reverts the post-save ROIs, Cancel aborts the switch); `DeleteROI` confirmed per-document; the new-acquisition inheritance source is the remaining open question (both "copy from current/last-active doc" and "union of all existing" refuted in controlled tests, though runs following the pipeline's pattern behave cleanly).

31. **Module/package structure (TODO #30)** — **done** (20260908): `autofrap/` is now
    a proper Python package with `__init__.py`. `autofrap/autofrap.py` renamed to
    `autofrap/pipeline.py` (avoids dir/module name collision). `__init__.py` inserts
    the repo root into `sys.path` on import — this makes `import nis_util` work from
    anywhere inside the package (and `nis_util` stays at the root for legacy code
    compatibility). Internal imports changed to `from autofrap import detection, mask_utils,
    nd2_helpers, qc`; `nis_util` stays as a bare `import` (root-level module). All
    bitsnpieces test files updated: consistent `ROOT = dirname(dirname(dirname(__file__)))`
    + `sys.path.insert(0, ROOT)`, imports use `from autofrap import ...` or `import nis_util`.
    `pipeline.py` has a fallback `sys.path.insert` for direct script execution
    (`python autofrap/pipeline.py`). Verified: `test_autofrap_detection_contract.py` (12/12),
    `test_detect_viz.py` (19/19), `test_read_channel.py` (24/24) pass; `test_autofrap_errors.py`
    needs `nuclei_20260901_110410.nd2` (real microscope data, not in repo); `test_nis_util_refactor.py`
    needs Windows (`tasklist`).

1. **Multi-FOV: loop `autofrap()` over stage positions** — run the
   verified single-FOV loop on a grid of positions (grid first; a
   center-out spiral ordering is a possible later upgrade — calmutils has
   no spiral function yet, only `centered_tiles` with `snake_rows`
   serpentine order). **Implemented** (`autofrap_grid` in
   `autofrap/pipeline.py`): `grid_positions` for the coordinate set, one
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
   `test_autofrap_errors.py`. **Superseded in part (20260907, TODO #21)**:
   the per-image-5xx → `RecoverableError` split is gone — *any* detection
   failure now → `NonRecoverableError` → grid aborts (`autofrap.py` no
   longer translates `requests` exception types at all); `RecoverableError`
   now covers NIS-side, FOV-local failures only (no polygon, ROI creation).
   Current contract test: `test_autofrap_detection_contract.py`
   (`test_autofrap_errors.py` is stale — left as-is per convention).
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
13. ~~**Cross-cycle label-id drift in `autofrap()`**~~ — **done**:
    replaced `merge_label_slices` + cycle-1 ID numbering with centroid-
    based matching against an accumulated "already-imaged" map
    (`imaged_centroids`, y,x numpy order). Each cycle matches detected
    objects to the map via `regionprops` centroids; `centroid_threshold`
    defaults to ``'auto'`` (per-cell ``equivalent_diameter_area`` as
    matching radius). Matched IDs form a skip set passed to
    `next_stimulatable_cell`. No remapping, no re-baselining, no ID
    shifts — id-independent from cycle 1. See `test_centroid_matching.py`
    (12 tests).
14. ~~**Startup check of the survey ND template (design goal 1a)**~~ — **done**:
    `_check_nd_acq_template(tabs)` pure logic + `_run_nd_acq_check(nis_exe)` wrapper
    in `pipeline.py`; called at the top of `autofrap_grid()` before any acquisition.
    Raises `NonRecoverableError` if Time, XY, or Large Image tab is active — a survey
    must be a single image.  Z is allowed (future support), Lambda (multi-channel)
    is fine, and nothing active is OK (single image with current laser/filter).
    11 unit tests (pure logic) + 1 wrapper test.
15. ~~**Spiral visit ordering (design goal 2)**~~ — **done**: `spiral_positions(position,
    fov, spacing, max_positions=None)` in `grid_utils.py`. Square spiral starting at
    center, counter-clockwise, spacing in FOV units. `autofrap_grid` already accepts a
    `positions` list, so wiring it in is just `positions=spiral_positions(start, fov, spacing)`. 12 unit tests + 4 visual plots.
16. **Stimulation ROI groups S1–S3 (low priority)**: `ChangeROIType(3)` puts
    the ROI into stimulation group 1 (label 'S1:<n>'); there are 3 groups
    (S1–S3). How to select/change the group is unknown — no group parameter in
    the macro API (`ChangeROIType` takes type 0–3 only; a grep of
    `nis_ar_help_html/` found no group function; `Stimulate(dur, StimMask,
    StimFinish)`'s `StimMask` selects lasers, not groups). Only S1 is needed
    now (one stimulation ROI per FOV); investigate if multiple stimulation
    ROIs per FOV ever become necessary.
17. ~~**Re-verify `get_optical_confs()` at the microscope**~~ — **done
    (20260909, at the microscope)**: the documented `sprintf(&buf, "conf%i",
    "i")` form works — direct call (read-only, no full live-script run) read
    back all conf names; the list has since grown to **17** (new
    `405 CSU-W1 FRAP` appended at index 16; `FRAPPA` still index 0).
18. **Flexible detectors / bring-your-own (BYOD)** — **done (20260909)**:
    `detect()` refactored into the minimal composer `build_detector` (see
    file map): `load_fun` / `detector_fun` / `stim_mask_fun(labels, image)` /
    `visualization_fun(image)` as plain callables, no `detector=` name
    dispatch; per-experiment channel loading (multi-channel `load_fun`)
    and mask policies are just callables. The open integration question is
    resolved by the detector-parameter commit (see top entry): detector
    files are `.py` files loaded via `load_detector_file` (CLI `--detector`),
    runtime parameters via `--detector-arg key=value` →
    `detection_fun(survey_file, **kwargs)`; `build_detector`'s
    `parameter_map` (None / `'auto'` / dict) routes the kwargs to the
    sub-functions (incl. name-collision renames), and a fully custom
    `detection_fun` simply accepts `**kwargs` by name.
19. ~~**`read_channel`: axis order, multi-channel, dimension validation, Z projection**~~
    (the four user TODOs in `nd2_helpers.py`) — **done (20260907)**: see top
    entry. Survey-only contract: C/Y/X (+Z with `z_projection='max'`),
    everything else (T, P, …) errors; `channel` int → 2D / tuple → (k, y, x);
    axis-order independent via `f.sizes` + `np.transpose` (no xarray).
20. ~~**Default detector composition in `autofrap()`**~~ — **struck**: no default
    detector; all detectors are shipped as `.py` files (`dummy_detector.py`,
    `cellpose_remote_detector.py`) and loaded via `load_detector_file()`.
    the same housekeeping.
21. ~~**Detection failure → always `NonRecoverableError`**~~ — **done
    (20260907)**: single `except Exception` → `NonRecoverableError` (retry
    logic stays in `remote_detect_objects`); `import requests` removed;
    supersedes part of TODO #9 (per-image 5xx used to skip the FOV — now
    aborts the grid: a 5xx after retries means the server is suspect for
    the next FOV too). Docstrings updated; test
    `test_autofrap_detection_contract.py`.
22. ~~**Allow a bare label map (ndarray) as `detection_fun` return**~~ —
    **done (20260907)**: `isinstance(det, np.ndarray)` → `(det,)` before
    the 1–3 tuple/list check; docstrings updated (`autofrap.py` module +
    parameter + Raises, `detection.py` module docstring); the design doc
    left as-is (it already lists the label map as the only non-optional
    item).
23. ~~**Degenerate cell mask: skip the cell, try the next one in the same
    survey**~~ — **done**: inner re-pick loop in `autofrap()` — if the picked
    cell has no polygon, skip it and try the next (`skipped` set passed as
    `stimulated | skipped` to `next_stimulatable_cell`). When all cells in a
    FOV lack polygons the cycle loop breaks (`fovd_done`), the next survey
    cycle is attempted. Nearly unreachable today (cells have ≥1 pixel in both
    masks by construction); defensive; can't be live-verified until a real case.
24. ~~**Survey document already open after the ND run**~~ — **done
    (20260909, at the microscope)**: live evidence (see top entries): after
    the ND run the *current* document is the survey file itself (even with
    detection running in between) — the proactive `open_image` was a no-op
    costing a macro round-trip per cycle. Now: `get_current_document`
    check first (a real verification), `open_image` only as a fallback on
    mismatch (re-verify → `NonRecoverableError` if it still fails) — the
    same verify/fallback pattern planned for #25. Confirmed live with a
    1×1 dummy run (full cycle, fallback never triggered, GUI left clean).
25. ~~**Don't re-open the survey just to close it again**~~ — **done
    (20260909, at the microscope)**: step 7 now activates the still-open
    survey document (`activate_opened_document`, no disk re-load; the list
    returns full paths, so path matching works and `get_current_document`
    follows the switch — the full-path-vs-title question is moot when the
    exact list spelling is used), verifies via `get_current_document`,
    deletes both ROIs, closes; `open_image` kept as fallback on mismatch
    (same verify/fallback pattern as #24); the `finally` cleanup uses the
    same pattern. Verified with a 2-cycle dummy run (each FRAP exactly
    2 ROIs, surveys 0, GUI + session clean). Along the way: the
    `delete_roi` calls proved **mandatory** — see the session-global ROIs
    discovery in the top entry — and new `nis_util.get_roi_ids` wrapper
    (used to clean a contaminated session).
26. ~~**`settle_s` after `set_position` may be redundant**~~ — **done
    (20260909, at the microscope)**: `StgMoveXY` blocks until arrival
    (immediate read-back ≤0.6 µm on all moves incl. 3 mm; ~constant wall
    time; the GUI "extra slow" stage speed has no effect on macro-driven
    moves) — per user the `settle_s` parameter, both sleeps and the
    `--settle` CLI flag were removed entirely (see top entry).
27. ~~**Move general-purpose mask/label utilities to
    `autofrap/mask_utils.py`**~~ — **done (20260907, move by the user,
    review + fixes here)**: `split_mask_equal_area` (renamed from
    `split_mask_along_axis_equal_area`), `shuffle_labels`,
    `relabel_by_distance`, `mask_to_polygon` + `half_object_stim_mask`
    (moved with the split it wraps) in `mask_utils.py`; `cell_mask` stays
    in `detection.py`. The move was broken at every entry point (missing
    `import numpy as np`; package-qualified `autofrap.mask_utils` imports
    that only resolve with the IDE's source-root settings) — fixed to the
    plain `import mask_utils` style; pre-existing `detection.py`
    `__main__` unpack bug fixed (2 values vs 3-tuple with viz). See top
    entry. Note: the bitsnpieces one-off
    `split_mask_along_axis_equal_area.py` keeps its own copy + old name
    (per convention; its "pipeline copy is canonical" note now points at
    `mask_utils.py`).
30. **Clean module/package structure for the final version** (user): the
    pipeline relies on `sys.path` manipulation (`autofrap.py` inserts the
    repo root for `nis_util`, bitsnpieces tests insert `autofrap/`) and
    on IDE source-root settings for some imports (`from autofrap import
    next_stimulatable_cell` in `test_qc_overlay.py`); the failed
    `autofrap.mask_utils` import of the #27 move shows how fragile that
    is. Decide a real layout (e.g. a proper `autofrap` package with
    `__init__.py`, or flat top-level modules) and remove the hacks.
28. **`half_object_stim_mask`: split within the object bbox** (regionprops)
    instead of over the full image per label — low priority: ~20 M element
    ops (tens of ms) vs ~2 s V100 inference.
29. ~~**One-stimulation-region-per-label helpers**~~ — **done**: `largest_region_per_label(labels, mask)` and
    `most_central_region_per_label(labels, mask)` added to `mask_utils.py` (generic ndarray, not tied to 2D). Composable via
    `stim_mask_fun`, **not** auto-applied in `build_detector` (the warning is a detector-debugging signal). 11 unit tests +
    5 visual plots. Previous TODO comment resolved in `mask_utils.py`.

## File map

Layout after the reorganization: generated pipeline code under `autofrap/`
(test-like bits under `autofrap/autofrap_bitsnpieces/`), all acquired test data
under `test_acquisitions/`, and the NIS macro wrappers + the cellpose server
at the root (`nis_util.py`, `cellpose_server.py`).

### Generated pipeline code

| file | purpose |
|---|---|
| `nis_util.py` | NIS macro wrappers on top of the shared `_run_macro` helper: `get_*`, `get_current_document`, `save_current_document`, `close_current_document`, `set_position`, `run_current_nd_experiment`, `run_stimulation_experiment`, `open_image`, `get_opened_documents`/`activate_document`/`activate_opened_document` (ImageDocument→Activate: list/activate open documents, no disk re-load), `add_polygon_roi`, `set_roi_type`, `delete_roi`, `get_roi_count`, `get_roi_ids`, `get_roi_info`, `NDAcquisition` (from-scratch builder), `export_nd2_to_tiff`, ... |
| `grid_utils.py` | pure grid geometry for tiled acquisitions: `gen_grid` (moved out of `nis_util.py` — not NIS-specific); used by the old wing-scanner `automation.py` + `NIS_Macro_Acquisition.ipynb` |
| `cellpose_server.py` | cellpose inference server for the GPU machine (runs on 10.163.69.12, V100): FastAPI, `POST /detect` (np.save bytes in/out + `model.eval()` query params), `GET /health` (cuda/device); model loaded once at startup with explicit `device`; setup + run instructions in its docstring |
| `autofrap/pipeline.py` | Part 3: auto-FRAP loop (survey → detect → stimulate next unused cell → repeat); takes `detection_fun` (`survey_file -> (labels[, stim_mask[, viz]])` or a bare label map); detector files shipped under `autofrap/detectors/` (see `build_detector` + `load_detector_file` in `detection.py` for composing custom detectors); `grid_positions(position, fov, nx, ny, spacing)` (pure grid math, moved here from `nis_util.py`); `autofrap_grid` runs the loop over that stage grid or a custom `positions` list, **one flat run dir by default** (`fov<NN>_` file prefix; `fov_subdirs=True` opt-in), return to start. **Error handling**: `AutofrapError` / `RecoverableError` (per-FOV, grid continues) / `NonRecoverableError` (grid aborts); `autofrap()` translates low-level exceptions at the points where their meaning is known: **any detection failure → `NonRecoverableError`** (no per-exception translation, no `requests` import — the detector may be any callable), NIS-side FOV-local failures → `RecoverableError` (incl. post-save file checks for NIS's silent failures); cleans up its ROIs/documents best-effort on failure; `autofrap_grid` catches the two classes, best-effort return-to-start in `finally`. **CLI** (`python autofrap/pipeline.py`):
argparse wrapper for `autofrap_grid`, args mirror the function parameters 1:1
(for a future notebook parameters cell); `--out` →
`test_acquisitions/autofrap_grid/`, `--detector` takes a `.py` file path defining `detection_fun` |
| `autofrap/nd2_helpers.py` | ND2 read helpers: `read_channel(nd2_file, channel=0, z_projection=None)` — **survey-only**: validates dimensions via `f.sizes` (C/Y/X, Z with `z_projection='max'`; T/P/… raise — timeseries are for downstream analysis), axis-order independent (`_extract` normalizes to (C, Z, Y, X) as present), `channel` int → 2D (y, x) / tuple → (k, y, x) in the given order (one file read for multi-channel `load_fun`s), NIS omits the C dimension of single-channel files (`channel=0`); test `autofrap_bitsnpieces/test_read_channel.py` (TODO #19), `stage_position` (per-frame `dXPos`/`dYPos`/`dZPos` → public `stagePositionUm`; the raw `pDeviceSetting` XY slots are *not* the stage — see TODO #11) |
| `autofrap/detection.py` | Part 2: `build_detector` (minimal composer: `build_detector(load_fun, detector_fun, relabel='distance', clear_border=True, stim_mask_fun=None, visualization_fun=None)` → `survey_file -> (labels[, stim_mask[, viz]])` or a bare label map, fixed positions: 2 = mask, 3 = viz; multi-channel images are (c, y, x), the viz the only (y, x, 3/4) array; optional housekeeping: `clear_border` + gap-free renumber, `relabel_by_distance`/`shuffle_labels` (from `mask_utils`); contract checks: 2D int labels matching the image, 2D mask, at most one connected stim region per cell — `build_detector` warns on violation, `mask_to_polygon` falls back to the largest region; `visualization_fun(image)` → 2D or RGB(A) for the QC overlay, best effort — a failure only warns and drops the viz, never breaks the run), `remote_detect_objects` (cellpose client), `dummy_detect_objects` (labels only), `cell_mask`; general-purpose mask/label utilities moved to `mask_utils.py` (TODO #27) |
| `autofrap/mask_utils.py` | general-purpose mask / label-map utilities (no pipeline-specific logic): `split_mask_equal_area(mask, axis)` (equal-area split of a binary mask along an axis — renamed from `split_mask_along_axis_equal_area`, ported from bitsnpieces), `half_object_stim_mask(labels)` (left half of each object — a `stim_mask_fun(labels, image)` building block, not applied automatically), `shuffle_labels` (random label permutation, `fastremap`), `relabel_by_distance` (1..N by centroid distance to a reference, default image center), `mask_to_polygon` (largest contour + Douglas-Peucker → (x, y) tuples) |
| `autofrap/qc.py` | QC overlay renderer (TODO #8): `save_qc_overlay` — image (2D grayscale percentile-clipped, or RGB(A) as-is) + FRAP-mask fill + label contours + NIS polygons (cyan selected cell / magenta stim) + label IDs + legend + caption; explicit zorder, headless Agg (see top entry) |
| `autofrap/autofrap_bitsnpieces/overview_scan.py` | Part 1: grid scan script (move + capture per position) |
| `autofrap/detectors/` | shipped detector files for ``autofrap_grid --detector``: `dummy_detector.py` (circle+rect, no dependencies), `cellpose_remote_detector.py` (remote server), `example_detector.py` (minimal working example); each defines ``detection_fun`` |
| `autofrap/autofrap_bitsnpieces/inspect_microscope.ipynb` | notebook walking through the `get_*` functions + FOV |
| `autofrap/autofrap_bitsnpieces/stimulation_loop.py` | colleague's sketch (source of the label-remapping logic, now ported into `autofrap.py`) |
| `autofrap/autofrap_bitsnpieces/test_centroid_matching.py` | unit tests for centroid-based cross-cycle cell matching: `regionprops` centroid extraction (y,x order), `equivalent_diameter_area`-based auto mode, numeric override, matching vs non-matching centroids, `next_stimulatable_cell` integration, roundtrip; 12 tests |
| `autofrap/autofrap_bitsnpieces/split_mask_along_axis_equal_area.py` | mask utility: split a label mask into two equal-area halves along an axis (skimage); ported into `mask_utils.py` as `split_mask_equal_area` (the pipeline copy is canonical) |
| `autofrap/autofrap_bitsnpieces/test_stim_save.py` | one-off test script (save-current-document probe; cleanup candidate) |
| `autofrap/autofrap_bitsnpieces/test_cellpose.py` | one-off cellpose test: local model or `--server URL` (remote-client A/B), reports objects + saves image/label previews (run: `python autofrap/autofrap_bitsnpieces/test_cellpose.py <nd2> [channel] [--server URL]`) |
| `autofrap/autofrap_bitsnpieces/test_nd2_stage_position.py` | offline check of `nd2_helpers.stage_position` vs commanded coords: 20260901_160216 grid run (positions parsed from `grid_live_test.log`) + 20260819 overview run (coords in filenames), 5 µm tolerance (run: `python autofrap/autofrap_bitsnpieces/test_nd2_stage_position.py`) |
| `autofrap/autofrap_bitsnpieces/test_qc_overlay.py` | one-off visual + synthetic pixel test of `qc.save_qc_overlay` (run: `python autofrap/autofrap_bitsnpieces/test_qc_overlay.py`; uses the copied `0013_ch1.tif` + cellpose masks at the repo root, writes git-ignored 0013_qc_*.png) |
| `autofrap/autofrap_bitsnpieces/test_autofrap_errors.py` | offline test of the error handling: fake `nis_util` layer, 14 scenarios — each failure mode checked for its exception class (missing survey/FRAP file, macro abort, server down / 5xx / corrupt file, ROI failure + cleanup, broken open), grid continue/abort policy, best-effort return-to-start, `remote_detect_objects` retry (run: `python autofrap/autofrap_bitsnpieces/test_autofrap_errors.py`) |
| `autofrap/autofrap_bitsnpieces/test_nis_util_refactor.py` | equivalence check for the `_run_macro` refactor: all generated `.mac` bodies byte-identical to the frozen pre-refactor snapshot + round-trip / cleanup tests (run: `python autofrap/autofrap_bitsnpieces/test_nis_util_refactor.py`) |
| `autofrap/autofrap_bitsnpieces/test_nis_util_live.py` | live smoke test for the refactored wrappers (run at the microscope): all read-only `get_*` + `set_position` XY/piezo round-trip |
| `autofrap/autofrap_bitsnpieces/test_opened_documents_match.py` | offline tests of `_match_opened_document` (exact / case / base name / ambiguous / title / empty; run: `python autofrap/autofrap_bitsnpieces/test_opened_documents_match.py`) |
| `autofrap/autofrap_bitsnpieces/test_opened_documents_live.py` | live test of the opened-document wrappers (list, activate by exact entry / base name, switch + restore; run at the microscope) |
| `autofrap/autofrap_bitsnpieces/test_autofrap_grid_live.py` | live `autofrap_grid` test (run at the microscope): 2×2 grid, `max_cycles=1`, cellpose_remote detector; output `test_acquisitions/autofrap_grid/` |
| `autofrap/autofrap_bitsnpieces/test_detect_viz.py` | one-off contract test for `detect()`'s `visualization_fun` (synthetic, no nd2/server): all four return-tuple combinations, 2D/RGB viz from a multi-channel load, warn-and-drop on bad shape / raising viz, labels/mask contract checks still raise (run: `python autofrap/autofrap_bitsnpieces/test_detect_viz.py`) |
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
