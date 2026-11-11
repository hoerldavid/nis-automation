# Autofocus, Z drives and PFS in the NIS macro language

Capability reference compiled from the NIS **function reference** (`nis_ar_fr_eng.chm`, 2019 — per-function macro reference, cited below by function name) and the **main manual** (`NIS_AR_ENG.CHM`, 2020 — user manual with GUI semantics, device behavior, JOBS; cited below by its TOC path, e.g. "Devices > Auto Focus Setup"). Both CHMs come from a NIS installation — see `nis_manual/README.md` for how to obtain and extract them.

What is wrapped in `autofrap/microscope/nis.py` is marked **[wrapped]**; everything else is available but not yet wrapped — see §8.

## 1. Autofocus one-shot functions (`Device > Z`)

| Function | Signature | Return / notes |
|---|---|---|
| `StgFocus()` | — | "Auto Focus" command; uses the configured setup (§2). Respects the probe ROI: if a probe is active, focuses only inside it (main manual: Devices > Auto Focus) |
| `StgFocusAdaptive(Step)` | step µm | 1 ok / 0 fail / −1 bad step |
| `StgFocusAdaptiveTwoPasses(Step, FineStep)` | coarse + fine step µm | two-pass adaptive. **[wrapped]** as `do_autofocus` |
| `StgFocusInRange(Range, Step)` | starts Range/2 above current Z, steps down | 1 ok / 0 fail / −1 step<0 / −2 range<0 / −3 <3 planes |
| `StgFocusInRangeEx(Range, Speed)` | continuous: grabs frames while moving down Range; Speed 0–90 | 1 ok / 0 fail / −2 / −3 (white/black scene). `StgSetupFocusEx` is a compat alias |
| `StgFocusInRangeTwoPasses(Range, CoarseStep, FineStep)` | stepped two-pass in range | 1 ok / 0 / −1 / −2 / −3 |
| `StgFocusInRangeExTwoPasses(Range, Speed)` | continuous two-pass; Speed 0/1/2 = half / optimal / double | optimal speed complies with the objective-recommended step |
| `StgFocusPlane()` | — | "Focus using Surface": planar interpolation from ≥3 focus-surface points (§4) |
| `StgFocusCriteria()` | — | runs AF and dumps `Z, criterion` pairs to `FocusCriteria.txt` (debugging) |
| `StgScanFocus(Type, Every)` | 0 none / 1 manual / 2 AF; every Nth field | AF during scanning |
| `StgZ_AutoFocus(coarse)` | — | Coolscope only — not relevant |

## 2. AF configuration (`StgSetupFocus*`, dialog semantics)

* `StgSetupFocus(Method, Step, Stop)` — configures the Auto Focus Setup dialog. Method: 0 adaptive-in-steps, 1 continuous-in-range, 2 step-by-step-in-range, 3 interpolation (focus surface). `Stop=1` → fall back to manual focus on failure. `_StgSetupFocus()` opens the dialog.
* Setup-only variants: `StgSetupFocusTwoPasses(Range, CoarseStep, FineStep)`, `StgSetupFocusAdaptiveTwoPasses(Step, FineStep)`, `StgSetupFocusContinuousTwoPasses()` (two-pass toggle), `StgSetupFocusOffset(µm)` = "Offset Z after AF" (compensate AF offset, e.g. fluorescence).
* Dialog semantics (main manual: Devices > Auto Focus Setup): methods are **Steps in Range** and **Continuous in Range** (latter requires Nikon Ni / Ti2-E; step size then follows the camera frame rate; speed fast/normal/slow — slow = max precision). Two passes = coarse + fine pass. There is a per-objective *recommended step* button. Practical guidance from the manual: 4x → ~10 µm steps, 40x → ~3 µm; range for 20x ≈ 100 µm; never enter a range larger than half the objective's working distance. Documented causes of continuous two-pass failure: camera exposure incompatible with Z-drive speed, auto coarse step too large, very thin specimen → use slow speed, adjust exposure, or single pass.
* JOBS "Autofocus Settings" adds GUI-only options with no known macro equivalent: Z device choice incl. `<fastest>` (piezo preferred, main Z added when the range exceeds it), Exclude Border, Move to Original Z on Failure, Skip Focusing on Background, per-AF optical configuration, AF mode Auto (system picks step and 1 vs 2 passes) vs Manual (main manual: JOBS > Task reference > Autofocus + Focus Surface).

## 3. Focus criterion selection and readback

* `StgFocusSetCriterion(0–4)` — 0 brightfield contrast, 1 intensity (pixel-intensity independent), 2 fluorescence, 3 noise-resistant, 4 max-sum-intensity (confocal). **[wrapped]** in `do_autofocus` (`DEFAULT_FOCUS_CRITERION = 0`).
* `StgFocusSetCriterionDialog(0–3)` — criterion for the *live* focus indicator (Brightfield / Fluorescence / Max Sum / Yeast-Bacteria).
* `GetFocusCriterium()` → double — criterion of the active document, higher = sharper; accounts for the probe. Programmatic "how in-focus am I" readout.
* `EnableFocusCriterium(0/1)` + `ResetFocusCriteriumMax()` — criterion display / max tracking in the System Information Graph. `GetYeastFocusCtiterion()`.
* `Meas_Focus_Criterion` — GA3 measurement action ("Field intensity > Focus Criterion"), documented only in the main manual, not in the function reference; GA3-based alternative to `GetFocusCriterium()`.
* Post-hoc from ND2 z-stacks: `SelectBestFocusedFromND(criterion, componentMask)` (+ dialog variant `GA__SelectBestFocusedFromND`), `ShowFocusedGraphFromND()` (best-focus-plane graph).

## 4. Focus surface (XYZ Overview)

* `StgSetFocusPlane(x[], y[], z[], count)` — define surface from ≥3 points; `StgFocusPlane()` — move Z there (interpolated).
* `StgGetFocusPlaneType()` — 0 smooth / 1 nearest-neighbor interpolation. `StgIsFocusPlaneValid()`, `Stg_LoadLastFocusSurfacePoints()` (reload last session's points — points are cleared on app restart), `Stg_RemoveFocusSurfacePoints()`.
* GUI (main manual: View > Acquisition Controls > XYZ Overview, Focus Surface tab): Add Point at ≥3 XY positions, right-click → "Auto Focus All Points" to redefine Z of all points; interpolation method Smooth / Nearest Neighbor.
* Large-image scans: `Stg_LargeImageScan(Area)` `Autofocus` param = 0 off / 1 AF / 2 focus surface / 3 both. **[wrapped]** as `do_large_image_scan` (currently passes 0). JOBS large-image task adds AF every Nth field / after a stage distance / center tile at start, and skipping empty frames.

## 5. PFS (Perfect Focus System, `Device > PFS`)

PFS is a logical device on TE2000/Ti microscopes (main manual: Cameras & Devices > Logical Devices). **No PFS function is wrapped in `nis.py` yet** (only `ND_KeepPFSOnDuringStageMove(1)` in `NDAcquisition`).

Presence / status:

| Function | Meaning |
|---|---|
| `Stg_IsPFSPresent()` | PFS connected? |
| `Stg_GetPFSStatus()` | −1 error, 0 in-range/off, **1 focused/on**, 3 out-of-range, 4 dichroic mirror out, 5 searching, 6 search failed, 7 unsupported objective (the function-reference entry includes the Ti hardware→logical conversion table) |
| `Stg_IsPFSPlaneValid()` | PFS surface state (doc text unclear) |

On/off and focusing:

* `Stg_SetPFSStatus(0/1)` — 0 off; 1 on, find focused position, hold it.
* `Stg_SetPFSFocus()` — starts a PFS focus cycle, blocks up to 8 s; returns 0 focused / 1 not focused (e.g. Z out of PFS range).
* `Stg_WaitForPFS(timeout_s)` — wait for PFS to lock; recommended 8 s.
* `StgFocusPFS()` — "Auto PFS Focus": finds the best-focused plane and sets the **PFS offset** (contrast with `StgFocus`, whose result is a **Z position** — both end focused, but only the offset feeds the PFS lock).
* `Stg_PFSFindOffset()` — find the PFS offset for the current Z (Ti2 pad "Find PFS Offset" button).

Offset and surface:

* `Stg_GetPFSLensOffsetPosition()` / `Stg_SetPFSLensOffsetPosition(int)` — read/set the offset value (Ti pad offset edit box).
* `StgGetPFSPlanePoints()`, `StgResetPFSPlane()`, `StgUsePFSPlane()` — PFS surface ("Set PFS offset from PFS Surface"): a grid of offset values interpolated to any XY. **The PFS surface can only be created via the JOBS Editor** (main manual: Devices > Set PFS offset from PFS Surface; workflow "Creating PFS Surface": per well/point → PFS On and Focus → Auto PFS Focus → Assign PFS to Point/Well → PFS Off, then Create PFS Surface). The XYZ Overview "PFS Surface" tab visualizes it (requires JOBS Editor).

Hardware / optics:

* `Stg_PFSIsDMInserted()` / `Stg_PFSInsertExtractDM(0/1)`, `Stg_PFSIsDMMotorized()` — dichroic mirror; **PFS works only when the mirror is In** (Ti2 pad).
* `Stg_GetPFSLEDSwitch()` / `Stg_SetPFSLEDSwitch(0/1)` — 870 nm IR LED; `Stg_PFSSetLEDPower(0–4095)`.
* `Stg_PFSGetDishType()` / `Stg_PFSSetDishType(0 glass, 1 plastic)` / `Stg_PFSGetDishTypeChangeable()` — sample-holder material (Ti2 pad pull-down Glass/Plastic).
* `Stg_PFSEnableSignalOutput(0/1)` + `Stg_PFSIsSignalOutputEnabled()` + `Stg_PFSSignalOutputTest(0–5)` — PFS correction signal to the connector on the microscope body (drives a piezo). On Ti2 the same status lines are on BNC I/O: **Focus** (Hi = in focus), **Vicinity** (Hi = in range, not focused), **Up/Down**, **Surface** (Hi = in range) (main manual: Ti2-E microscope page, Configure > I/O tab).
* `_StgSetupPFSFocus()` — opens the Auto PFS Focus Setup dialog (same settings as Auto Focus Setup, but range/step in µm or PFS units; incl. Exclude Border).

Behavioral notes (from the device pages, worth encoding in code comments):

* PFS has a **limited correction range** — "a vast shift of the Z drive can cause PFS to be unable to refocus" (Ti pad; Focus/In-Focus indicator stops shining). If Z jumps too far, `Stg_SetPFSFocus` returns not-focused — poll `Stg_GetPFSStatus` and re-run AF/PFS-Focus when status leaves 1.
* **"Keep Z position" is automatically turned off when PFS is on** (Ti and Ti2) — don't combine `StgZ_SetKeepPosition(1)` with PFS.
* JOBS "PFS On and Focus" task: the Z plane must not be too far from the PFS focal plane or the task fails; on failure you can chain AF or a Z shift. JOBS "Autofocus" labels failed captures with a Focus Failed flag (suppressed if Z moved ≥10 µm after the failed AF).

## 6. Z devices (`Device > Z`)

Two-drive model (main manual: XY Stages and Z Drives Tips): up to two logical Z drives (main + piezo); status bar shows Z1, Z2 and **absolute Z = Z1 + Z2**. The **Active Z** (`StgZ_SetActiveZ(index)` / `StgZ_GetActiveZ()`, = Devices > Mouse Joystick and Auto Focus Z) is the drive used by **autofocus and the joystick** — this is the `focus_with_piezo` switch of **[wrapped]** `do_autofocus`. `StgZ_IsPresent(index)`, `StgZ_GetPiezoDevice()`.

* Moves: `StgMoveZ(z, rel)` (active drive, returns DR_* codes), `StgMoveMainZ`, `StgMovePiezoZ`, `StgMoveZNamed(z, rel, names, sep)`. **[wrapped]** in `set_position` (StgMoveXY / StgMoveMainZ / StgMovePiezoZ). Readback `StgGetPosZ(&z, deviceIdx)` (0 primary / 1 secondary) — **[wrapped]** in `get_position` (z0/z1).
* Piezo homing: `StgMovePiezoZToHomePosition()`, `StgMovePiezoZToHomeKeepZ()` (centers piezo, compensates via the other drive so absolute Z is unchanged).
* If piezo is selected for a Z-stack/AF and the range exceeds its travel, **main + piezo are combined automatically** (main manual: Z-series Acquisition, "Two Z devices").
* Limits: `StgZ_SetLimits(min, max)` / `StgZ_GetLimits(&min, &max)` (see quirk note §9). Speeds: motor/joystick speed get/set + step variants, speed ramp; `StgZ_SetJoystickZStep(µm)` (mouse wheel); `StgZ_MoveVirtualJoystickAtSpeed(µm/s)` — continuous Z move, 0 stops, **hardware-damage warning** in the docs; `StgToggleJoystickZ` / `StgEnableJoystickZ`.
* Maintenance: `StgZ_Escape()` (retract objective from stage; Ti2: toggling off returns Z to the original position, "Cancel" clears the escape flag staying put) and `StgZ_Refocus()`. `StgZ_SetKeepPosition(0/1)` — Ti only, and auto-disabled under PFS (§5).

## 7. ND-experiment integration (per-loop AF / PFS)

Macro params map 1:1 onto ND Acquisition dialog options:

| ND dialog option | Macro |
|---|---|
| common "Autofocus" + Define; global "Use PFS" | `ND_DefineExperiment(..., ND_UsePFS, ...)` (nis.py passes 0) |
| Time tab: AF at beginning / before each phase (single-phase caveat: "before each phase" is skipped with one phase), before/after each loop & phase commands (incl. every Nth) | `ND_SetTimeExp(..., AFType, AFWhen, AFArg1, AFArg2, cmdBefore, cmdAfter)`; getter `ND_GetTimeExp` |
| XY tab: Autofocus None / Steps in Range / Continuous; per-point PFS offsets; "Leave PFS offset ON between points"; "Redefine reference Z after AF/PFS" | `ND_SetMultipointExp(..., AFType, AFArg1, AFArg2)` (AFType 0 none / 1 steps-in-range / 3 continuous-in-range; args = step-or-speed, range); `ND_AppendMultipointPoint(x,y,z,name)` and `ND_AppendMultipointPointPFS(x,y,z,PFSoffset,name)`; `ND_SetMultiPointsEx(..., pfsArray)`; **[wrapped]** `ND_KeepPFSOnDuringStageMove(1)`; `ND_RedefineAbsZAfterPFS(0/1)` → calls `ND_RedefineAbsZ()` after each multipoint phase; `ND_OffsetAllMultipoints` (only during a running ND) |
| Lambda tab: per-channel AF; per-channel Focus Offset (parfocality) | `ND_AppendLambdaChannel(..., AFType, AFArg1, AFArg2)` — AFType 0/1/3/**4 = Focus Plane** (nis.py passes 0s) |
| Z tab: device combo box (Main/Piezo), shutter during Z move, before/after each Z loop | `ND_SetZSeriesExp(..., ND_ZSeriesDevice, cmdBefore, cmdAfter)` — **[wrapped]** via `NDAcquisition.set_z(device_name)`; getter `ND_GetZSeriesExp`; `ND_SetZSeriesType` |
| per-frame PFS offset metadata; on-image display | `ND_GetRecordingDataInt("PFS_OFFSET", seqIndex, ...)`; `AddNDInformation("PFS_OFFSET" / "PFS_STATUS")` |

## 8. Coverage in `autofrap/microscope/nis.py`

Wrapped: `do_autofocus` = `StgZ_SetActiveZ` + `StgFocusSetCriterion` + `StgFocusAdaptiveTwoPasses` + `Freeze`; Z moves/reads (`set_position`, `get_position`); `ND_KeepPFSOnDuringStageMove(1)` and the Z-device name in `ND_SetZSeriesExp` (`NDAcquisition`); `do_large_image_scan` (AF param currently 0).

Not wrapped (candidates): all PFS control (§5); `StgFocusInRange*` / `StgFocusInRangeExTwoPasses`; `StgFocusPlane` focus surfaces; `GetFocusCriterium()` readback; `ND_UsePFS`, `ND_RedefineAbsZAfterPFS`, per-point PFS offsets (`ND_AppendMultipointPointPFS`), per-loop AF in `ND_SetTimeExp` / `ND_SetMultipointExp` / `ND_AppendLambdaChannel`; `StgFocusCriteria()` for AF debugging.

## 9. Manual quirks (don't trust these blindly)

* `StgZ_GetLimits` is documented as a getter but the text says "sets the stops"; `StgZ_SetLimits` text says "retrieves" — they are getter/setter, use as such.
* `StgZ_Escape` / `StgZ_Refocus` descriptions are swapped copy-paste (the main manual's Ti2 page gives the correct semantics, §6).
* `ND_SetMultipointExp`'s `ND_MultipointUseZCoord` text: "1 The Z dimension will not be taken into account" — copy-paste bug (1 = Z taken into account, per the GUI "Include Z").
* `_SelectBestFocusedFromND` claims it prompts for the criterion despite taking it as a parameter.
* `Stg_IsPFSPlaneValid` description ("returns if the PFS on Ti is on/off") is unclear.
* Function reference (2019) vs main manual (2020): the function reference has the full function list; the main manual's command pages cover only menu commands. Focus-relevant names documented only in the main manual: `Meas_Focus_Criterion`, `GA__SelectBestFocusedFromND`.
