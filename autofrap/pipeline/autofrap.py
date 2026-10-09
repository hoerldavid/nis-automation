"""
autoFRAP pipeline
=================

Move across multiple FOVs, run one or more (survey -> FRAP) cycles per FOV.
See autofrap_loop_inner() for the per-cycle inner loop details.

Main entry point: main() -> autofrap() -> autofrap_loop_outer() -> autofrap_loop_inner().

See DESIGN_GOALS_AUTOFRAP.md for the full workflow description.

Error handling policy:
  * FOV level: any failure during a cycle fails only that FOV; run continues
    and aborts after max_consecutive_failures consecutive FOV failures.
  * Run level: AbortRunError aborts immediately for configuration/resource
    problems (NIS not running, misconfigured survey template, etc.).
  * AutofrapInterruptedException: clean user stop (Ctrl-C) - registered and causes 
    stop between cycles (or optionally after next survey acquisition).
  * Timeout assumption: all non-acquisition macros complete well within 10-20 s;
    acquisition macros run with 300 s timeout. Timeouts are treated as genuine faults

Logging: this module logs via logging.getLogger(__name__) (NullHandler, no
level set - library style). The CLI (main()) configures logging: INFO by
default, DEBUG with --verbose (per-cycle detail + the NIS macro traffic -
macro bodies, ini results, nis_ar output - logged by autofrap.microscope.nis).
Library callers that want output should call logging.basicConfig() themselves.
While a grid run is active (autofrap_loop_outer), failed macro temp files are
preserved in <run_dir>/macro_debug/ via nis_util.macro_debug_dir(run_dir).
"""

import logging
import os
import time
import argparse
import signal
import sys

import numpy as np
from skimage.measure import regionprops

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# acquisition macro timeout (seconds)
ACQUISITION_MACRO_TIMEOUT = 300  # For operations that may run long acquisitions

# NOTE: For dry-runs, the FakeNIS patcher patches at the nis *module* level,
# therefore, don't import individual functions directly (always use nis_util.fun())
import autofrap.microscope.nis as nis_util

from autofrap.core.utils.grid import grid_positions, spiral_positions
from autofrap.core.detection import (
    load_detector_file,
    parse_detector_args,
    unpack_detection_result,
)
from autofrap.core.image.mask import match_imaged_centroids, select_next_cell
from autofrap.core.image.qc import save_qc_overlay
from autofrap.core.utils.retry import run_with_retries

# cycle-number tag in output file names (e.g. <prefix>_cycle01_survey.nd2);
# spelled out rather than 'c' to avoid the color-channel reading
CYCLE_PREFIX = "cycle"

# default number of positions when visiting in the (CLI default) centre-out
# spiral order: 25 = the start position plus two square loops around it
SPIRAL_DEFAULT_POSITIONS = 25

# ND Acquisition tab names that are *not* valid for a survey image
# (multi-position, time-lapse, or large-image scans)
_SURVEY_TABS_FORBIDDEN = frozenset({"Time", "XY", "Large Image"})


def _check_nd_acq_template(tabs):
    """
    Validate an ND Acquisition tab configuration for survey use.

    Parameters
    ----------
    tabs: dict {tab_name: bool}
        result of ``nis_util.get_nd_acq_tabs()``

    Raises
    ------
    AbortRunError
        survey template is misconfigured (Time/XY/Large Image active)
    """
    forbidden = {
        tab for tab, active in tabs.items() if active and tab in _SURVEY_TABS_FORBIDDEN
    }
    if forbidden:
        raise AbortRunError(
            "survey ND template is misconfigured: "
            f'{", ".join(sorted(forbidden))} tab(s) active — '
            "a survey must be a single image with no loop"
        )


def setup_microscope(nis_exe):
    """
    Read of ND acquisition tabs, stage position and resolution.
    Check for valid configuration, return FOV info and position for subsequent grid generation.

    Returns
    -------
    pos : tuple
        (x, y, z0, z1) stage position
    res : tuple
        (xres, yres, pixel_size, magnification)

    Raises
    ------
    AbortRunError
        on repeated macro failures or if the survey ND template is misconfigured
    """
    calls = [
        (nis_util.OP_ND_ACQ_TABS, {}),
        (nis_util.OP_POSITION, {}),
        (nis_util.OP_RESOLUTION, {}),
    ]
    # short timeout for reads, retried on transient NIS/OS failures
    try:
        results = run_with_retries(
            lambda: nis_util.batch_run_macro(nis_exe, calls, timeout=10),
            "setup_microscope",
            retry_on=(KeyError, OSError, TimeoutError, RuntimeError),
        )
    except (KeyError, OSError, TimeoutError, RuntimeError) as e:
        raise AbortRunError(f"microscope setup failed after retries: {e!r}") from e
    _check_nd_acq_template(results["nd_acq_tabs_0"])
    return results["position_1"], results["resolution_2"]


def move_stage_with_retry(nis_exe, pos_xy, tolerance_um=1.0):
    """Move stage to target position with retry and verification.
    
    Parameters
    ----------
    nis_exe : str
        path to nis_ar.exe
    pos_xy : tuple
        target (x, y) position in micrometers
    tolerance_um : float
        acceptable error in micrometers (default: 1.0)
    """
    def _move_and_verify():
        # Batch set position + get position to verify we reached destination
        calls = [
            (nis_util.OP_SET_POSITION, {"x": pos_xy[0], "y": pos_xy[1]}),
            (nis_util.OP_POSITION, {}),
        ]
        results = nis_util.batch_run_macro(nis_exe, calls, timeout=10)
        actual_pos = results["position_1"]
        actual_xy = actual_pos[:2]  # (x, y)
        
        # Check if we reached destination within tolerance
        dx = abs(actual_xy[0] - pos_xy[0])
        dy = abs(actual_xy[1] - pos_xy[1])
        if dx > tolerance_um or dy > tolerance_um:
            raise RuntimeError(
                f"stage move verification failed: target ({pos_xy[0]:.1f}, {pos_xy[1]:.1f}), "
                f"actual ({actual_xy[0]:.1f}, {actual_xy[1]:.1f}), "
                f"errors ({dx:.1f}, {dy:.1f}) > tolerance {tolerance_um:.1f} um"
            )
        logger.debug(f"stage at ({actual_xy[0]:+.1f}, {actual_xy[1]:+.1f}) um "
                     f"(target ({pos_xy[0]:+.1f}, {pos_xy[1]:+.1f}))")
        return actual_xy
    
    run_with_retries(
        _move_and_verify,
        "move_stage",
        retry_on=(KeyError, OSError, TimeoutError, RuntimeError),
    )


def cleanup_run(nis_exe, start_pos, return_to_start=True):
    """Best-effort cleanup after an autoFRAP run.

    * Optionally move back to start position with retry.
    * Delete all ROIs + Close all open documents.
    
    Failures are logged but not raised (we're quitting anyway).
    """
    if return_to_start and start_pos is not None:
        try:
            move_stage_with_retry(nis_exe, start_pos[:2])
            logger.info(f"moved back to start ({start_pos[0]:+.2f}, {start_pos[1]:+.2f})")
        except Exception as e:
            logger.warning(f"could not return to start: {e!r}")

    try:
        nis_cleanup_everything(nis_exe)
    except Exception as e:
        logger.warning(f"nis_cleanup_everything failed: {e!r}")


def nis_cleanup_everything(nis_exe):
    """Best-effort thorough cleanup in NIS after each cycle:
    Delete ROIs and close all open documents in one batched macro.
    """

    def _cleanup():
        docs = nis_util.get_opened_documents(nis_exe)
        n = len(docs)
        if n == 0:
            logger.debug("cleanup_everything: no open documents")
            return
        # (delete all ROIs, close) n times -> should clean & close all
        calls = [
            (nis_util.OP_DELETE_ALL_ROIS_IN_CURRENT_DOCUMENT, {}),
            (nis_util.OP_CLOSE_CURRENT_DOCUMENT, {"save_flag": 2}),
        ] * n
        nis_util.batch_run_macro(nis_exe, calls, timeout=20)
        logger.debug(f"cleanup_everything: cleaned {n} document(s)")

    try:
        run_with_retries(_cleanup, "cleanup_everything", retry_on=TimeoutError)
    except Exception as e:
        logger.warning(f"nis_cleanup_everything failed: {e!r}")
        raise


def autofrap(
    nis_exe,
    out_dir,
    nx=2,
    ny=2,
    spacing=1.0,
    spiral=True,
    max_positions=None,
    max_cycles=None,
    detection_fun=None,
    frap_oc="FRAPPA",
    centroid_threshold="auto",
    fov_subdirs=False,
    name=None,
    use_timestamp=True,
    stop_check=None,
    allow_interrupt_after_survey=False,
    max_consecutive_failures=3,
    return_to_start=True,
    **detector_kwargs,
):
    """Outermost autoFRAP entry point: microscope setup, position
    building, the loop over positions (autofrap_loop_outer ->
    autofrap_loop_inner) and guaranteed cleanup.

    Positions form a centre-out square spiral (SPIRAL_DEFAULT_POSITIONS
    by default) around the current stage position; spiral=False selects
    the plain nx*ny row-major grid. All acquisition settings come from
    the NIS GUI (the ND acquisition definition carries the survey's
    optical configuration).

    Output: <out_dir>/<run_name>/ with per-cycle files
    <fovNN>_cycle<NN>_survey.nd2 / _frap.nd2 / _survey_qc.png
    (fov_subdirs=True: one fov<NN>/ sub-directory per FOV;
    run_name = <YYYYmmdd_HHMMSS>[_<name>]).

    Error handling and clean-stop policy: see the module docstring
    (FOV-level failures vs. AbortRunError vs. AutofrapInterruptedException).

    Parameters
    ----------
    nis_exe: str
        path to the nis_ar.exe executable
    out_dir: str
        output directory; a <run_name> sub-directory is created in it
    nx, ny: int
        grid dimensions (grid mode only, i.e. spiral=False; 1x1 = single FOV)
    spacing: float
        distance between positions in units of the field of view (1 = touching)
    spiral: bool
        True (default): centre-out square spiral over max_positions
        positions; False: plain nx*ny grid, truncated by max_positions
    max_positions: int, optional
        cap on the number of positions to visit (spiral default:
        SPIRAL_DEFAULT_POSITIONS; grid default: all of nx*ny)
    max_cycles: int, optional
        max FRAP cycles per FOV (default: until all detected cells are done)
    detection_fun: callable, required
        survey_file -> (labels[, stimulation_mask[, visualization]])
        or a bare label map - the detector contract is documented in
        WRITING_DETECTOR.md and autofrap.core.detection; a detector
        file is loaded via load_detector_file for CLI use. Without a
        stimulation mask the whole cell is FRAPed; the visualization is
        the QC-overlay background (absent -> blank canvas)
    frap_oc: str
        optical configuration to activate before each stimulation
    centroid_threshold: float or 'auto'
        centroid distance (px) for matching cells across cycles:
        'auto' (default) matches within each cell's equivalent_diameter
        (regionprops); a number sets a fixed matching radius
    fov_subdirs: bool
        give each FOV its own <run_name>/fov<NN>/ sub-directory
        (default: all FOVs in the single run directory, the position
        encoded in the file names)
    name: str, optional
        experiment name appended to the run directory name
        (<timestamp>_<name>, or exactly <name> with use_timestamp=False);
        restricted to [A-Za-z0-9._-]
    use_timestamp: bool
        prefix the run directory name with a timestamp (default True)
    stop_check: callable, optional
        zero-arg callable returning True when a clean stop was requested
        (e.g. Ctrl-C via the CLI); checked between cycles and between
        FOVs, and after survey + detection when
        allow_interrupt_after_survey is set. A stop raises
        AutofrapInterruptedException at the next safe boundary so the
        cleanup runs from a known state
    allow_interrupt_after_survey: bool
        allow the stop between detection and ROI creation instead of
        waiting for the end of the current cycle (default False; the
        cycle end is the cleanest exit state)
    max_consecutive_failures: int
        abort the run after this many FOVs failed in a row (default 3)
    return_to_start: bool
        move back to the starting position after the run (default True)
    detector_kwargs: dict, optional
        extra keyword arguments forwarded to detection_fun at each call,
        e.g. {'diameter': 30}; from the CLI these come from
        --detector-arg key=value (repeatable)

    Raises
    ------
    AbortRunError
        configuration or resource problems found before or between FOVs
        (NIS not running, misconfigured survey template, invalid
        detector, run directory collision) - the run aborts immediately
    AutofrapInterruptedException
        a clean stop was requested (stop_check) - raised at the next
        safe boundary; the run ends cleanly, not as a failure
    """
    if detection_fun is None:
        raise AbortRunError(
            "detection_fun is required - pass a detector file via "
            "--detector (see autofrap/detectors/ and WRITING_DETECTOR.md)"
        )

    # setup microscope
    start_pos, res = setup_microscope(nis_exe)
    fov = nis_util.get_fov_from_res(res)
    positions = build_positions(
        start_pos[:2],
        fov,
        nx=nx,
        ny=ny,
        spacing=spacing,
        spiral=spiral,
        max_positions=max_positions,
    )
    try:
        autofrap_loop_outer(
            nis_exe,
            out_dir,
            positions,
            max_cycles=max_cycles,
            detection_fun=detection_fun,
            frap_oc=frap_oc,
            centroid_threshold=centroid_threshold,
            fov_subdirs=fov_subdirs,
            name=name,
            use_timestamp=use_timestamp,
            stop_check=stop_check,
            allow_interrupt_after_survey=allow_interrupt_after_survey,
            max_consecutive_failures=max_consecutive_failures,
            **detector_kwargs,
        )
    finally:
        cleanup_run(nis_exe, start_pos, return_to_start=return_to_start)


class AutofrapError(Exception):
    """base class for auto-FRAP pipeline errors; raised directly for
    intentional FOV-level failures (one failed FOV - the run
    decides continue/abort via the consecutive-failure policy)"""


class AbortRunError(AutofrapError):
    """failure that makes the run impossible or pointless from where it
    stands - a configuration or resource problem found before or
    between FOVs (NIS not running, survey template misconfigured, run
    directory collision, invalid detector): the run aborts immediately"""


class AutofrapInterruptedException(AutofrapError):
    """the user requested a clean stop (Ctrl-C); raised at the next safe
    boundary (end of a cycle, or after survey + detection when
    ``allow_interrupt_after_survey`` is set)"""


def _inner_loop_do_survey(nis_exe, survey_file, cycle):
    """
    Run ND acquisition (survey image) and ensure the survey document is open & selected.
    """
    t0 = time.time()
    nis_util.run_current_nd_experiment(nis_exe, outfile=survey_file, progress_bar=True, timeout=ACQUISITION_MACRO_TIMEOUT)
    logger.info(f"[c{cycle:02d}] survey saved ({time.time() - t0:.1f} s)")
    if not os.path.isfile(survey_file):
        raise AutofrapError(
            f"survey file missing after the ND run: {survey_file} "
            "(NIS did not save it - check the GUI / disk)"
        )
    # ensure survey document is current
    doc = nis_util.get_current_document(nis_exe)
    if os.path.normcase(doc) != os.path.normcase(survey_file):
        nis_util.open_image(nis_exe, survey_file)
        doc = nis_util.get_current_document(nis_exe)
    if os.path.normcase(doc) != os.path.normcase(survey_file):
        raise AutofrapError(f"could not open {survey_file} (current document: {doc})")
    return survey_file


def _save_cycle_qc_overlay(
    viz_image, labels, stimulation_mask, cycle, out_dir, file_prefix, **qc_kwargs
):
    """
    Save the per-cycle QC overlay PNG (<file_prefix>_cycle<NN>_survey_qc.png),
    warn-and-continue on failure. Works with or without a selected cell —
    without one (qc_kwargs only holding a caption) the overlay still shows
    the image, all labels and the stimulation mask, so a "no cell found"
    FOV can be diagnosed (truly empty vs. detector thresholds too strict).
    """
    try:
        save_qc_overlay(
            viz_image,
            labels,
            os.path.join(
                out_dir, f"{file_prefix}_{CYCLE_PREFIX}{cycle:02d}_survey_qc.png"
            ),
            stimulation_mask=stimulation_mask,
            **qc_kwargs,
        )
    except Exception as e:
        logger.warning(f"[c{cycle:02d}] QC overlay failed: {e!r}")


def _inner_loop_select_cell_and_qc(
    labels,
    stimulation_mask,
    viz_image,
    imaged_centroids,
    centroid_threshold,
    cycle,
    out_dir,
    file_prefix,
):
    """Match detected objects to already-imaged map, pick next cell, build polygons and save QC overlay.
    Returns (cell, cell_poly, stim_poly, n_obj) or (None, None, None, n_obj) if no cell is available
    (a QC overlay is saved in that case too, without cell/ROI polygons).
    """

    n_obj = len(np.unique(labels)) - 1
    # match detected objects to the already-imaged centroid map
    matched = match_imaged_centroids(labels, imaged_centroids,
                                    centroid_threshold)

    cell, cell_poly, stim_poly, skipped = select_next_cell(
        labels, matched, stimulation_mask
    )
    if skipped:
        logger.info(
            f"[c{cycle:02d}] cells {skipped}: no viable polygon, skipped"
        )
    if cell is None:
        if skipped:
            logger.info(
                f"[c{cycle:02d}] all {n_obj} objects have no "
                "polygon -> move to next FOV"
            )
        else:
            logger.info(
                f"[c{cycle:02d}] {n_obj} objects, all stimulated or no "
                "stimulation mask -> stop"
            )
        _save_cycle_qc_overlay(
            viz_image, labels, stimulation_mask, cycle, out_dir, file_prefix,
            caption=(
                f"{CYCLE_PREFIX}{cycle:02d} no cell (no viable polygon)"
                if skipped
                else f"{CYCLE_PREFIX}{cycle:02d} no cell "
                "(all stimulated / no FRAP mask)"
            ),
        )
        return None, None, None, n_obj

    logger.info(f"[c{cycle:02d}] {n_obj} objects, stimulating cell {cell}")
    logger.debug(f"[c{cycle:02d}] cell {cell}: polygons with "
                 f"{len(cell_poly)} (cell) / {len(stim_poly)} (stim) vertices")
    # QC overlay before stimulation
    _save_cycle_qc_overlay(
        viz_image, labels, stimulation_mask, cycle, out_dir, file_prefix,
        cell_id=cell,
        cell_poly=cell_poly,
        stim_poly=stim_poly,
        caption=f"{CYCLE_PREFIX}{cycle:02d} cell {cell}",
    )

    return cell, cell_poly, stim_poly, n_obj


def _inner_loop_stimulation(nis_exe, frap_file, frap_oc, cell_poly, stim_poly, cycle):
    """Create ROIs in NIS, run FRAP stimulation and save the timeseries.
    Raises on failure - the exception propagates to the outer loop
    (the cycle's finally-cleanup runs and the FOV fails).
    """
    roi_calls = [
        (nis_util.OP_DELETE_ALL_ROIS_IN_CURRENT_DOCUMENT, {}),
        (nis_util.OP_CREATE_AND_SET_STIM_ROI, {"points": stim_poly}),
        (nis_util.OP_ADD_POLYGON_ROI, {"points": cell_poly}),
    ]
    roi_results = run_with_retries(
        lambda: nis_util.batch_run_macro(nis_exe, roi_calls),
        "create ROIs",
        retry_on=TimeoutError
    )
    stim_roi = roi_results["create_and_set_stim_roi_1"]
    cell_roi = roi_results["add_polygon_roi_2"]
    logger.debug(f"[c{cycle:02d}] ROIs created: stim={stim_roi}, cell={cell_roi}")

    if cell_roi <= 0:
        raise AutofrapError(f"cell ROI creation failed (id={cell_roi})")
    if stim_roi <= 0:
        raise AutofrapError(f"stim ROI creation failed (id={stim_roi})")

    # Batch set OC and run stimulation experiment
    stim_calls = [
        (nis_util.OP_SET_OPTICAL_CONFIGURATION, {"name": frap_oc}),
        (nis_util.OP_RUN_STIMULATION_EXPERIMENT, {}),
    ]
    t0 = time.time()
    nis_util.batch_run_macro(nis_exe, stim_calls, timeout=ACQUISITION_MACRO_TIMEOUT)
    logger.info(f"[c{cycle:02d}] stimulation done ({time.time() - t0:.1f} s)")

    # safeguard: move GUI focus to unsaved "ND Acquisition" (the FRAP timeseries we just did)
    # in case user selected a different open image (e.g. the survey)
    # Batch activate and save the FRAP document
    activate_save_calls = [
        (nis_util.OP_ACTIVATE_DOCUMENT, {"name": "ND Acquisition"}),
        (nis_util.OP_SAVE_CURRENT_DOCUMENT, {"outfile": frap_file}),
    ]
    run_with_retries(
        lambda: nis_util.batch_run_macro(nis_exe, activate_save_calls),
        "activate and save FRAP document",
        retry_on=TimeoutError
    )
    
    if not os.path.isfile(frap_file):
        raise AutofrapError(
            f"FRAP file missing after save_current_document: "
            f"{frap_file} (ImageSaveAs wrote nothing)"
        )


def autofrap_loop_inner(
    nis_exe,
    out_dir,
    max_cycles=None,
    detection_fun=None,
    frap_oc="FRAPPA",
    centroid_threshold="auto",
    file_prefix=None,
    stop_check=None,
    allow_interrupt_after_survey=False,
    **detector_kwargs,
):
    """Internal: the per-FOV cycle loop (a layer of autofrap(); see
    there for the shared parameters nis_exe, out_dir, max_cycles,
    detection_fun, frap_oc, centroid_threshold, stop_check,
    allow_interrupt_after_survey and detector_kwargs).

    Per cycle:
      1. acquire the survey image via the current ND experiment,
         saved to <file_prefix>_cycle<NN>_survey.nd2 and kept open
      2. run detection_fun on it (detector contract: WRITING_DETECTOR.md)
      3. pick the next cell: the smallest label not yet in the
         accumulated "already-imaged" map (centroid match via
         centroid_threshold) that has stimulation-eligible pixels and
         viable ROI polygons
      4. save the QC overlay PNG (<file_prefix>_cycle<NN>_survey_qc.png:
         detection, FRAP mask, selected cell + its polygons as sent to
         NIS) - before stimulation, so it survives it; also saved when
         no cell is selectable (without the polygons, for diagnosis)
      5. create the stimulation + whole-cell ROIs, switch to frap_oc,
         run the current sequential stimulation experiment and save
         the timeseries to <file_prefix>_cycle<NN>_frap.nd2
      6. cleanup (ROIs deleted, documents closed - finally-cleanup) and
         record the stimulated cell's centroid as imaged

    The loop stops when no cell is selectable, at max_cycles, or on a
    stop_check request: at cycle end (P1, the default, cleanest state)
    or after survey + detection (P2, opt-in via
    allow_interrupt_after_survey); a stop raises
    AutofrapInterruptedException so the finally-cleanup runs from a
    known state. Any other failure propagates to the outer loop (one
    failed FOV; see the module docstring for the failure policy).

    file_prefix: per-FOV file-name prefix; autofrap_loop_outer passes
        'fov<NN>', the default is a timestamp (standalone runs); ''
        gives plain cycle<NN>_... names
    """

    os.makedirs(out_dir, exist_ok=True)
    if file_prefix is None:
        file_prefix = time.strftime("%Y%m%d_%H%M%S")
    
    imaged_centroids = []  # list of (y, x) tuples — centroids of stimulated cells
    cycle = 0

    while max_cycles is None or cycle < max_cycles:
        # safe stop point P1: the previous cycle fully completed (ROIs
        # deleted, documents closed) — nothing is left to clean up
        if stop_check is not None and stop_check():
            raise AutofrapInterruptedException(
                f"stop requested by user after {cycle} completed cycle(s)"
            )

        cycle += 1
        survey_file = os.path.join(
            out_dir, f"{file_prefix}_{CYCLE_PREFIX}{cycle:02d}_survey.nd2"
        )
        frap_file = os.path.join(
            out_dir, f"{file_prefix}_{CYCLE_PREFIX}{cycle:02d}_frap.nd2"
        )

        try:

            # 1. acquire survey image
            _inner_loop_do_survey(nis_exe, survey_file, cycle)

            # 2a. run detection function
            try:
                detection_results = detection_fun(survey_file, **detector_kwargs)
            except Exception as e:
                # a detection failure is one failed FOV: the grid run
                # continues (the consecutive-failure policy decides
                # whether to abort)
                raise AutofrapError(f"detection failed on {survey_file}: {e!r}") from e

            # 2b. unpack detector output (the shared contract logic;
            # the detector runner uses the same function)
            try:
                labels, stimulation_mask, viz_image = unpack_detection_result(
                    detection_results
                )
            except ValueError as e:
                raise AutofrapError(f"invalid detector output: {e}") from e

            # safe stop point P2: survey acquired + detected, no ROIs
            # created yet (the finally-cleanup just closes the survey
            # document) — opt-in, end-of-cycle is the default
            if allow_interrupt_after_survey and stop_check is not None and stop_check():
                raise AutofrapInterruptedException(
                    f"stop requested by user after survey + detection "
                    f"of cycle {cycle}"
                )

            # 3. find next cell and create QC image
            cell, cell_poly, stim_poly, _ = _inner_loop_select_cell_and_qc(
                labels,
                stimulation_mask,
                viz_image,
                imaged_centroids,
                centroid_threshold,
                cycle,
                out_dir,
                file_prefix,
            )
            if cell is None:
                # no stimulatable cell or no viable polygon; move to next FOV
                break

            # 4. run stimulation / FRAP
            _inner_loop_stimulation(
                nis_exe, frap_file, frap_oc, cell_poly, stim_poly, cycle
            )

            # add the stimulated cell's centroid to the already-imaged map
            for rp in regionprops(labels):
                if rp.label == cell:
                    imaged_centroids.append(rp.centroid)  # (y, x)
                    break
        finally:
            # best-effort cleanup: close all open docs and delete ROIs,
            # so the next FOV starts from a clean GUI state; 
            nis_cleanup_everything(nis_exe)

    logger.info(f"FOV done after {cycle} cycle(s), output in {out_dir}")


def autofrap_loop_outer(
    nis_exe,
    out_dir,
    positions,
    max_cycles=None,
    detection_fun=None,
    frap_oc="FRAPPA",
    centroid_threshold="auto",
    fov_subdirs=False,
    name=None,
    use_timestamp=True,
    stop_check=None,
    allow_interrupt_after_survey=False,
    max_consecutive_failures=3,
    **detector_kwargs,
):
    """Internal: loop over the precomputed stage positions, running
    autofrap_loop_inner per FOV (a layer of autofrap(); see there for
    the shared parameters nis_exe, out_dir, max_cycles, detection_fun,
    frap_oc, centroid_threshold, stop_check,
    allow_interrupt_after_survey and detector_kwargs).

    Creates the run directory (<out_dir>/<YYYYmmdd_HHMMSS>[_<name>]; an
    existing non-empty directory aborts the run before any
    acquisition), visits the positions in order (fov_subdirs=True: one
    fov<NN>/ sub-directory per FOV), and applies the consecutive-FOV-
    failure policy (see the module docstring): a failed stage move or
    inner loop skips the position, max_consecutive_failures in a row
    abort the run with AbortRunError. stop_check is honored between
    FOVs; a stop ends the run cleanly (AutofrapInterruptedException),
    the remaining positions are not visited.

    positions: list of (x, y)
        stage positions in µm, in visit order, generated outside
        (autofrap uses build_positions; see core.utils.grid for
        grid_positions / spiral_positions)

    Raises
    ------
    AbortRunError
        invalid experiment name, positions=None, or a non-empty run
        directory (setup failures are raised by setup_microscope,
        before this function)
    AutofrapInterruptedException
        a clean stop was requested (re-raised after the summary log)
    """
    if name is not None and not all(c.isalnum() or c in "._-" for c in name):
        raise AbortRunError(
            f'invalid experiment name {name!r}: only letters, digits, "_", "." '
            'and "-" are allowed'
        )
    if positions is None:
        raise AbortRunError(
            "positions must be supplied; generate them outside autofrap()"
        )

    os.makedirs(out_dir, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if name is None:
        run_name = stamp
    else:
        run_name = f"{stamp}_{name}" if use_timestamp else name
    run_dir = os.path.join(out_dir, run_name)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise AbortRunError(
            f"run directory {run_dir} already exists and is non-empty - "
            "choose a different name or move the old run"
        )
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"run: {len(positions)} position(s)")

    # while the grid run is active, failed macro temp files are
    # preserved in <run_dir>/macro_debug/ (see nis_util.macro_debug_dir)
    with nis_util.macro_debug_dir(run_dir):
        consecutive_failures = 0
        for i, (x, y) in enumerate(positions, 1):
            fov_dir = os.path.join(run_dir, f"fov{i:02d}") if fov_subdirs else run_dir
            logger.info(
                f"=== [{i}/{len(positions)}] ({x:+.1f}, {y:+.1f}) um "
                f"-> {fov_dir} (fov{i:02d})"
            )

            try:
                # move with retry; a failed move counts as a FOV failure
                move_stage_with_retry(nis_exe, (x, y))

                autofrap_loop_inner(
                    nis_exe,
                    fov_dir,
                    max_cycles=max_cycles,
                    detection_fun=detection_fun,
                    frap_oc=frap_oc,
                    centroid_threshold=centroid_threshold,
                    file_prefix=f"fov{i:02d}",
                    stop_check=stop_check,
                    allow_interrupt_after_survey=allow_interrupt_after_survey,
                    **detector_kwargs,
                )
            except AutofrapInterruptedException as e:
                # user stop
                logger.info(
                    f"Run stopped by user: {i-1}/{len(positions)} FOV(s) done. "
                    f"output in {run_dir}"
                )
                raise e

            except Exception as e:
                # stage move or FOV failure: skip this position and abort
                # the grid once max_consecutive_failures FOVs fail in a
                # row (the failure is then systemic - disk full, NIS
                # wedged, detector down)
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        f"FOV {i} failed: {e!r} - "
                        f"{consecutive_failures} consecutive failure(s), "
                        "aborting the run"
                    )
                    raise AbortRunError()
                else:
                    logger.warning(
                        f"FOV {i} failed ({consecutive_failures}/"
                        f"{max_consecutive_failures} consecutive): {e!r} "
                        f"- moving on to the next position"
                    )
            else:
                # a completed FOV (even with zero cycles) proves NIS,
                # detection and disk work - reset the failure counter
                consecutive_failures = 0

    logger.info(f"Run done: output in {run_dir}")



def build_positions(
    start_xy, fov, nx=2, ny=2, spacing=1.0, spiral=True, max_positions=None
):
    """Generate stage positions for grid or centre-out spiral.

    Parameters
    ----------
    start_xy : tuple
        Centre stage position (x, y) in µm.
    fov : tuple
        Field of view (fov_x, fov_y) in µm.
    nx, ny : int
        Grid dimensions (grid mode only; unused in spiral mode).
    spacing : float
        Spacing in FOV units.
    spiral : bool
        If True (default), generate a centre-out square spiral;
        False for a plain NxM grid.
    max_positions : int or None
        Hard cap on number of positions. For spiral mode, if None it
        defaults to SPIRAL_DEFAULT_POSITIONS; for grid mode None means
        the full nx*ny grid.

    Returns
    -------
    positions : list of (x, y)
    """
    if spiral:
        max_pos = (
            max_positions
            if max_positions is not None
            else SPIRAL_DEFAULT_POSITIONS
        )
        positions = spiral_positions(
            start_xy, fov=fov, max_positions=max_pos, spacing=spacing
        )
    else:
        positions = grid_positions(start_xy, fov=fov, nx=nx, ny=ny, spacing=spacing)

    if max_positions is not None:
        if max_positions < 1:
            raise ValueError("--max-positions must be >= 1")
        positions = positions[:max_positions]
    return positions


def parse_cli_args(argv=None):
    p = argparse.ArgumentParser(
        description="auto-FRAP over a set of stage positions, by default "
        "a centre-out spiral (see autofrap())"
    )
    g_detection = p.add_argument_group("detection")
    g_detection.add_argument(
        "--detector",
        required=True,
        help="path to a .py file defining detection_fun "
        "(required; built-ins in autofrap/detectors/, "
        "see WRITING_DETECTOR.md)",
    )
    g_detection.add_argument(
        "--detector-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra parameter to pass to the detector, "
        "e.g. --detector-arg diameter=30 "
        "(repeatable)",
    )
    g_output = p.add_argument_group("output")
    g_output.add_argument(
        "--out",
        "-o",
        default="autofrap_out",
        help="output directory (a <run_stamp>/ sub-directory is "
        "created in it); resolved against the current "
        "working directory and passed to NIS as an "
        "absolute path [default: %(default)s]",
    )
    g_output.add_argument(
        "--name",
        help="experiment name, appended to the run directory name: "
        "<timestamp>_<name> (or exactly <name> with --no-timestamp); "
        "without it the run directory is named just <timestamp>",
    )
    g_output.add_argument(
        "--no-timestamp",
        action="store_true",
        help="name the run directory exactly --name (requires " "--name)",
    )
    g_microscope = p.add_argument_group("microscope / acquisition")
    g_microscope.add_argument(
        "--nis",
        default=r"C:\Program Files\NIS-Elements\nis_ar.exe",
        help="path to nis_ar.exe [default: %(default)s]",
    )
    g_microscope.add_argument(
        "--frap-oc",
        default="FRAPPA",
        help="optical configuration name to use for FRAP stimulation [default: %(default)s]",
    )
    g_positions = p.add_argument_group("stage positions")
    g_positions.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="grid spacing in units of FOV (1 = touching) " "[default: %(default)s]",
    )
    g_positions.add_argument(
        "--max-positions",
        "--num-positions",
        type=int,
        default=None,
        help="number of positions to visit. Default: 25 in the default "
        "spiral order (the start position plus two loops around it); "
        "in --grid mode, all of the nx*ny grid unless given as a cap "
        "[default: spiral: 25, grid: all]",
    )
    g_positions.add_argument(
        "--grid",
        action="store_true",
        help="visit positions in a plain NxM grid (--nx, --ny) in "
        "row-major order instead of the default centre-out square "
        "spiral; --max-positions truncates the grid",
    )
    g_positions.add_argument(
        "--nx",
        type=int,
        default=2,
        help="grid size in x (--grid mode; 1 = single FOV) [default: %(default)s]",
    )
    g_positions.add_argument(
        "--ny",
        type=int,
        default=2,
        help="grid size in y (--grid mode; 1 = single FOV) [default: %(default)s]",
    )
    g_cycles = p.add_argument_group("cycles per FOV")
    g_cycles.add_argument(
        "--max-cycles",
        type=int,
        default=1,
        help="max FRAP cycles per FOV [default: %(default)s]",
    )
    g_cycles.add_argument(
        "--until-done",
        action="store_true",
        help="run until all cells of a FOV are stimulated " "(ignore --max-cycles)",
    )
    g_behavior = p.add_argument_group("run behavior")
    g_behavior.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="abort the run after this many consecutive "
        "FOV failures [default: %(default)s]",
    )
    g_behavior.add_argument(
        "--allow-interrupt-after-survey",
        action="store_true",
        help="allow Ctrl-C to stop after survey + detection "
        "(opt-in; otherwise waits for end of cycle)",
    )
    g_behavior.add_argument(
        "--no-return",
        action="store_true",
        help="don't move back to the start position after the run",
    )
    g_logging = p.add_argument_group("logging")
    g_logging.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="DEBUG logging: per-cycle detail plus the NIS macro traffic "
        "(macro bodies, ini results, nis_ar output); failed macros are "
        "always preserved in <run_dir>/macro_debug/ [default: INFO]",
    )
    args = p.parse_args(argv)
    if args.no_timestamp and not args.name:
        p.error("--no-timestamp requires --name")
    return args


def main(argv=None):
    """CLI entry point: parse arguments, load the detector, run
    autofrap() and translate its exit conditions into exit codes:

      0   run finished (fully or partially)
      1   AbortRunError (configuration / resource problem)
      130 AutofrapInterruptedException (clean user stop)
    """
    # Ctrl-C handling: first press requests a clean stop at the next safe
    # boundary (end of cycle / between FOVs - the current macro call
    # runs to completion, we never kill it); a second press raises
    # KeyboardInterrupt immediately (the finally-cleanup still runs)
    _stop = {"requested": False, "count": 0}

    def _on_sigint(signum, frame):
        _stop["count"] += 1
        if _stop["count"] == 1:
            _stop["requested"] = True
            logger.info("Ctrl-C: stopping after the current cycle "
                        "(press again to interrupt immediately)")
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    args = parse_cli_args(argv)
    # the CLI owns the logging configuration (the pipeline modules only
    # attach NullHandlers - library style)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if args.verbose:
        # third-party DEBUG is chatty (matplotlib font matching, ...);
        # --verbose is about our own DEBUG output
        logging.getLogger('matplotlib').setLevel(logging.INFO)
    # NIS macros resolve relative paths against the NIS executable's
    # directory - the pipeline must hand them absolute paths
    args.out = os.path.abspath(args.out)

    logger.info(f"loading detector from: {args.detector}")
    detection_fun = load_detector_file(args.detector)

    detector_kwargs = {}
    try:
        detector_kwargs = parse_detector_args(args.detector_arg)
    except ValueError as e:
        logger.error(f"--detector-arg {e}")
        sys.exit(1)

    try:
        autofrap(
            args.nis,
            args.out,
            nx=args.nx,
            ny=args.ny,
            spacing=args.spacing,
            spiral=not args.grid,
            max_positions=args.max_positions,
            max_cycles=None if args.until_done else args.max_cycles,
            detection_fun=detection_fun,
            frap_oc=args.frap_oc,
            name=args.name,
            use_timestamp=not args.no_timestamp,
            stop_check=lambda: _stop["requested"],
            allow_interrupt_after_survey=args.allow_interrupt_after_survey,
            max_consecutive_failures=args.max_consecutive_failures,
            return_to_start=not args.no_return,
            **detector_kwargs,
        )
    except AutofrapInterruptedException:
        sys.exit(130)
    except AbortRunError as e:
        logger.error(f"{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
