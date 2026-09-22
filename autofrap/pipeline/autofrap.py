"""
Auto-FRAP inner loop: survey -> detect -> pick unused cell -> stimulate -> repeat.

All acquisition settings come from the NIS GUI (the ND acquisition
definition carries the survey's optical configuration); this script
just runs them in a loop.

Per cycle:
  1. run the current ND experiment, saved to
     <file_prefix>_cycle<NN>_survey.nd2 (file_prefix defaults to a
     timestamp for standalone runs; autofrap_loop_outer passes 'fov<NN>')
  2. detect objects in the survey image (detection_fun — by default
     cellpose on the GPU server via cellpose_server.py) — returns
     (labels[, stimulation_mask[, visualization]]): only the label
     map is required (a bare label map is accepted); without a
     stimulation mask the whole cell is FRAPed, the visualization is
     used for the QC overlay only
  3. match detected objects to the accumulated "already-imaged" map
     via centroid distance; pick the smallest unmatched label that
     has at least one pixel in the stimulation mask
  4. compute the ROI polygons and save a QC overlay PNG
     (<file_prefix>_cycle<NN>_survey_qc.png: detection, FRAP mask, selected
     cell, polygons as sent to NIS — on a blank canvas when the
     detector provides no visualization); warn-and-continue on
     failure, saved before the stimulation run so it survives it
  5. open the survey image in NIS, add two ROIs: the whole cell
     (for downstream analysis) and the stimulation region
     ((labels == cell_id) & stimulation_mask), the latter set to
     stimulation mode (type 3)
  6. switch optical conf to FRAPPA, run the current sequential
     stimulation experiment
  7. save the FRAP timeseries to <file_prefix>_cycle<NN>_frap.nd2 (the
     stimulation ROI is part of the saved file)
  8. delete both ROIs (so they don't linger for the next cycle) and
     close the FRAP + survey documents
  9. add the stimulated cell's centroid to the "already-imaged" map
-> next cycle (the ND experiment definition restores the survey OC)

The loop stops when every detected object has been stimulated, when
no cell has stimulation-eligible pixels, when max_cycles is reached, or
when the user requests a clean stop (stop_check, e.g. Ctrl-C via the
CLI): the run then ends at the next safe boundary (end of a cycle,
or after survey + detection with allow_interrupt_after_survey) with
the usual finally-cleanup, and a grid run reports 'stopped by user'
instead of aborting.

Error handling: every failure is translated into one of two exception
classes - RecoverableError (this FOV is lost, a grid run may continue)
or NonRecoverableError (the microscope/detection state is unknown or
broken, a grid run should abort). A failed cycle best-effort deletes
its own ROIs and closes its documents before re-raising, so a grid run
that continues starts the next FOV from a clean GUI state.
AutofrapInterruptedException is not a failure - it only carries a
requested stop to the point that can act on it.
"""
import os
import time
import argparse
import signal
import sys

import numpy as np
from skimage.measure import regionprops

# NOTE: For dry-runs, the FakeNIS patcher patches at the nis *module* level,
# therefore, don't import individual functions directly (always use nis_util.fun())
import autofrap.microscope.nis as nis_util

from autofrap.core.detection import load_detector_file
from autofrap.core.image.mask import mask_to_polygon, cell_mask
from autofrap.core.image.qc import save_qc_overlay


# cycle-number tag in output file names (<prefix>_cycle01_survey.nd2);
# spelled out rather than 'c' to avoid the color-channel reading
CYCLE_PREFIX = 'cycle'

# ND Acquisition tab names that are *not* valid for a survey image
# (multi-position, time-lapse, or large-image scans)
_SURVEY_TABS_FORBIDDEN = frozenset({'Time', 'XY', 'Large Image'})


def _check_nd_acq_template(tabs):
    """
    Validate an ND Acquisition tab configuration for survey use.

    Parameters
    ----------
    tabs: dict {tab_name: bool}
        result of ``nis_util.get_nd_acq_tabs()``

    Returns
    -------
    tabs: dict
        the input dict unchanged (for convenient use as a passthrough)

    Raises
    ------
    NonRecoverableError
        survey template is misconfigured (Time/XY/Large Image active)
    """
    forbidden = {tab for tab, active in tabs.items()
                 if active and tab in _SURVEY_TABS_FORBIDDEN}
    if forbidden:
        raise NonRecoverableError(
            'survey ND template is misconfigured: '
            f'{", ".join(sorted(forbidden))} tab(s) active — '
            'a survey must be a single image with no loop')
    return tabs


def setup_microscope(nis_exe):
    """Batch read of ND acquisition tabs, stage position and resolution.

    Returns
    -------
    pos : tuple
        (x, y, z0, z1) stage position
    res : tuple
        (xres, yres, pixel_size, magnification)
    """
    calls = [
        (nis_util._OP_ND_ACQ_TABS, {}),
        (nis_util._OP_POSITION, {}),
        (nis_util._OP_RESOLUTION, {}),
    ]
    # short timeout for reads with retry
    last_exc = None
    for attempt, delay in enumerate([0, 2, 4], start=1):
        try:
            if delay:
                time.sleep(delay)
            results = nis_util.batch_run_macro(nis_exe, calls, timeout=10)
            tabs = results['nd_acq_tabs_0']
            pos = results['position_1']
            res = results['resolution_2']
            _check_nd_acq_template(tabs)
            if attempt > 1:
                print(f'[setup_microscope] succeeded on attempt {attempt}', flush=True)
            return pos, res
        except Exception as e:
            last_exc = e
            if attempt == 3:
                break
            print(f'[setup_microscope] attempt {attempt} failed: {e!r}, retrying in {delay}s', flush=True)
    raise last_exc


def move_stage_with_retry(nis_exe, pos_xy):
    """Move stage with retry on timeout / KeyError / OSError.

    Retries 3 times with delays 0s, 2s, 4s.
    """
    last_err = None
    for attempt, delay in enumerate([0, 2, 4], start=1):
        try:
            if delay:
                time.sleep(delay)

            # TODO: do a set_pos + get_pos batch, check if we reached destination (+- a few micron tolerance)?
            nis_util.set_position(nis_exe, pos_xy=pos_xy)
            if attempt > 1:
                print(f'[move_stage] succeeded on attempt {attempt}', flush=True)
            return
        except (KeyError, OSError, TimeoutError) as e:
            last_err = e
            if attempt == 3:
                break
            print(f'[move_stage] attempt {attempt} failed: {e!r}, retry in {delay}s', flush=True)
    raise last_err


def cleanup_run(nis_exe, start_pos, return_to_start=True):
    """Best-effort cleanup after a grid run.

    * Optionally move back to start position with retry.
    * Delete all ROIs in current document.
    * Close all open documents.
    All operations are idempotent and failures are logged but not raised.
    """
    if return_to_start and start_pos is not None:
        try:
            move_stage_with_retry(nis_exe, start_pos[:2])
            print(f'moved back to start ({start_pos[0]:+.2f}, {start_pos[1]:+.2f})')
        except Exception as e:
            print(f'!!! could not return to start: {e!r}', flush=True)

    # Idempotent cleanup
    try:
        nis_cleanup_everything(nis_exe)
    except Exception as e:
        print(f'!!! cleanup_everything failed: {e!r}', flush=True)


def nis_cleanup_everything(nis_exe):
    """Best-effort thorough cleanup in NIS:
    Delete ROIs and close all open documents in one batched macro.
    Retries on TimeoutError.
    """
    last_exc = None
    for attempt, delay in enumerate([0, 2, 4], start=1):
        try:
            if delay:
                time.sleep(delay)
            docs = nis_util.get_opened_documents(nis_exe)
            n = len(docs)
            if n == 0:
                print('[cleanup_everything] no open documents')
                return
            calls = [
                (nis_util._OP_DELETE_ALL_ROIS_IN_CURRENT_DOCUMENT, {}),
                (nis_util._OP_CLOSE_CURRENT_DOCUMENT, {'save_flag': 2}),
            ] * n
            nis_util.batch_run_macro(nis_exe, calls, timeout=20)
            print(f'[cleanup_everything] cleaned {n} document(s)')
            return
        except TimeoutError as e:
            last_exc = e
            print(f'[cleanup_everything] attempt {attempt} timed out: {e!r}', flush=True)
            if attempt == 3:
                break
        except Exception as e:
            print(f'!!! cleanup_everything error: {e!r}', flush=True)
            return
    if last_exc:
        print(f'!!! cleanup_everything failed after retries: {last_exc!r}', flush=True)


def autofrap(nis_exe, out_dir,
            nx=2, ny=2, spacing=1.0,
            spiral=False, max_positions=None,
            max_cycles=None,
            detection_fun=None,
            frap_oc='FRAPPA',
            centroid_threshold='auto',
            fov_subdirs=False,
            name=None, use_timestamp=True,
            stop_check=None,
            allow_interrupt_after_survey=False,
            return_to_start=True,
            **detector_kwargs):
    """Outermost autoFRAP entry point.

    Performs setup, builds positions from grid parameters, runs the outer
    loop over positions and guarantees cleanup.
    """
    # setup microscope
    start_pos, res = setup_microscope(nis_exe)
    fov = nis_util.get_fov_from_res(res)
    positions = build_positions(
        start_pos[:2], fov,
        nx=nx, ny=ny, spacing=spacing,
        spiral=spiral, max_positions=max_positions
    )
    try:
        results = autofrap_loop_outer(
            nis_exe, out_dir, positions,
            max_cycles=max_cycles,
            detection_fun=detection_fun,
            frap_oc=frap_oc,
            centroid_threshold=centroid_threshold,
            fov_subdirs=fov_subdirs,
            name=name, use_timestamp=use_timestamp,
            stop_check=stop_check,
            allow_interrupt_after_survey=allow_interrupt_after_survey,
            **detector_kwargs
        )
    finally:
        cleanup_run(nis_exe, start_pos, return_to_start=return_to_start)


    return results


class AutofrapError(Exception):
    """base class for auto-FRAP pipeline errors"""


class RecoverableError(AutofrapError):
    """failure confined to the current FOV (no polygon for the cell, ROI
    creation failed); a run can continue with the next position"""


class NonRecoverableError(AutofrapError):
    """failure that makes further FOVs pointless or unsafe (NIS state
    unknown, detection failed - the detector/server state is suspect,
    disk full); a run aborts"""


class AutofrapInterruptedException(AutofrapError):
    """the user requested a clean stop (Ctrl-C); raised at the next safe
    boundary (end of a cycle, or after survey + detection when
    ``allow_interrupt_after_survey`` is set), so the ``finally`` cleanup
    runs from a known state; the grid stops (it is not a failure)"""


# TODO: move to mask functions?
def next_stimulatable_cell(labels, stimulated, stimulation_mask=None):
    """
    Find the next unstimulated cell (smallest label first).

    Iterates over labels in sorted order. For each candidate label
    that is not in the stimulated set, checks whether it has any pixels
    in the stimulation mask (if one is given); candidates without
    stimulation-eligible pixels are skipped.

    Parameters
    ----------
    labels: 2D np.ndarray
        label map (0 = background, 1..N = objects)
    stimulated: set of int
        already-stimulated cell IDs
    stimulation_mask: 2D np.ndarray, optional
        binary mask of areas eligible for photostimulation; if given,
        cells without any pixels in it are skipped

    Returns
    -------
    cell_id: int or None
        the next stimulatable cell, or None if none found
    """
    for lbl in sorted(np.unique(labels).tolist()):
        if lbl > 0 and lbl not in stimulated:
            if stimulation_mask is None or np.any((labels == lbl) & stimulation_mask):
                return lbl
    return None


def _inner_loop_do_survey(nis_exe, survey_file, cycle):
    """Run ND survey acquisition and ensure the survey document is current."""
    t0 = time.time()
    nis_util.run_current_nd_experiment(nis_exe, outfile=survey_file, progress_bar=True)
    print(f'[c{cycle:02d}] survey saved ({time.time() - t0:.1f} s)', flush=True)
    if not os.path.isfile(survey_file):
        raise NonRecoverableError(
            f'survey file missing after the ND run: {survey_file} '
            '(NIS did not save it - check the GUI / disk)'
        )
    # ensure survey document is current
    doc = nis_util.get_current_document(nis_exe)
    if os.path.normcase(doc) != os.path.normcase(survey_file):
        nis_util.open_image(nis_exe, survey_file)
        doc = nis_util.get_current_document(nis_exe)
    if os.path.normcase(doc) != os.path.normcase(survey_file):
        raise NonRecoverableError(
            f'could not open {survey_file} (current document: {doc})'
        )
    return survey_file


def _inner_loop_select_cell_and_qc(labels, stimulation_mask, viz_image, imaged_centroids, centroid_threshold, cycle, out_dir, file_prefix):
    """Match detected objects to already-imaged map, pick next cell, build polygons and save QC overlay.
    Returns (cell, cell_poly, stim_poly, n_obj) or (None, None, None, n_obj) if no cell is available.
    """

    n_obj = len(np.unique(labels)) - 1
    # match detected objects to already-imaged map
    if imaged_centroids:
        matched = set()
        for rp in regionprops(labels):
            cy, cx = rp.centroid
            if centroid_threshold == 'auto':
                radius = rp.equivalent_diameter_area
            else:
                radius = centroid_threshold
            for iy, ix in imaged_centroids:
                if (cy - iy)**2 + (cx - ix)**2 < radius**2:
                    matched.add(rp.label)
                    break
    else:
        matched = set()

    cell = next_stimulatable_cell(labels, matched, stimulation_mask)
    if cell is None:
        print(
            f'[c{cycle:02d}] {n_obj} objects, all stimulated or no '
            'stimulation mask -> stop'
        )
        return None, None, None, n_obj

    skipped = set()
    while True:
        cell_poly = mask_to_polygon(cell_mask(labels, cell))
        stim_poly = mask_to_polygon(cell_mask(labels, cell, stimulation_mask))
        if cell_poly and stim_poly:
            break
        skipped.add(cell)
        print(f'[c{cycle:02d}] cell {cell}: no polygon, skipping')
        cell = next_stimulatable_cell(labels, matched | skipped, stimulation_mask)
        if cell is None:
            print(
                f'[c{cycle:02d}] all {n_obj} objects have no '
                'polygon -> move to next FOV'
            )
            return None, None, None, n_obj

    print(f'[c{cycle:02d}] {n_obj} objects, stimulating cell {cell}')
    # QC overlay before stimulation
    try:
        save_qc_overlay(
            viz_image, labels,
            os.path.join(out_dir, f'{file_prefix}_{CYCLE_PREFIX}{cycle:02d}_survey_qc.png'),
            stimulation_mask=stimulation_mask, cell_id=cell,
            cell_poly=cell_poly, stim_poly=stim_poly,
            caption=f'{CYCLE_PREFIX}{cycle:02d} cell {cell}'
        )
    except Exception as e:
        print(f'[c{cycle:02d}] WARNING: QC overlay failed: {e!r}', flush=True)

    return cell, cell_poly, stim_poly, n_obj


def _inner_loop_stimulation(nis_exe, frap_file, frap_oc, cell_poly, stim_poly, cycle):
    """Create ROIs, run FRAP stimulation and save the timeseries.
    Raises RecoverableError / NonRecoverableError on failure.
    """
    roi_calls = [
        (nis_util._OP_DELETE_ALL_ROIS_IN_CURRENT_DOCUMENT, {}),
        (nis_util._OP_ADD_POLYGON_ROI, {'points': cell_poly}),
        (nis_util._OP_ADD_POLYGON_ROI, {'points': stim_poly}),
    ]
    try:
        roi_results = nis_util.batch_run_macro(nis_exe, roi_calls)
        cell_roi = roi_results['add_polygon_roi_1']
        stim_roi = roi_results['add_polygon_roi_2']
    except TimeoutError:
        roi_results = nis_util.batch_run_macro(nis_exe, roi_calls)
        cell_roi = roi_results['add_polygon_roi_1']
        stim_roi = roi_results['add_polygon_roi_2']

    if cell_roi <= 0:
        raise RecoverableError(f'cell ROI creation failed (id={cell_roi})')
    if stim_roi <= 0:
        raise RecoverableError(f'stim ROI creation failed (id={stim_roi})')

    nis_util.set_roi_type(nis_exe, stim_roi, 3)

    nis_util.set_optical_configuration(nis_exe, frap_oc)
    t0 = time.time()
    nis_util.run_stimulation_experiment(nis_exe)
    print(f'[c{cycle:02d}] stimulation done ({time.time() - t0:.1f} s)', flush=True)

    # safeguard: move GUI focus off the survey / FRAP documents to an unsaved ND Acquisition
    # to avoid accidental user interaction with ROIs while the next cycle prepares
    try:
        nis_util.activate_document(nis_exe, 'ND Acquisition')
    except Exception:
        # ND Acquisition should always be present; ignore if activation fails
        pass

    nis_util.save_current_document(nis_exe, frap_file)
    if not os.path.isfile(frap_file):
        raise NonRecoverableError(
            f'FRAP file missing after save_current_document: '
            f'{frap_file} (ImageSaveAs wrote nothing)'
        )


def autofrap_loop_inner(nis_exe, out_dir, max_cycles=None, detection_fun=None,
             frap_oc='FRAPPA', centroid_threshold='auto',
             file_prefix=None, stop_check=None,
             allow_interrupt_after_survey=False, **detector_kwargs):
    """
    run the auto-FRAP loop

    Parameters
    ----------
    nis_exe: str
        path to the nis_ar.exe executable
    out_dir: str
        output directory for survey + FRAP files
    max_cycles: int, optional
        stop after this many cycles (default: until all cells done)
    detection_fun: callable, required
        survey_file -> (labels[, stimulation_mask[, visualization]])
        or a bare label map; only the label map is required.
        stimulation_mask (FRAP sub-regions): None or absent -> the
        whole cell is FRAPed.
        visualization (2D or RGB(A), detector-assembled, e.g.
        multi-channel): used for the QC overlay only; absent -> the
        overlay is drawn on a blank canvas (autofrap() does not know
        which channel(s) the detector used). A detector file is
        loaded via :func:`detection.load_detector_file` for CLI use;
    frap_oc: str
        optical configuration to activate before each stimulation
    centroid_threshold: float or 'auto'
        centroid distance threshold (px) for matching cells across
        consecutive cycles.  ``'auto'`` (default): uses each cell's
        ``equivalent_diameter`` from ``regionprops`` — a matched cell
        is one whose centroid lies within one equivalent-diameter of
        a previously stimulated cell's centroid.  A numeric value
        overrides this heuristic with a fixed radius.
    file_prefix: str, optional
        prefix for the per-cycle file names
        (<file_prefix>_cycle<NN>_survey.nd2, ...); default: a timestamp
        (YYYYmmdd_HHMMSS) for standalone runs — autofrap_loop_outer passes
        'fov<NN>' per position. Set to '' for plain cycle<NN>_... names.
    stop_check: callable, optional
        zero-arg callable returning True when a clean stop was requested
        (e.g. by Ctrl-C); checked at the start of each cycle (after the
        previous cycle fully completed — ROIs deleted, documents
        closed) and, when allow_interrupt_after_survey is True, right
        after survey + detection (before any ROI is created). A stop
        raises AutofrapInterruptedException so the finally-cleanup runs
        from a known state.
    allow_interrupt_after_survey: bool
        allow the stop between detection and ROI creation (default
        False — the stop always waits for the end of the current cycle,
        which is the cleanest exit state).
    detector_kwargs: dict, optional
        extra keyword arguments forwarded to ``detection_fun`` at each
        call, e.g. ``{'diameter': 30, 'channel': 0}``.  From the CLI
        these come from ``--detector-arg key=value`` (repeatable).

    Returns
    -------
    results: list of (cycle, cell, survey_file, frap_file)

    Raises
    ------
    RecoverableError
        this FOV could not be processed (no polygon for the cell, ROI
        creation failed)
    NonRecoverableError
        the state is unknown or broken (survey/FRAP file not saved, NIS
        macro aborted, detection failed for any reason - the
        detector/server state is suspect, OS error); further cycles are
        unlikely to succeed
    AutofrapInterruptedException
        a clean stop was requested (stop_check) and the next safe
        boundary was reached; the grid run stops, this is not a failure
    """

    os.makedirs(out_dir, exist_ok=True)
    if file_prefix is None:
        file_prefix = time.strftime('%Y%m%d_%H%M%S')
    imaged_centroids = []  # list of (y, x) tuples — centroids of stimulated cells
    results = []

    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        # safe stop point P1: the previous cycle fully completed (ROIs
        # deleted, documents closed) — nothing is left to clean up
        if stop_check is not None and stop_check():
            raise AutofrapInterruptedException(
                f'stop requested by user after {cycle} completed cycle(s)')

        cycle += 1
        survey_file = os.path.join(
            out_dir, f'{file_prefix}_{CYCLE_PREFIX}{cycle:02d}_survey.nd2')
        frap_file = os.path.join(
            out_dir, f'{file_prefix}_{CYCLE_PREFIX}{cycle:02d}_frap.nd2')

        try:

            # 1. acquire survey image
            _inner_loop_do_survey(nis_exe, survey_file, cycle)

            # run detection function 
            try:
                det = detection_fun(survey_file, **detector_kwargs)
            except Exception as e:
                # TODO: detection failure -> recoverable error & continue with next FOV?
                raise NonRecoverableError(
                    f'detection failed on {survey_file}: {e!r}') from e

            # 2. unpack detector output
            # a bare label map is accepted (normalized to a 1-tuple);
            # otherwise: a 1-3 tuple/list (labels[, mask[, viz]])
            if isinstance(det, np.ndarray):
                det = (det,)
            if (not isinstance(det, (tuple, list)) or not 1 <= len(det) <= 3):
                raise NonRecoverableError(
                    f'detection_fun returned {type(det).__name__}; expected '
                    '(labels[, stimulation_mask[, visualization]])')
            labels = det[0]
            stimulation_mask = det[1] if len(det) > 1 else None
            viz_image = det[2] if len(det) > 2 else None

            # safe stop point P2: survey acquired + detected, no ROIs
            # created yet (the finally-cleanup just closes the survey
            # document) — opt-in, end-of-cycle is the default
            if (allow_interrupt_after_survey and stop_check is not None
                    and stop_check()):
                raise AutofrapInterruptedException(
                    f'stop requested by user after survey + detection '
                    f'of cycle {cycle}')

            # 3. find next cell and create QC image
            cell, cell_poly, stim_poly, _ = _inner_loop_select_cell_and_qc(
                labels, stimulation_mask, viz_image, imaged_centroids,
                centroid_threshold, cycle, out_dir, file_prefix
            )
            if cell is None:
                # no stimulatable cell or no viable polygon; move to next FOV
                break

            # 4. run stimulation / FRAP
            _inner_loop_stimulation(nis_exe, frap_file, frap_oc, cell_poly, stim_poly, cycle)

            results.append((cycle, cell, survey_file, frap_file))
            # add the stimulated cell's centroid to the already-imaged map
            for rp in regionprops(labels):
                if rp.label == cell:
                    imaged_centroids.append(rp.centroid)  # (y, x)
                    break
        except (RecoverableError, NonRecoverableError):
            raise
        except TimeoutError as e:
            # permissive: treat macro timeout as recoverable for this FOV
            # finally block will clean ROIs / close docs
            raise RecoverableError(
                f'NIS macro timed out: {e}. Skipping this FOV.'
            ) from e
        except KeyError as e:
            # an empty ini read-back means the NIS macro aborted partway -
            # the GUI state is now unknown, so don't queue more FOVs on top
            raise NonRecoverableError(
                f'NIS macro failed (no read-back: {e!r}) - '
                'the NIS state is now unknown') from e
        except OSError as e:
            raise NonRecoverableError(f'OS error: {e!r}') from e
        finally:
            # best-effort cleanup: close all open docs and delete ROIs
            try:
                nis_cleanup_everything(nis_exe)
            except Exception:
                pass

    print(f'\nDone: {len(results)} cell(s) stimulated in {cycle} cycle(s), output in {out_dir}')
    return results


# TODO: move to core.utils.grid?
def grid_positions(position, fov, nx=2, ny=2, spacing=1.0):
    """
    compute a grid of stage positions centered on the given position

    Parameters
    ----------
    position: (x, y)
        center of the grid (e.g. the current stage position)
    fov: (fov_x, fov_y)
        field of view per axis (see nis_util.get_fov_from_res)
    nx, ny: int
        number of grid positions in x and y
    spacing: float
        distance between neighboring positions in units of FOV size:
        1 -> touching (non-overlapping) FOVs,
        <1 -> overlapping FOVs,
        >1 -> non-overlapping FOVs with a gap

    Returns
    -------
    positions: list of 2-tuples
        (x, y) stage positions, row-major order
    """
    fov_x, fov_y = fov
    x0, y0 = position
    step_x = spacing * fov_x
    step_y = spacing * fov_y

    return [(x0 + (i - (nx - 1) / 2) * step_x,
             y0 + (j - (ny - 1) / 2) * step_y)
            for j in range(ny) for i in range(nx)]


def autofrap_loop_outer(nis_exe, out_dir, positions,
                  max_cycles=None,
                  detection_fun=None, frap_oc='FRAPPA',
                  centroid_threshold='auto',
                  fov_subdirs=False, name=None, use_timestamp=True,
                  stop_check=None, allow_interrupt_after_survey=False,
                  **detector_kwargs):
    """
    Go over multiple stage positions / FOVs and run one or more autoFRAP cycles at each.

    By default all FOVs are written to a single run directory; the
    'fov<NN>' file prefix (matching the log lines) keeps files
    self-describing and makes one folder easy to browse (QC PNGs side
    by side) or to hand to downstream analysis:

        <out_dir>/<run_name>/
            <fovNN>_cycleNN_survey.nd2
            <fovNN>_cycleNN_frap.nd2
            <fovNN>_cycleNN_survey_qc.png

    run_name is a timestamp (<YYYYmmdd_HHMMSS>) by default, or
    <timestamp>_<name> when name is given (and <name> alone when
    use_timestamp=False). An existing non-empty run directory aborts
    the run before any acquisition (an empty one is reused).

    With fov_subdirs=True, each FOV goes into its own sub-directory
    instead (<run_stamp>/fov<NN>/, same file names).

    Parameters
    ----------
    nis_exe, out_dir: str
        as in autofrap(); a <run_stamp> sub-directory is created in
        out_dir for this grid run
    positions: list of (x, y)
        precomputed stage positions in visit order; the list is generated
        outside (e.g. via :func:`grid_positions` for a plain NxM grid or
        :func:`spiral_positions` for a centre‑out spiral).
    return_to_start: bool
        move back to the starting position after the last FOV
    max_cycles, detection_fun, frap_oc, centroid_threshold:
        passed through to autofrap() unchanged
    fov_subdirs: bool
        give each FOV its own <run_name>/fov<NN>/ sub-directory
        (default: all FOVs in the single run directory, position
        encoded in the file names)
    name: str, optional
        experiment name appended to the run directory name (see
        above); restricted to [A-Za-z0-9._-]
    use_timestamp: bool
        prefix the run directory name with a timestamp (default True);
        only meaningful together with name
    stop_check, allow_interrupt_after_survey:
        passed through to autofrap_loop_inner() unchanged; additionally the grid
        checks stop_check between FOVs. A requested stop stops the run
        at the next safe boundary (AutofrapInterruptedException from
        autofrap_loop_inner()) — this is not a failure: the remaining positions
        are simply not visited and the partial results are returned
    detector_kwargs: dict, optional
        extra keyword arguments forwarded to ``autofrap_loop_inner`` →
        ``detection_fun`` (see :func:`autofrap_loop_inner` for details); from the
        CLI these come from ``--detector-arg key=value`` (repeatable)

    Returns
    -------
    results: list of (i, x, y, fov_dir, fov_results)
        fov_results is autofrap_loop_inner's per-cycle results, or None if that FOV
        failed; fov_dir is the per-FOV sub-directory (fov_subdirs=True)
        or the shared run directory. A RecoverableError skips the FOV
        and continues; a NonRecoverableError aborts the run (the
        remaining positions are not visited and do not appear in
        results)

    Raises
    ------
    NonRecoverableError
        if the starting stage position cannot be read, the ND
        Acquisition template is misconfigured, the detector name is
        invalid, the run directory already exists and is non-empty
    """
    if name is not None and not all(c.isalnum() or c in '._-'
                                    for c in name):
        raise NonRecoverableError(
            f'invalid experiment name {name!r}: only letters, digits, "_", "." '
            'and "-" are allowed')
    os.makedirs(out_dir, exist_ok=True)
    if positions is None:
        raise NonRecoverableError(
            'positions must be supplied; generate them outside autofrap()')
    # positions are now required to be pre-computed
    stamp = time.strftime('%Y%m%d_%H%M%S')
    if name is None:
        run_name = stamp
    else:
        run_name = f'{stamp}_{name}' if use_timestamp else name
    run_dir = os.path.join(out_dir, run_name)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise NonRecoverableError(
            f'run directory {run_dir} already exists and is non-empty - '
            'choose a different name or move the old run')
    os.makedirs(run_dir, exist_ok=True)

    print(f'grid: {len(positions)} position(s)')
    results = []
    aborted = None
    stopped = False
    for i, (x, y) in enumerate(positions, 1):
            fov_dir = (os.path.join(run_dir, f'fov{i:02d}')
                       if fov_subdirs else run_dir)
            print(f'\n=== [{i}/{len(positions)}] ({x:+.1f}, {y:+.1f}) um '
                  f'-> {fov_dir} (fov{i:02d})',
                  flush=True)

            # move with retry
            try:
                move_stage_with_retry(nis_exe, (x, y))
            except (KeyError, OSError, TimeoutError) as e:
                print(f'!!! FOV {i}: stage move failed after retries: {e!r} - aborting the grid run', flush=True)
                results.append((i, x, y, fov_dir, None))
                aborted = i
                break

            # set_position blocks until the stage has arrived (verified
            # on scope 20260909) - no settling wait needed

            try:
                fov_results = autofrap_loop_inner(
                    nis_exe, fov_dir, max_cycles=max_cycles,
                    detection_fun=detection_fun,
                    frap_oc=frap_oc,
                    centroid_threshold=centroid_threshold,
                    file_prefix=f'fov{i:02d}', stop_check=stop_check,
                    allow_interrupt_after_survey=allow_interrupt_after_survey,
                    **detector_kwargs)
            except NonRecoverableError as e:
                print(f'!!! FOV {i}: non-recoverable error: {e} '
                      f'- aborting the grid run', flush=True)
                results.append((i, x, y, fov_dir, None))
                aborted = i
                break
            except AutofrapInterruptedException:
                # user stop: not a failure, the FOV's state is clean (its
                # finally-cleanup already ran); just stop the grid — the
                # unvisited positions (incl. this one) are simply not
                # in results
                stopped = True
                break
            except RecoverableError as e:
                print(f'!!! FOV {i} failed: {e} - moving on to the next '
                      f'position', flush=True)
                fov_results = None

            results.append((i, x, y, fov_dir, fov_results))

    # TODO: this is the only time we make use of the results list
    # for printing (x of N positions done) we could just use a counter here in the outer loop
    # printing n_cells here is not really necessary, we could just print some stats in inner loop
    # Thus, remove the results passing from this, inner_loop and wrapper?
    # May cause problems for FakeNIS dry runs if that relies on results, but for production use it's not necessary. 

    n_ok = sum(1 for r in results if r[4] is not None)
    n_cells = sum(len(r[4]) for r in results if r[4] is not None)
    if stopped:
        n_not = len(positions) - len(results)
        print(f'\nGrid stopped by user: {n_ok}/{len(positions)} FOV(s) done, '
              f'{n_cells} cell(s) stimulated, {n_not} FOV(s) not visited, '
              f'output in {run_dir}')
    elif aborted is not None:
        n_not = len(positions) - aborted + 1
        print(f'\nGrid ABORTED at FOV {aborted}: {n_ok}/{len(results)} visited '
              f'FOV(s) ok, {n_cells} cell(s) stimulated, {n_not} FOV(s) not '
              f'visited, output in {run_dir}')
    else:
        print(f'\nGrid done: {n_ok}/{len(positions)} FOV(s), {n_cells} cell(s) '
              f'stimulated, output in {run_dir}')
    return results


def _default_out_dir():
    """repo root (this file lives one level down in autofrap/) +
    test_acquisitions/autofrap_grid"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, 'test_acquisitions', 'autofrap_grid')


def build_positions(start_xy, fov, nx=2, ny=2, spacing=1.0,
                    spiral=False, max_positions=None):
    """Generate stage positions for grid or centre-out spiral.

    Parameters
    ----------
    start_xy : tuple
        Centre stage position (x, y) in µm.
    fov : tuple
        Field of view (fov_x, fov_y) in µm.
    nx, ny : int
        Grid dimensions.
    spacing : float
        Spacing in FOV units.
    spiral : bool
        If True, generate a centre-out square spiral.
    max_positions : int or None
        Hard cap on number of positions. For spiral mode, if None it
        defaults to nx*ny.

    Returns
    -------
    positions : list of (x, y)
    """
    if spiral:
        from autofrap.core.utils.grid import spiral_positions
        max_pos = max_positions if max_positions is not None else nx * ny
        # spiral_positions expects a scalar FOV; use mean of x/y for rectangular FOVs
        fov_scalar = float(fov[0]) if isinstance(fov, (list, tuple)) else float(fov)
        if isinstance(fov, (list, tuple)) and len(fov) > 1:
            fov_scalar = (float(fov[0]) + float(fov[1])) / 2.0
        positions = spiral_positions(start_xy, fov=fov_scalar, spacing=spacing,
                                     max_positions=max_pos)
    else:
        positions = grid_positions(start_xy, fov=fov, nx=nx, ny=ny, spacing=spacing)

    if max_positions is not None:
        if max_positions < 1:
            raise ValueError('--max-positions must be >= 1')
        positions = positions[:max_positions]
    return positions


def parse_cli_args(argv=None):
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default_detector = os.path.join(
        _repo_root, 'autofrap', 'detectors', 'cellpose_remote_detector.py')

    p = argparse.ArgumentParser(
        description='auto-FRAP over a grid of stage positions '
                    '(see autofrap())')
    p.add_argument('--out', '-o', default=_default_out_dir(),
                   help='output directory (a <run_stamp>/ sub-directory is '
                        'created in it) [default: %(default)s]')
    p.add_argument('--nis', default=r'C:\Program Files\NIS-Elements\nis_ar.exe',
                   help='path to nis_ar.exe [default: %(default)s]')
    p.add_argument('--nx', type=int, default=2,
                   help='grid size in x (1 = single FOV) [default: %(default)s]')
    p.add_argument('--ny', type=int, default=2,
                   help='grid size in y (1 = single FOV) [default: %(default)s]')
    p.add_argument('--spacing', type=float, default=1.0,
                   help='grid spacing in units of FOV (1 = touching) '
                        '[default: %(default)s]')
    p.add_argument('--max-cycles', type=int, default=1,
                   help='max FRAP cycles per FOV [default: %(default)s]')
    p.add_argument('--frap-oc', default='FRAPPA',
                   help='optical configuration name to use for FRAP stimulation [default: %(default)s]')
    p.add_argument('--until-done', action='store_true',
                   help='run until all cells of a FOV are stimulated '
                        '(ignore --max-cycles)')
    p.add_argument('--no-return', action='store_true',
                   help="don't move back to the start position after the run")
    p.add_argument('--spiral', action='store_true',
                   help='use a centre-out square spiral instead of a plain NxM grid; '
                        '--max-positions sets the number of positions, otherwise nx*ny is used')
    p.add_argument('--max-positions', '--num-positions', type=int, default=None,
                   help='maximum number of positions to visit, applied as a hard cap to both '
                        'grid and spiral visit orders. For spiral mode, if omitted it defaults '
                        'to --nx * --ny; for grid mode it truncates the generated NxM grid '
                        '[default: None]')
    p.add_argument('--detector', default=_default_detector,
                   help='path to a .py file defining detection_fun '
                        '[default: %(default)s]')
    p.add_argument('--detector-arg', action='append', default=[],
                   metavar='KEY=VALUE',
                   help='extra parameter to pass to the detector, '
                        'e.g. --detector-arg diameter=30 '
                        '(repeatable)')
    p.add_argument('--name',
                   help='experiment name: the run directory is named '
                        '<timestamp>_<name> (or <name> with --no-timestamp) '
                        '[default: <timestamp>]')
    p.add_argument('--no-timestamp', action='store_true',
                   help='name the run directory exactly --name (requires '
                        '--name)')
    args = p.parse_args(argv)
    if args.no_timestamp and not args.name:
        p.error('--no-timestamp requires --name')
    return args


if __name__ == '__main__':

    # Ctrl-C handling: first press requests a clean stop at the next safe
    # boundary (end of cycle / between FOVs - the current macro call
    # runs to completion, we never kill it); a second press raises
    # KeyboardInterrupt immediately (the finally-cleanup still runs)
    _stop = {'requested': False, 'count': 0}

    def _on_sigint(signum, frame):
        _stop['count'] += 1
        if _stop['count'] == 1:
            _stop['requested'] = True
            print('\nCtrl-C: stopping after the current cycle '
                  '(press again to interrupt immediately)', flush=True)
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)


    args = parse_cli_args()


    print(f'loading detector from: {args.detector}', flush=True)
    detection_fun = load_detector_file(args.detector)

    detector_kwargs = {}
    for arg in args.detector_arg:
        if '=' not in arg:
            print(f'ERROR: --detector-arg expects KEY=VALUE, got: {arg!r}')
            sys.exit(1)
        key, val = arg.split('=', 1)
        try:
            val = float(val) if '.' in val else int(val)
        except ValueError:
            pass
        detector_kwargs[key] = val

    try:
        autofrap(
            args.nis, args.out,
            nx=args.nx, ny=args.ny, spacing=args.spacing,
            spiral=args.spiral, max_positions=args.max_positions,
            max_cycles=None if args.until_done else args.max_cycles,
            detection_fun=detection_fun,
            frap_oc=args.frap_oc,
            name=args.name, use_timestamp=not args.no_timestamp,
            stop_check=lambda: _stop['requested'],
            return_to_start=not args.no_return,
            **detector_kwargs
        )
    except AutofrapInterruptedException:
        sys.exit(130)
    except NonRecoverableError as e:
        print(f'\nERROR: {e}')
        sys.exit(1)
