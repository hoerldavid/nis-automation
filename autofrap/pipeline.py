"""
Auto-FRAP loop: survey -> detect -> pick unused cell -> stimulate -> repeat.

All acquisition settings come from the NIS GUI (the ND acquisition
definition carries the survey's optical configuration); this script
just runs them in a loop.

Per cycle:
  1. run the current ND experiment, saved to
     <file_prefix>_cycle<NN>_survey.nd2 (file_prefix defaults to a
     timestamp for standalone runs; autofrap_grid passes 'fov<NN>')
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
no cell has stimulation-eligible pixels, or when max_cycles is reached.

Error handling: every failure is translated into one of two exception
classes - RecoverableError (this FOV is lost, a grid run may continue)
or NonRecoverableError (the microscope/detection state is unknown or
broken, a grid run should abort). A failed cycle best-effort deletes
its own ROIs and closes its documents before re-raising, so a grid run
that continues starts the next FOV from a clean GUI state.
"""
import os
import sys
import time
from functools import partial

import numpy as np

# Fallback: when run directly as a script (not through the package),
# the repo root isn't on sys.path. __init__.py handles this when
# imported as 'import autofrap', but direct execution needs it too.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _here not in sys.path:
    sys.path.insert(0, _here)

import nis_util  # root-level module
from autofrap import detection, mask_utils, nd2_helpers, qc
from skimage.measure import regionprops

# default detector: cellpose (cpdino-vitb) on the GPU server
# (cellpose_server.py), DAPI channel
CELLPOSE_SERVER_URL = 'http://10.163.69.12:8000'
SURVEY_CHANNEL = 0

def default_detector(detector_fun=None):
    """
    compose the default detection pipeline

    The default detector loads ``SURVEY_CHANNEL``, runs the given
    ``detector_fun`` (cellpose on the server by default, ``dummy`` for
    testing), applies the left-half stimulation mask, and returns the
    loaded image as visualization.

    Parameters
    ----------
    detector_fun: callable, optional
        image -> labels; defaults to ``remote_detect_objects`` on
        ``CELLPOSE_SERVER_URL``.

    Returns
    -------
    detection_fun: callable
        survey_file -> (labels[, stimulation_mask[, visualization]]);
        suitable for passing as ``detection_fun`` to :func:`autofrap`.

    Examples
    --------
    Default (cellpose on GPU server):

        detection_fun = default_detector()
        results = autofrap(nis, out_dir, detection_fun=detection_fun)

    Testing without the server:

        detection_fun = default_detector(detection.dummy_detect_objects)
        results = autofrap(nis, out_dir, detection_fun=detection_fun)

    Custom detector (e.g. with a local model):

        detection_fun = default_detector(my_local_detector)
    """
    if detector_fun is None:
        detector_fun = partial(detection.remote_detect_objects,
                               server_url=CELLPOSE_SERVER_URL)
    return detection.build_detector(
        partial(nd2_helpers.read_channel, channel=SURVEY_CHANNEL),
        detector_fun,
        stim_mask_fun=_half_object_stim_mask,
        visualization_fun=lambda image: image,
    )


# cycle-number tag in output file names (<prefix>_cycle01_survey.nd2);
# spelled out rather than 'c' to avoid the color-channel reading
CYCLE_PREFIX = 'cycle'


# ND Acquisition tab names that are *not* valid for a survey image
# (multi-position, time-lapse, or multi-position large-image scans
# are acquisition pipelines, not single-image surveys)
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


def _run_nd_acq_check(nis_exe):
    """Pre-flight wrapper: queries NIS then validates the template."""
    tabs = nis_util.get_nd_acq_tabs(nis_exe)
    return _check_nd_acq_template(tabs)


def _half_object_stim_mask(labels, image):
    """stim_mask_fun adapter: half_object_stim_mask only needs the
    labels (the build_detector() contract passes the image as well)"""
    return mask_utils.half_object_stim_mask(labels)


class AutofrapError(Exception):
    """base class for auto-FRAP pipeline errors"""


class RecoverableError(AutofrapError):
    """failure confined to the current FOV (no polygon for the cell, ROI
    creation failed); a grid run can continue with the next position"""


class NonRecoverableError(AutofrapError):
    """failure that makes further FOVs pointless or unsafe (NIS state
    unknown, detection failed - the detector/server state is suspect,
    disk full); a grid run aborts"""


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


def autofrap(nis_exe, out_dir, max_cycles=None, detection_fun=None,
             frap_oc='FRAPPA', centroid_threshold='auto',
             file_prefix=None):
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
    detection_fun: callable, optional
        survey_file -> (labels[, stimulation_mask[, visualization]])
        or a bare label map; only the label map is required.
        stimulation_mask (FRAP sub-regions): None or absent -> the
        whole cell is FRAPed.
        visualization (2D or RGB(A), detector-assembled, e.g.
        multi-channel): used for the QC overlay only; absent -> the
        overlay is drawn on a blank canvas (autofrap() does not know
        which channel(s) the detector used). Default: use
        :func:`default_detector` (cellpose on the server,
        ``SURVEY_CHANNEL``, left-half stimulation mask). For testing
        without the server: ``default_detector(dummy_detect_objects)``.
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
        (YYYYmmdd_HHMMSS) for standalone runs — autofrap_grid passes
        'fov<NN>' per position. Set to '' for plain cycle<NN>_... names.

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
    """

    if detection_fun is None:
        detection_fun = default_detector()

    os.makedirs(out_dir, exist_ok=True)
    if file_prefix is None:
        file_prefix = time.strftime('%Y%m%d_%H%M%S')
    imaged_centroids = []  # list of (y, x) tuples — centroids of stimulated cells
    results = []

    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        survey_file = os.path.join(
            out_dir, f'{file_prefix}_{CYCLE_PREFIX}{cycle:02d}_survey.nd2')
        frap_file = os.path.join(
            out_dir, f'{file_prefix}_{CYCLE_PREFIX}{cycle:02d}_frap.nd2')
        cell_roi = stim_roi = None

        try:
            # 1+2. survey: run the GUI-configured ND experiment, saved
            t0 = time.time()
            nis_util.run_current_nd_experiment(nis_exe, outfile=survey_file,
                                               progress_bar=True)
            print(f'[c{cycle:02d}] survey saved ({time.time() - t0:.1f} s)',
                  flush=True)
            # NIS can fail to save without saying so (disk full, GUI dialog,
            # crash) - check instead of trusting
            if not os.path.isfile(survey_file):
                raise NonRecoverableError(
                    f'survey file missing after the ND run: {survey_file} '
                    '(NIS did not save it - check the GUI / disk)')

            # 3. detect (the client already retried once; any failure
            # that survives is run-level: the detector/server state is
            # suspect, so a grid run aborts - no per-exception
            # translation here, the detector may be any callable)
            try:
                det = detection_fun(survey_file)
            except Exception as e:
                raise NonRecoverableError(
                    f'detection failed on {survey_file}: {e!r}') from e

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
            n_obj = len(np.unique(labels)) - 1

            # 4. match detected objects to the "already-imaged" map
            # via centroid distance; pick the smallest unmatched label
            # that has at least one pixel in the stimulation mask
            if imaged_centroids:
                # build a set of current-cycle labels that match an
                # already-imaged cell (centroid-based, id-independent).
                # In 'auto' mode the threshold is the cell's own
                # equivalent_diameter; a numeric value is used as-is.
                matched = set()
                for rp in regionprops(labels):
                    cy, cx = rp.centroid  # (y, x) — numpy order
                    # determine per-cell matching radius
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
                break

            # Inner loop: if the picked cell has no polygon, skip it and
            # try the next one in the same survey (defensive; nearly
            # unreachable with current mask logic).
            skipped = set()
            fovd_done = False
            while True:
                cell_poly = mask_utils.mask_to_polygon(
                    detection.cell_mask(labels, cell)
                )
                stim_poly = mask_utils.mask_to_polygon(
                    detection.cell_mask(labels, cell, stimulation_mask)
                )

                if cell_poly and stim_poly:
                    break  # found a viable cell

                # This cell can't be used — skip it and try the next.
                skipped.add(cell)
                print(
                    f'[c{cycle:02d}] cell {cell}: no polygon, skipping'
                )
                cell = next_stimulatable_cell(
                    labels, matched | skipped, stimulation_mask
                )
                if cell is None:
                    print(
                        f'[c{cycle:02d}] all {n_obj} objects have no '
                        'polygon -> move to next FOV'
                    )
                    fovd_done = True
                    break

            if fovd_done:
                break  # exit the cycle loop, go to next FOV

            print(f'[c{cycle:02d}] {n_obj} objects, stimulating cell {cell}')

            # 5. ROIs + stimulation run: whole cell (saved for downstream
            #    analysis) + stimulation region, the latter set to
            #    stimulation mode

            # QC artifact before the stimulation run, so it is on disk
            # even if the NIS part of the cycle fails; a rendering
            # problem must not abort the run
            try:
                qc.save_qc_overlay(
                    viz_image, labels,
                    os.path.join(out_dir, f'{file_prefix}_'
                                 f'{CYCLE_PREFIX}{cycle:02d}_survey_qc.png'),
                    stimulation_mask=stimulation_mask, cell_id=cell,
                    cell_poly=cell_poly, stim_poly=stim_poly,
                    caption=f'{CYCLE_PREFIX}{cycle:02d} cell {cell}')
            except Exception as e:
                print(f'[c{cycle:02d}] WARNING: QC overlay failed: {e!r}',
                      flush=True)

            # TODO: image should still be open after acquisition, no need to re/open?
            # check if this gives any timing advantage or is just a no-op?
            nis_util.open_image(nis_exe, survey_file)
            
            doc = nis_util.get_current_document(nis_exe)
            if os.path.normcase(doc) != os.path.normcase(survey_file):
                raise NonRecoverableError(
                    f'could not open {survey_file} '
                    f'(current document: {doc})')
            cell_roi = nis_util.add_polygon_roi(nis_exe, cell_poly)
            if cell_roi <= 0:
                raise RecoverableError(
                    f'cell ROI creation failed (id={cell_roi})')
            stim_roi = nis_util.add_polygon_roi(nis_exe, stim_poly)
            if stim_roi <= 0:
                raise RecoverableError(
                    f'stim ROI creation failed (id={stim_roi})')
            nis_util.set_roi_type(nis_exe, stim_roi, 3)  # 3 = stimulation

            nis_util.set_optical_configuration(nis_exe, frap_oc)
            t0 = time.time()
            nis_util.run_stimulation_experiment(nis_exe)
            print(f'[c{cycle:02d}] stimulation done ({time.time() - t0:.1f} s)',
                  flush=True)

            # 6. save the FRAP timeseries (includes the stimulation ROI)
            nis_util.save_current_document(nis_exe, frap_file)
            # ImageSaveAs can silently write nothing (frozen live view, disk
            # full) - check instead of trusting
            if not os.path.isfile(frap_file):
                raise NonRecoverableError(
                    f'FRAP file missing after save_current_document: '
                    f'{frap_file} (ImageSaveAs wrote nothing)')

            # 7. close FRAP document (current), then delete both ROIs on the
            #    survey document (after saving, so they stay in the saved
            #    FRAP file but don't linger for the next cycle) and close it
            nis_util.close_current_document(nis_exe, save='discard')
            doc = nis_util.get_current_document(nis_exe)
            
            # TODO: don't re-open just to close it again
            if os.path.normcase(doc) != os.path.normcase(survey_file):
                nis_util.open_image(nis_exe, survey_file)
            nis_util.delete_roi(nis_exe, stim_roi)
            nis_util.delete_roi(nis_exe, cell_roi)
            nis_util.close_current_document(nis_exe, save='discard')
            cell_roi = stim_roi = None

            results.append((cycle, cell, survey_file, frap_file))
            # add the stimulated cell's centroid to the already-imaged map
            for rp in regionprops(labels):
                if rp.label == cell:
                    imaged_centroids.append(rp.centroid)  # (y, x)
                    break
        except (RecoverableError, NonRecoverableError):
            raise
        except KeyError as e:
            # an empty ini read-back means the NIS macro aborted partway -
            # the GUI state is now unknown, so don't queue more FOVs on top
            raise NonRecoverableError(
                f'NIS macro failed (no read-back: {e!r}) - '
                'the NIS state is now unknown') from e
        except OSError as e:
            raise NonRecoverableError(f'OS error: {e!r}') from e
        finally:
            if cell_roi is not None or stim_roi is not None:
                # failed mid-cycle: delete this cycle's ROIs and close its
                # documents again, best effort (NIS may itself be the
                # problem, in which case just give up quietly)
                try:
                    doc = nis_util.get_current_document(nis_exe)
                    if os.path.normcase(doc) != os.path.normcase(survey_file):
                        nis_util.close_current_document(nis_exe, save='discard')
                        nis_util.open_image(nis_exe, survey_file)
                    if stim_roi is not None:
                        nis_util.delete_roi(nis_exe, stim_roi)
                    if cell_roi is not None:
                        nis_util.delete_roi(nis_exe, cell_roi)
                    nis_util.close_current_document(nis_exe, save='discard')
                except Exception:
                    pass

    print(f'\nDone: {len(results)} cell(s) stimulated in {cycle} cycle(s), output in {out_dir}')
    return results


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


def autofrap_grid(nis_exe, out_dir, nx=2, ny=2, spacing=1.0, positions=None,
                  settle_s=2.0, return_to_start=True, max_cycles=None,
                  detection_fun=None, frap_oc='FRAPPA',
                  centroid_threshold='auto',
                  fov_subdirs=False):
    """
    run the autofrap() loop on every position of a stage grid, centered
    on the current stage position

    By default all FOVs are written to a single run directory; the
    'fov<NN>' file prefix (matching the log lines) keeps files
    self-describing and makes one folder easy to browse (QC PNGs side
    by side) or to hand to downstream analysis:

        <out_dir>/<run_stamp>/
            <fovNN>_cycleNN_survey.nd2
            <fovNN>_cycleNN_frap.nd2
            <fovNN>_cycleNN_survey_qc.png

    With fov_subdirs=True, each FOV goes into its own sub-directory
    instead (<run_stamp>/fov<NN>/, same file names).

    (the exact stage position of each FOV is in the nd2 metadata of
    the saved files — nd2_helpers.stage_position reads it back)

    Parameters
    ----------
    nis_exe, out_dir: str
        as in autofrap(); a <run_stamp> sub-directory is created in
        out_dir for this grid run
    nx, ny: int
        number of grid positions in x and y
    spacing: float
        neighbor distance in units of FOV (1 = touching, <1 = overlap)
    positions: list of (x, y), optional
        precomputed stage positions in visit order; if given, nx/ny/spacing
        are ignored. Any visit order works (grid is row-major; a
        center-out spiral order could be passed in later)
    settle_s: float
        settling time [s] after each stage move
    return_to_start: bool
        move back to the starting position after the last FOV
    max_cycles, detection_fun, frap_oc, centroid_threshold:
        passed through to autofrap() unchanged
    fov_subdirs: bool
        give each FOV its own <run_stamp>/fov<NN>/ sub-directory
        (default: all FOVs in the single run directory, position
        encoded in the file names)

    Returns
    -------
    results: list of (i, x, y, fov_dir, fov_results)
        fov_results is autofrap's per-cycle results, or None if that FOV
        failed; fov_dir is the per-FOV sub-directory (fov_subdirs=True)
        or the shared run directory. A RecoverableError skips the FOV
        and continues; a NonRecoverableError aborts the run (the
        remaining positions are not visited and do not appear in
        results)

    Raises
    ------
    NonRecoverableError
        if the starting stage position cannot be read, the ND
        Acquisition template is misconfigured, or the detection
        server is unreachable (passed on from autofrap())
    """
    os.makedirs(out_dir, exist_ok=True)
    # Pre-flight check: verify the ND Acquisition template is sane
    # (single image survey, no loops)
    _run_nd_acq_check(nis_exe)
    try:
        start_xy = nis_util.get_position(nis_exe)[:2]
    except (KeyError, OSError) as e:
        raise NonRecoverableError(
            f'could not read the starting stage position: {e!r}') from e
    if positions is None:
        fov = nis_util.get_fov_from_res(nis_util.get_resolution(nis_exe))
        positions = grid_positions(start_xy, fov, nx=nx, ny=ny, spacing=spacing)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(out_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f'grid: {len(positions)} position(s), '
          f'start=({start_xy[0]:+.2f}, {start_xy[1]:+.2f}) um')
    results = []
    aborted = None
    try:
        for i, (x, y) in enumerate(positions, 1):
            fov_dir = (os.path.join(run_dir, f'fov{i:02d}')
                       if fov_subdirs else run_dir)
            print(f'\n=== [{i}/{len(positions)}] ({x:+.1f}, {y:+.1f}) um '
                  f'-> {fov_dir} (fov{i:02d})',
                  flush=True)

            try:
                nis_util.set_position(nis_exe, pos_xy=(x, y))
            except (KeyError, OSError) as e:
                print(f'!!! FOV {i}: stage move failed: {e!r} '
                      f'- aborting the grid run', flush=True)
                results.append((i, x, y, fov_dir, None))
                aborted = i
                break

            # TODO: set_position may actually block until target is reached
            # making this redundent - test on scope and remove if unnecessary?
            time.sleep(settle_s)

            try:
                fov_results = autofrap(nis_exe, fov_dir, max_cycles=max_cycles,
                                       detection_fun=detection_fun,
                                       frap_oc=frap_oc,
                                       centroid_threshold=centroid_threshold,
                                       file_prefix=f'fov{i:02d}')
            except NonRecoverableError as e:
                print(f'!!! FOV {i}: non-recoverable error: {e} '
                      f'- aborting the grid run', flush=True)
                results.append((i, x, y, fov_dir, None))
                aborted = i
                break
            except RecoverableError as e:
                print(f'!!! FOV {i} failed: {e} - moving on to the next '
                      f'position', flush=True)
                fov_results = None

            results.append((i, x, y, fov_dir, fov_results))
    finally:
        if return_to_start:
            # best effort: even after an abort or an unexpected error the
            # stage should end up back where the run started
            try:
                nis_util.set_position(nis_exe, pos_xy=start_xy)
                time.sleep(settle_s)
                print(f'moved back to start ({start_xy[0]:+.2f}, '
                      f'{start_xy[1]:+.2f})')
            except Exception as e:
                print(f'!!! could not return to start: {e!r}', flush=True)

    n_ok = sum(1 for r in results if r[4] is not None)
    n_cells = sum(len(r[4]) for r in results if r[4] is not None)
    if aborted is not None:
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


if __name__ == '__main__':
    # CLI for autofrap_grid; the arguments mirror its parameters 1:1 so
    # the same values can be used in a notebook call instead
    import argparse

    p = argparse.ArgumentParser(
        description='auto-FRAP over a grid of stage positions '
                    '(see autofrap_grid)')
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
    p.add_argument('--until-done', action='store_true',
                   help='run until all cells of a FOV are stimulated '
                        '(ignore --max-cycles)')
    p.add_argument('--settle', type=float, default=2.0,
                   help='stage settling time [s] after each move '
                        '[default: %(default)s]')
    p.add_argument('--no-return', action='store_true',
                   help="don't move back to the start position after the run")
    p.add_argument('--detector', choices=['cellpose-remote', 'dummy'],
                   default='cellpose-remote',
                   help='detection backend [default: %(default)s]')
    a = p.parse_args()

    if a.detector == 'cellpose-remote':
        detector_fun = partial(detection.remote_detect_objects,
                               server_url=CELLPOSE_SERVER_URL)
    else:
        detector_fun = detection.dummy_detect_objects
    detection_fun = default_detector(detector_fun)

    autofrap_grid(a.nis, a.out, nx=a.nx, ny=a.ny, spacing=a.spacing,
                  settle_s=a.settle, return_to_start=not a.no_return,
                  max_cycles=None if a.until_done else a.max_cycles,
                  detection_fun=detection_fun)
