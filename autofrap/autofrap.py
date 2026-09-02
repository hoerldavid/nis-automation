"""
Auto-FRAP loop: survey -> detect -> pick unused cell -> stimulate -> repeat.

All acquisition settings come from the NIS GUI (the ND acquisition
definition carries the survey's optical configuration); this script
just runs them in a loop.

Per cycle:
  1. run the current ND experiment, saved to <stamp>_c<NN>_survey.nd2
  2. detect objects in the survey image (detection_fun — by default
     cellpose on the GPU server via cellpose_server.py) — returns a
     (labels, stimulation_mask) tuple
  3. remap object labels against the previous cycle (IoU matching)
     and pick the smallest not-yet-stimulated label that has at least
     one pixel in the stimulation mask
  4. open the survey image in NIS, add two ROIs: the whole cell
     (for downstream analysis) and the stimulation region
     ((labels == cell_id) & stimulation_mask), the latter set to
     stimulation mode (type 3)
  5. switch optical conf to FRAPPA, run the current sequential
     stimulation experiment
  6. save the FRAP timeseries to <stamp>_c<NN>_frap.nd2 (the stimulation
     ROI is part of the saved file)
  7. delete both ROIs (so they don't linger for the next cycle) and
     close the FRAP + survey documents
-> next cycle (the ND experiment definition restores the survey OC)

The loop stops when every detected object has been stimulated, when
no cell has stimulation-eligible pixels, or when max_cycles is reached.
"""
import os
import sys
import time
from functools import partial

import numpy as np
from calmutils.segmentation import merge_label_slices

# repo root (for nis_util) — this script lives one level down in autofrap/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detection
import nis_util

# IoU threshold for matching object labels between consecutive
# survey images (see merge_label_slices)
IOU_THRESHOLD = 0.3

# default detector: cellpose (cpdino-vitb) on the GPU server
# (cellpose_server.py), DAPI channel
CELLPOSE_SERVER_URL = 'http://10.163.69.12:8000'
SURVEY_CHANNEL = 0


def next_cell(labels, stimulated):
    """
    smallest label > 0 not in `stimulated`; None if none left

    Kept for backward compatibility; use ``next_stimulatable_cell``
    when a stimulation mask is available.
    """
    for lbl in sorted(np.unique(labels).tolist()):
        if lbl > 0 and lbl not in stimulated:
            return lbl
    return None


def next_stimulatable_cell(labels, stimulation_mask, stimulated, skip=None):
    """
    Find the next unstimulated cell that has at least one nonzero
    pixel in the stimulation mask.

    Iterates over labels in sorted order. For each candidate label
    that is not in the stimulated set, checks whether
    ``(labels == lbl) & stimulation_mask`` has any nonzero pixels;
    if not, skips to the next candidate.

    Parameters
    ----------
    labels: 2D np.ndarray
        label map (0 = background, 1..N = objects)
    stimulation_mask: 2D np.ndarray
        binary mask of areas eligible for photostimulation
    stimulated: set of int
        already-stimulated cell IDs
    skip: int or None
        if provided, skip labels <= skip (useful for retrying
        after a failed stimulation)

    Returns
    -------
    cell_id: int or None
        the next stimulatable cell, or None if none found
    """
    for lbl in sorted(np.unique(labels).tolist()):
        if lbl > 0 and lbl not in stimulated:
            if skip is not None and lbl <= skip:
                continue
            if np.any((labels == lbl) & stimulation_mask):
                return lbl
    return None


def autofrap(nis_exe, out_dir, max_cycles=None, detection_fun=None,
             frap_oc='FRAPPA', iou_threshold=IOU_THRESHOLD):
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
        survey_file -> (labels, stimulation_mask); defaults to
        detection.detect with the cellpose server (CELLPOSE_SERVER_URL)
        on SURVEY_CHANNEL, e.g. partial(detection.detect,
        detector='cellpose-remote', server_url=..., channel=...);
        pass partial(detection.detect, detector='dummy') for testing
        without the server
    frap_oc: str
        optical configuration to activate before each stimulation
    iou_threshold: float
        IoU threshold for matching labels between cycles

    Returns
    -------
    results: list of (cycle, cell, survey_file, frap_file)
    """
    if detection_fun is None:
        detection_fun = partial(
            detection.detect,
            detector='cellpose-remote',
            server_url=CELLPOSE_SERVER_URL,
            channel=SURVEY_CHANNEL,
        )

    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    stimulated = set()
    prev_labels = None
    results = []

    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        survey_file = os.path.join(out_dir, f'{stamp}_c{cycle:02d}_survey.nd2')
        frap_file = os.path.join(out_dir, f'{stamp}_c{cycle:02d}_frap.nd2')

        # 1+2. survey: run the GUI-configured ND experiment, saved
        t0 = time.time()
        nis_util.run_current_nd_experiment(nis_exe, outfile=survey_file,
                                           progress_bar=True)
        print(f'[c{cycle:02d}] survey saved ({time.time() - t0:.1f} s)', flush=True)

        # 3. detect
        labels, stimulation_mask = detection_fun(survey_file)
        n_obj = len(np.unique(labels)) - 1

        # 4. pick the next unused cell with stimulation-eligible pixels
        if prev_labels is None:
            cur_labels = labels
        else:
            # relabel the new detection into the previous cycle's numbering
            # (merge_label_slices adjusts the *new* labels to the old ones; new
            # objects get fresh IDs above the previous max) so the `stimulated`
            # set, expressed in cycle-1 numbering, stays valid unchanged
            # caveat: if a cell *vanishes* between cycles, its id leaves a gap
            # and merge_label_slices' re-baselining (relabel_sequential on the
            # previous map) shifts the ids of everything above the gap, so
            # `stimulated` can point at the wrong cells (double FRAP). Benign
            # for the intended <=2 cycles/FOV; for longer runs, exclude FRAPed
            # cells by centroid instead of label id (see STATUS.md, TODO #13)
            _, cur_labels = merge_label_slices(
                [prev_labels, labels], iou_threshold=iou_threshold
            )

        cell = next_stimulatable_cell(
            cur_labels, stimulation_mask, stimulated
        )

        if cell is None:
            print(
                f'[c{cycle:02d}] {n_obj} objects, all stimulated or no '
                'stimulation mask -> stop'
            )
            break

        print(f'[c{cycle:02d}] {n_obj} objects, stimulating cell {cell}')

        # 5. ROIs + stimulation run: whole cell (saved for downstream
        #    analysis) + stimulation region, the latter set to
        #    stimulation mode
        cell_poly = detection.label_to_polygon(cur_labels, cell)
        stim_poly = detection.detect_polygon_stim_mask(
            cur_labels, stimulation_mask, cell
        )
        if not cell_poly or not stim_poly:
            raise RuntimeError(f'no polygon for cell {cell}')

        nis_util.open_image(nis_exe, survey_file)
        cell_roi = nis_util.add_polygon_roi(nis_exe, cell_poly)
        if cell_roi <= 0:
            raise RuntimeError(f'cell ROI creation failed (id={cell_roi})')
        stim_roi = nis_util.add_polygon_roi(nis_exe, stim_poly)
        if stim_roi <= 0:
            raise RuntimeError(f'stim ROI creation failed (id={stim_roi})')
        nis_util.set_roi_type(nis_exe, stim_roi, 3)  # 3 = stimulation

        nis_util.set_optical_configuration(nis_exe, frap_oc)
        t0 = time.time()
        nis_util.run_stimulation_experiment(nis_exe)
        print(f'[c{cycle:02d}] stimulation done ({time.time() - t0:.1f} s)', flush=True)

        # 6. save the FRAP timeseries (includes the stimulation ROI)
        nis_util.save_current_document(nis_exe, frap_file)

        # 7. close FRAP document (current), then delete both ROIs on the
        #    survey document (after saving, so they stay in the saved FRAP
        #    file but don't linger for the next cycle) and close it
        nis_util.close_current_document(nis_exe, save='discard')
        doc = nis_util.get_current_document(nis_exe)
        if os.path.normcase(doc) != os.path.normcase(survey_file):
            nis_util.open_image(nis_exe, survey_file)
        nis_util.delete_roi(nis_exe, stim_roi)
        nis_util.delete_roi(nis_exe, cell_roi)
        nis_util.close_current_document(nis_exe, save='discard')

        results.append((cycle, cell, survey_file, frap_file))
        prev_labels = cur_labels
        stimulated.add(cell)

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
                  iou_threshold=IOU_THRESHOLD):
    """
    run the autofrap() loop on every position of a stage grid, centered
    on the current stage position

    Each FOV gets its own sub-directory, so the per-cycle filenames of
    autofrap() keep working unchanged:

        <out_dir>/<run_stamp>/fov<i>/

    (the exact stage position of each FOV is in the nd2 metadata of
    the saved files)
            <stamp>_cNN_survey.nd2
            <stamp>_cNN_frap.nd2

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
    max_cycles, detection_fun, frap_oc, iou_threshold:
        passed through to autofrap() unchanged

    Returns
    -------
    results: list of (i, x, y, fov_dir, fov_results)
        fov_results is autofrap's per-cycle results, or None if that FOV
        failed (the run continues with the next position)
    """
    os.makedirs(out_dir, exist_ok=True)
    start_xy = nis_util.get_position(nis_exe)[:2]
    if positions is None:
        fov = nis_util.get_fov_from_res(nis_util.get_resolution(nis_exe))
        positions = grid_positions(start_xy, fov, nx=nx, ny=ny, spacing=spacing)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(out_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f'grid: {len(positions)} position(s), '
          f'start=({start_xy[0]:+.2f}, {start_xy[1]:+.2f}) um')
    results = []
    for i, (x, y) in enumerate(positions, 1):
        fov_dir = os.path.join(run_dir, f'fov{i:02d}')
        print(f'\n=== [{i}/{len(positions)}] ({x:+.1f}, {y:+.1f}) um -> {fov_dir}',
              flush=True)

        nis_util.set_position(nis_exe, pos_xy=(x, y))
        time.sleep(settle_s)

        try:
            fov_results = autofrap(nis_exe, fov_dir, max_cycles=max_cycles,
                                   detection_fun=detection_fun, frap_oc=frap_oc,
                                   iou_threshold=iou_threshold)
        except Exception as e:
            print(f'!!! FOV {i} failed: {e!r} - moving on to the next position',
                  flush=True)
            fov_results = None

        results.append((i, x, y, fov_dir, fov_results))

    if return_to_start:
        nis_util.set_position(nis_exe, pos_xy=start_xy)
        time.sleep(settle_s)
        print(f'moved back to start ({start_xy[0]:+.2f}, {start_xy[1]:+.2f})')

    n_ok = sum(1 for r in results if r[4] is not None)
    n_cells = sum(len(r[4]) for r in results if r[4] is not None)
    print(f'\nGrid done: {n_ok}/{len(positions)} FOV(s), {n_cells} cell(s) '
          f'stimulated, output in {run_dir}')
    return results


if __name__ == '__main__':
    NIS_EXE = r'C:\Program Files\NIS-Elements\nis_ar.exe'
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autofrap_out')
    autofrap(NIS_EXE, OUT_DIR, max_cycles=2)
