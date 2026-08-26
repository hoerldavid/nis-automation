"""
Auto-FRAP loop: survey -> detect -> pick unused cell -> stimulate -> repeat.

All acquisition settings come from the NIS GUI (the ND acquisition
definition carries the survey's optical configuration); this script
just runs them in a loop.

Per cycle:
  1. run the current ND experiment, saved to <stamp>_c<NN>_survey.nd2
  2. detect objects in the survey image (detection.detect) — returns
     a (labels, stimulation_mask) tuple
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

import numpy as np
from calmutils.segmentation import merge_label_slices

# repo root (for nis_util) — this script lives one level down in autofrap/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detection
import nis_util

# IoU threshold for matching object labels between consecutive
# survey images (see merge_label_slices)
IOU_THRESHOLD = 0.3


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


def remap_stimulated(prev_labels, labels, stimulated, iou_threshold):
    """
    re-express `stimulated` labels (numbered in `prev_labels`) in the
    numbering of the new `labels` map

    Returns
    -------
    (labels, stimulated)
        labels: the new label map, unchanged
        stimulated: remapped set (labels that vanished are dropped)
    """
    merged = merge_label_slices([prev_labels, labels], iou_threshold=iou_threshold)
    prev_remapped, cur_remapped = merged[0], merged[1]

    stim = set()
    for old in stimulated:
        mask = prev_remapped == old
        if np.any(mask):
            stim.add(int(cur_remapped[mask].max()))
    return cur_remapped, stim


def autofrap(nis_exe, out_dir, max_cycles=None, survey_channel=0,
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
    survey_channel: int
        channel of the survey image to detect on
    frap_oc: str
        optical configuration to activate before each stimulation
    iou_threshold: float
        IoU threshold for matching labels between cycles

    Returns
    -------
    results: list of (cycle, cell, survey_file, frap_file)
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    stimulated = set()
    prev_labels = None
    results = []

    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        survey_file = os.path.join(out_dir, '%s_c%02d_survey.nd2' % (stamp, cycle))
        frap_file = os.path.join(out_dir, '%s_c%02d_frap.nd2' % (stamp, cycle))

        # 1+2. survey: run the GUI-configured ND experiment, saved
        t0 = time.time()
        nis_util.run_current_nd_experiment(nis_exe, outfile=survey_file,
                                           progress_bar=True)
        print('[c%02d] survey saved (%.1f s)' % (cycle, time.time() - t0), flush=True)

        # 3. detect
        labels, stimulation_mask = detection.detect(
            survey_file, channel=survey_channel
        )
        n_obj = len(np.unique(labels)) - 1

        # 4. pick the next unused cell with stimulation-eligible pixels
        if prev_labels is None:
            cur_labels = labels
        else:
            cur_labels, stimulated = remap_stimulated(
                prev_labels, labels, stimulated, iou_threshold
            )

        cell = next_stimulatable_cell(
            cur_labels, stimulation_mask, stimulated
        )

        if cell is None:
            print(
                '[c%02d] %d objects, all stimulated or no '
                'stimulation mask -> stop' % (cycle, n_obj)
            )
            break

        print('[c%02d] %d objects, stimulating cell %d' % (cycle, n_obj, cell))

        # 5. ROIs + stimulation run: whole cell (saved for downstream
        #    analysis) + stimulation region, the latter set to
        #    stimulation mode
        cell_poly = detection.label_to_polygon(cur_labels, cell)
        stim_poly = detection.detect_polygon_stim_mask(
            cur_labels, stimulation_mask, cell
        )
        if not cell_poly or not stim_poly:
            raise RuntimeError('no polygon for cell %d' % cell)

        nis_util.open_image(nis_exe, survey_file)
        cell_roi = nis_util.add_polygon_roi(nis_exe, cell_poly)
        if cell_roi <= 0:
            raise RuntimeError('cell ROI creation failed (id=%d)' % cell_roi)
        stim_roi = nis_util.add_polygon_roi(nis_exe, stim_poly)
        if stim_roi <= 0:
            raise RuntimeError('stim ROI creation failed (id=%d)' % stim_roi)
        nis_util.set_roi_type(nis_exe, stim_roi, 3)  # 3 = stimulation

        nis_util.set_optical_configuration(nis_exe, frap_oc)
        t0 = time.time()
        nis_util.run_stimulation_experiment(nis_exe)
        print('[c%02d] stimulation done (%.1f s)' % (cycle, time.time() - t0), flush=True)

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

    print('\nDone: %d cell(s) stimulated in %d cycle(s), output in %s'
          % (len(results), cycle, out_dir))
    return results


if __name__ == '__main__':
    NIS_EXE = r'C:\Program Files\NIS-Elements\nis_ar.exe'
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autofrap_out')
    autofrap(NIS_EXE, OUT_DIR, max_cycles=2)
