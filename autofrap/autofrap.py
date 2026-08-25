"""
Auto-FRAP loop: survey -> detect -> pick unused cell -> stimulate -> repeat.

All acquisition settings come from the NIS GUI (the ND acquisition
definition carries the survey's optical configuration); this script
just runs them in a loop.

Per cycle:
  1. run the current ND experiment, saved to <stamp>_c<NN>_survey.nd2
  2. detect objects in the survey image (detection.detect)
  3. remap object labels against the previous cycle (IoU matching)
     and pick the smallest not-yet-stimulated label
  4. open the survey image in NIS, add the object's polygon as a
     stimulation ROI (type 3)
  5. switch optical conf to FRAPPA, run the current sequential
     stimulation experiment
  6. save the FRAP timeseries to <stamp>_c<NN>_frap.nd2 (the stimulation
     ROI is part of the saved file)
  7. delete the stimulation ROI (so it doesn't linger for the next cycle)
     and close the FRAP + survey documents
-> next cycle (the ND experiment definition restores the survey OC)

The loop stops when every detected object has been stimulated or
max_cycles is reached.
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
    """
    for lbl in sorted(np.unique(labels).tolist()):
        if lbl > 0 and lbl not in stimulated:
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
        labels = detection.detect(survey_file, channel=survey_channel)
        n_obj = len(np.unique(labels)) - 1

        # 4. pick the next unused cell
        if prev_labels is None:
            cur_labels = labels
            cell = next_cell(cur_labels, stimulated)
        else:
            cur_labels, stimulated = remap_stimulated(prev_labels, labels,
                                                      stimulated, iou_threshold)
            cell = next_cell(cur_labels, stimulated)

        if cell is None:
            print('[c%02d] %d objects, all stimulated -> stop' % (cycle, n_obj))
            break

        print('[c%02d] %d objects, stimulating cell %d' % (cycle, n_obj, cell))

        # 5. stimulation ROI + stimulation run
        poly = detection.label_to_polygon(cur_labels, cell)
        if not poly:
            raise RuntimeError('no polygon for cell %d' % cell)

        nis_util.open_image(nis_exe, survey_file)
        roi_id = nis_util.add_polygon_roi(nis_exe, poly, color='green')
        if roi_id <= 0:
            raise RuntimeError('ROI creation failed (id=%d)' % roi_id)
        nis_util.set_roi_type(nis_exe, roi_id, 3)  # 3 = stimulation

        nis_util.set_optical_configuration(nis_exe, frap_oc)
        t0 = time.time()
        nis_util.run_stimulation_experiment(nis_exe)
        print('[c%02d] stimulation done (%.1f s)' % (cycle, time.time() - t0), flush=True)

        # 6. save the FRAP timeseries (includes the stimulation ROI)
        nis_util.save_current_document(nis_exe, frap_file)

        # 7. close FRAP document (current), then delete the stimulation ROI
        #    on the survey document (after saving, so it stays in the saved
        #    FRAP file but doesn't linger for the next cycle) and close it
        nis_util.close_current_document(nis_exe, save='discard')
        doc = nis_util.get_current_document(nis_exe)
        if os.path.normcase(doc) != os.path.normcase(survey_file):
            nis_util.open_image(nis_exe, survey_file)
        nis_util.delete_roi(nis_exe, roi_id)
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
