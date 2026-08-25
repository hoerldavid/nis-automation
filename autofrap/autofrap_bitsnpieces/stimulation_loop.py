"""
A minimal, copy‑and‑paste‑ready example that shows how to use
`calmutils.segmentation.merge_label_slices` with a user‑settable
IOU threshold.

Replace the placeholder functions `acquire_image`, `detect_cells`,
and `do_photostimulate` with your own microscope SDK calls.
"""

import numpy as np
from calmutils.segmentation import merge_label_slices

# ------------------------------------------------------------------
# User‑settable IOU threshold.  Set to 0.0 for the default behaviour
# (no threshold).  Increase if you want to be stricter about which
# objects are considered the same.
IOU_THRESHOLD = 0.3

# ------------------------------------------------------------------
# Helper: pick the next cell that hasn't been stimulated yet

def next_cell(seg, stimulated):
    """Return the smallest label >0 that is NOT in `stimulated`."""
    labels = np.unique(seg)
    labels = labels[labels > 0]            # drop background
    for lbl in sorted(labels):
        if lbl not in stimulated:
            return lbl
    return None  # no new cells left

# ------------------------------------------------------------------
stimulated = set()          # labels that have already been stimulated
prev_seg = None             # segmentation from the previous frame

while True:
    # ---- 1. Acquire image (replace with your camera SDK) ----
    img = acquire_image()     # <-- user‑defined

    # ---- 2. Detect cells → 2‑D label map --------------------
    seg = detect_cells(img)   # <-- your segmentation routine

    # ---- 3. If this is the first frame ----------------------
    if prev_seg is None:
        prev_seg = seg
        cell_to_stim = next_cell(seg, stimulated)
        if cell_to_stim is None:
            print("No cells detected.")
            break
    else:
        # ---- 4. Align new segmentation to the previous one ----
        merged = merge_label_slices([prev_seg, seg], iou_threshold=IOU_THRESHOLD)
        prev_remapped, cur_remapped = merged[0], merged[1]

        # Map previously stimulated labels to the *new* numbering
        # (only keep labels that survived)
        stim_map = {
            cur_remapped[prev_remapped == old_lbl].max()
            for old_lbl in stimulated
            if np.any(prev_remapped == old_lbl)  # old label still present
        }
        stimulated = stim_map

        # ---- 5. Pick next cell to stimulate ---------------------
        cell_to_stim = next_cell(cur_remapped, stimulated)
        if cell_to_stim is None:
            print("All cells have been stimulated.")
            break

    # ---- 6. Stimulate the chosen cell -------------------------
    do_photostimulate(cur_remapped, cell_to_stim)   # <-- your routine

    # ---- 7. Record that we stimulated this cell ---------------
    stimulated.add(cell_to_stim)

    # ---- 8. Prepare for next loop ----------------------------
    prev_seg = cur_remapped

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# End of script.
""