"""
Built-in dummy detector: fixed circle + rectangle, left-half stim mask.

Usage::

    autofrap_grid --detector autofrap/autofrap_bitsnpieces/dummy_detector.py \
        --nis C:\\Program Files\\NIS-Elements\\nis_ar.exe --nx 2 --ny 2
"""
import os
import sys

# Ensure the repo root is on sys.path (needed when run as __main__)
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np

from autofrap.detection import dummy_detect_objects
from autofrap.mask_utils import half_object_stim_mask


def detection_fun(survey_file):
    """
    Dummy detection function.

    Parameters
    ----------
    survey_file: str
        ignored (the dummy detector uses a fixed 512×512 canvas)

    Returns
    -------
    labels, stim_mask : tuple of np.ndarray
        labels: 2D integer array (0 = background, 1 = circle, 2 = rect)
        stim_mask: 2D boolean array (left half of each object)
    """
    image = np.zeros((512, 512), dtype=np.uint16)
    labels = dummy_detect_objects(image)
    stim_mask = half_object_stim_mask(labels)
    return labels, stim_mask


if __name__ == '__main__':
    labels, stim_mask = detection_fun('/dev/null')
    print(f'labels shape: {labels.shape}, dtype: {labels.dtype}')
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        area = np.sum(labels == lbl)
        stim_area = np.sum((labels == lbl) & stim_mask)
        print(f'  label {lbl}: {area} px, stim: {stim_area} px')
