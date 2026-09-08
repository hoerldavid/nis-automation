"""
Example custom detector file for autofrap_grid --detector.

This is a simple detector using the dummy detector (fixed circle +
rectangle) with a left-half stimulation mask. It demonstrates the
minimal contract: define ``detection_fun(survey_file)`` and return
(labels, stim_mask) — or just (labels,) for whole-cell FRAP.

Usage::

    autofrap_grid --detector autofrap/autofrap_bitsnpieces/example_detector.py \
        --nis C:\\Program Files\\NIS-Elements\\nis_ar.exe --nx 1 --ny 1

The file is imported by the runner; it must define a top-level
callable named ``detection_fun`` with the signature:

    survey_file -> (labels[, stim_mask[, viz]])

where labels is a 2D integer array (0 = background, 1..N = objects).
"""
import os
import sys

# Ensure the repo root is on sys.path (needed when run as __main__).
# The file lives in autofrap/autofrap_bitsnpieces/; the repo root is
# two levels up from here.
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np

from autofrap.detection import dummy_detect_objects
from autofrap.mask_utils import half_object_stim_mask


def detection_fun(survey_file):
    """
    Example detection function using the dummy detector.

    Parameters
    ----------
    survey_file: str
        path to the survey image (ignored by the dummy detector,
        only the shape matters — a real detector would read this file)

    Returns
    -------
    labels, stim_mask : tuple of np.ndarray
        labels: 2D integer array (0 = background, 1..N = objects)
        stim_mask: 2D boolean array of FRAP-eligible areas
    """
    # In a real detector, you would load the image here, e.g.:
    #   import nd2_helpers
    #   image = nd2_helpers.read_channel(survey_file, channel=0)

    # For this example, just create a dummy image with the right shape.
    # A real detector would use the actual image dimensions.
    image = np.zeros((512, 512), dtype=np.uint16)

    # Run the dummy detector
    labels = dummy_detect_objects(image)

    # Compute the stimulation mask (left half of each object)
    stim_mask = half_object_stim_mask(labels)

    return labels, stim_mask


if __name__ == '__main__':
    # Standalone test: run the detector on a synthetic image and print
    # the result for verification.
    labels, stim_mask = detection_fun('/dev/null')
    print(f'labels shape: {labels.shape}, dtype: {labels.dtype}')
    print(f'unique labels: {np.unique(labels)}')
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        area = np.sum(labels == lbl)
        stim_area = np.sum((labels == lbl) & stim_mask)
        print(f'  label {lbl}: {area} px, stim: {stim_area} px')
