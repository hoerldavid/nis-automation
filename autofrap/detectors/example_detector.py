"""
Example custom detector file for the pipeline's --detector option.

A simple detector: it reads channel 0 of the survey image, runs the
dummy detector (fixed circle + rectangle) and returns a left-half
stimulation mask. It demonstrates the minimal contract: define
``detection_fun(survey_file)`` and return (labels, stim_mask) — or just
(labels,) for whole-cell FRAP.

Usage::

    python -m autofrap.pipeline --detector autofrap/detectors/example_detector.py \
        --nis C:\\Program Files\\NIS-Elements\\nis_ar.exe --nx 1 --ny 1

The file is imported by the runner; it must define a top-level
callable named ``detection_fun`` with the signature:

    survey_file -> (labels[, stim_mask[, viz]])

where labels is a 2D integer array (0 = background, 1..N = objects).
"""
import numpy as np

from autofrap.io.nd2 import read_channel
from autofrap.core.image.segmentation import dummy_detect_objects
from autofrap.core.image.mask import half_object_stim_mask


def detection_fun(survey_file):
    """
    Example detection function using the dummy detector.

    Parameters
    ----------
    survey_file: str
        path to the survey image (nd2 file)

    Returns
    -------
    labels, stim_mask : tuple of np.ndarray
        labels: 2D integer array (0 = background, 1..N = objects)
        stim_mask: 2D boolean array of FRAP-eligible areas
    """
    # Read the survey image (channel 0). The dummy detector only needs
    # the image shape; a real detector would analyse the pixel values.
    image = read_channel(survey_file, channel=0)

    # Run the dummy detector
    labels = dummy_detect_objects(image)

    # Compute the stimulation mask (left half of each object)
    stim_mask = half_object_stim_mask(labels)

    return labels, stim_mask


if __name__ == '__main__':
    # Standalone test: run the detector on the provided nd2 file and
    # print the result for verification.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='survey nd2 file')
    args = parser.parse_args()
    labels, stim_mask = detection_fun(args.file)
    print(f'labels shape: {labels.shape}, dtype: {labels.dtype}')
    print(f'unique labels: {np.unique(labels)}')
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        area = np.sum(labels == lbl)
        stim_area = np.sum((labels == lbl) & stim_mask)
        print(f'  label {lbl}: {area} px, stim: {stim_area} px')
