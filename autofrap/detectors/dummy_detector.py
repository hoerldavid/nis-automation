"""
Built-in dummy detector: fixed circle + rectangle, left-half stim mask.

Usage::

    python -m autofrap.pipeline --detector autofrap/detectors/dummy_detector.py \
        --nis C:\\Program Files\\NIS-Elements\\nis_ar.exe --nx 2 --ny 2
"""
import numpy as np

from autofrap.core.image.segmentation import dummy_detect_objects
from autofrap.core.image.mask import half_object_stim_mask


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
