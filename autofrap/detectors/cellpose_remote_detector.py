"""
Built-in cellpose detector: remote server on GPU machine.

Uses ``CELLPOSE_SERVER_URL`` environment variable (default:
``http://10.163.69.12:8000``) to connect to the cellpose server.
Channel 0 is loaded from the survey.

Usage::

    export CELLPOSE_SERVER_URL=http://10.163.69.12:8000
    autofrap_grid --detector autofrap/detectors/cellpose_remote_detector.py \
        --nx 2 --ny 2 --detector-arg diameter=70
"""
import os
import sys

# Ensure the repo root is on sys.path (needed when run as __main__)
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from autofrap.detection import remote_detect_objects
from autofrap.mask_utils import half_object_stim_mask

CELLPOSE_SERVER_URL = os.environ.get(
    'CELLPOSE_SERVER_URL', 'http://10.163.69.12:8000')
SURVEY_CHANNEL = 0


def detection_fun(survey_file, **detector_kwargs):
    """
    Cellpose remote detection function.

    Parameters
    ----------
    survey_file: str
        path to the survey nd2 file
    detector_kwargs: dict, optional
        cellpose model.eval() parameters forwarded to the server
        (diameter, min_size, cellprob_threshold, flow_threshold,
        max_size_fraction); from the CLI these come from
        --detector-arg KEY=VALUE (repeatable)

    Returns
    -------
    labels, stim_mask : tuple of np.ndarray
        labels: 2D integer array (0 = background, 1..N = detected cells)
        stim_mask: 2D boolean array (left half of each object)
    """
    from autofrap import nd2_helpers

    image = nd2_helpers.read_channel(survey_file, channel=SURVEY_CHANNEL)
    labels = remote_detect_objects(image, server_url=CELLPOSE_SERVER_URL,
                                   **detector_kwargs)
    stim_mask = half_object_stim_mask(labels)
    return labels, stim_mask


if __name__ == '__main__':
    # Standalone test: run on a provided nd2 file
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='survey nd2 file')
    parser.add_argument('--diameter', type=float, default=None,
                        help='cellpose diameter parameter')
    parser.add_argument('--min-size', type=int, default=None,
                        help='cellpose min_size parameter')
    args = parser.parse_args()

    try:
        kwargs = {k: v for k, v in
                  (('diameter', args.diameter), ('min_size', args.min_size))
                  if v is not None}
        labels, stim_mask = detection_fun(args.file, **kwargs)
        print(f'labels shape: {labels.shape}, dtype: {labels.dtype}')
        print(f'stim_mask shape: {stim_mask.shape}')
        print(f'unique labels: {list(np.unique(labels))}')
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
