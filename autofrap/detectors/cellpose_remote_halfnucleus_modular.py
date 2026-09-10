"""
Built-in cellpose detector with QC visualization: remote server on the
GPU machine, assembled with build_detector.

    channel 0 of the survey nd2  ->  cellpose on the server  ->
    left-half stimulation mask   +   the DAPI channel itself as
    the QC overlay background.

The compositor also applies its standard label housekeeping: objects
touching the image border are discarded (clear_border=True) and labels
are renumbered 1..N by increasing centroid distance to the image center
(relabel='distance') -- so the cell closest to the center is
stimulated first.

Uses ``CELLPOSE_SERVER_URL`` environment variable (default:
``http://10.163.69.12:8000``).

Usage::

    export CELLPOSE_SERVER_URL=http://10.163.69.12:8000
    autofrap_grid --detector autofrap/detectors/cellpose_remote_halfnucleus_modular.py \
        --nx 2 --ny 2 --detector-arg diameter=70

--detector-arg values are routed to the server call
(parameter_map='auto'): diameter, min_size, cellprob_threshold,
flow_threshold, max_size_fraction.
"""
import os
import sys
from functools import partial

# Ensure the repo root is on sys.path (needed when run as __main__)
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from autofrap import nd2_helpers
from autofrap.detection import build_detector, remote_detect_objects
from autofrap.mask_utils import half_object_stim_mask

CELLPOSE_SERVER_URL = os.environ.get(
    'CELLPOSE_SERVER_URL', 'http://10.163.69.12:8000')
SURVEY_CHANNEL = 0


detection_fun = build_detector(
    load_fun=partial(nd2_helpers.read_channel, channel=SURVEY_CHANNEL),
    detector_fun=partial(remote_detect_objects,
                         server_url=CELLPOSE_SERVER_URL),
    stim_mask_fun=lambda labels, image: half_object_stim_mask(labels),
    # QC visualization: the loaded DAPI channel itself (2D; rendered
    # grayscale with 1-99.5 % percentile clipping by qc.save_qc_overlay)
    visualization_fun=lambda image: image,
    parameter_map='auto',  # --detector-arg diameter=... reaches the server
)


if __name__ == '__main__':
    # Standalone test: run the detector on a provided nd2 file
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='survey nd2 file')
    parser.add_argument('--diameter', type=float, default=None,
                        help='cellpose diameter parameter')
    parser.add_argument('--min-size', type=int, default=None,
                        help='cellpose min_size parameter')
    args = parser.parse_args()

    kwargs = {k: v for k, v in
              (('diameter', args.diameter), ('min_size', args.min_size))
              if v is not None}
    det = detection_fun(args.file, **kwargs)
    shapes = [d.shape for d in det if hasattr(d, 'shape')]
    print(f'returned {len(det)} array(s), shapes: {shapes}')
    labels = det[0]
    print(f'labels: {labels.shape} {labels.dtype}, objects={labels.max()}')
