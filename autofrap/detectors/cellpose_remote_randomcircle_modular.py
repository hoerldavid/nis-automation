"""
Built-in cellpose detector with QC visualization: remote server on the
GPU machine, assembled with build_detector.

    channel 0 of the survey nd2  ->  cellpose on the server  ->
    random-circle stimulation mask   +   RGB composite of loaded channels
    as the QC overlay background.

The compositor also applies its standard label housekeeping: objects
touching the image border are discarded (clear_border=True) and labels
are renumbered 1..N by increasing centroid distance to the image center
(relabel='distance') -- so the cell closest to the center is
stimulated first.

Uses ``CELLPOSE_SERVER_URL`` environment variable (default:
``http://10.163.69.12:8000``), or an explicit ``server_url`` passed via
``--detector-arg server_url=...``.

Usage::

    python -m autofrap.pipeline --detector autofrap/detectors/cellpose_remote_randomcircle_modular.py \
        --nx 2 --ny 2 --detector-arg diameter=70 --detector-arg server_url=http://... \
        --detector-arg load_channel=all --detector-arg det_channel=0

--detector-arg values are routed via parameter_map='auto':
  load_channel -> load function channel selection for read_channel
  det_channel  -> channel selection for remote_detect_objects
  server_url   -> cellpose server URL
  diameter, min_size, ... -> cellpose eval kwargs
"""
import os

from autofrap.io.nd2 import read_channel
from autofrap.core.detection import build_detector
from autofrap.core.image.qc import default_visualization
from autofrap.core.image.segmentation import remote_detect_objects
from autofrap.core.image.mask import random_circle_stim_mask, filter_intensity_inside

DEFAULT_CELLPOSE_SERVER_URL = 'http://10.163.69.12:8000'
SURVEY_CHANNEL = 0


def _load(survey_file, load_channel=SURVEY_CHANNEL):
    # load_channel can be int, tuple/list, or 'all'
    return read_channel(survey_file, channel=load_channel)


def _remote_detect(image, server_url=None, det_channel=0, **kwargs):
    if server_url is None:
        server_url = os.environ.get('CELLPOSE_SERVER_URL', DEFAULT_CELLPOSE_SERVER_URL)
    return remote_detect_objects(image, server_url=server_url, channel=det_channel, **kwargs)

def _filter_intensity(labels, image, channel=0, threshold=550, metric='mean'):
    return filter_intensity_inside(labels, image, channel=channel, metric=metric, threshold=threshold)

detection_fun = build_detector(
    load_fun=_load,
    detector_fun=_remote_detect,
    stim_mask_fun=lambda labels, image: random_circle_stim_mask(labels, area_fraction=0.25),
    visualization_fun=default_visualization,
    filter_function=_filter_intensity,
    parameter_map='auto',  # --detector-arg load_channel=..., det_channel=..., server_url=..., diameter=...
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
