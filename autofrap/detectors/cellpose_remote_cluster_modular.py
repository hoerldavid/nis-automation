"""
Built-in cellpose detector with QC visualization: remote server on the
GPU machine, assembled with build_detector.

    channel 0 of the survey nd2  ->  cellpose on the server  ->
    cluster stimulation mask (channel 2)   +   RGB composite of loaded channels
    as the QC overlay background.

The compositor also applies its standard label housekeeping: objects
touching the image border are discarded (clear_border=True) and labels
are renumbered 1..N by increasing centroid distance to the image center
(relabel='distance') -- so the cell closest to the center is
stimulated first.

Server URL: ``DEFAULT_CELLPOSE_SERVER_URL`` below (edit it there if
the server moves), or per run via ``--detector-arg server_url=...``.

Usage::

    python -m autofrap.pipeline --detector autofrap/detectors/cellpose_remote_cluster_modular.py \
        --nx 2 --ny 2 --detector-arg diameter=70 --detector-arg server_url=http://... \
        --detector-arg load_channel=all --detector-arg det_channel=0

Offline test on a single image (no microscope; needs the
cellpose server):

    python -m autofrap.detectors --detector autofrap/detectors/cellpose_remote_cluster_modular.py \
        path/to/survey.nd2 --detector-arg diameter=70

--detector-arg values are routed via an explicit parameter_map:
  load_channel -> channel selection for loading (read_channel)
  det_channel  -> channel selection for remote_detect_objects
  filter_channel -> channel selection for the intensity filter
  server_url   -> cellpose server URL
  diameter, min_size, cellprob_threshold, flow_threshold,
  max_size_fraction -> cellpose eval kwargs
  threshold, metric -> intensity filter settings
"""

from autofrap.io.nd2 import read_channel
from autofrap.core.detection import build_detector
from autofrap.core.image.qc import default_visualization
from autofrap.core.image.segmentation import remote_detect_objects
from autofrap.core.image.mask import cluster_stim_mask, filter_intensity_inside

DEFAULT_CELLPOSE_SERVER_URL = 'http://10.163.69.12:8000'
SURVEY_CHANNEL = 0


def _load(survey_file, load_channel=SURVEY_CHANNEL):
    # load_channel can be int, tuple/list, or 'all'
    return read_channel(survey_file, channel=load_channel)


def _remote_detect(image, server_url=DEFAULT_CELLPOSE_SERVER_URL,
                   det_channel=0, **kwargs):
    return remote_detect_objects(image, server_url=server_url, channel=det_channel, **kwargs)

def _filter_intensity(labels, image, channel=0, threshold=550, metric='mean'):
    return filter_intensity_inside(labels, image, channel=channel, metric=metric, threshold=threshold)

detection_fun = build_detector(
    load_fun=_load,
    detector_fun=_remote_detect,
    stim_mask_fun=lambda labels, image: cluster_stim_mask(labels, image, channel=2),
    visualization_fun=default_visualization,
    filter_function=_filter_intensity,
    parameter_map={
        'load_fun':        {'load_channel': 'load_channel'},
        'detector_fun':    {'det_channel': 'det_channel',
                            'server_url': 'server_url',
                            'diameter': 'diameter',
                            'min_size': 'min_size',
                            'cellprob_threshold': 'cellprob_threshold',
                            'flow_threshold': 'flow_threshold',
                            'max_size_fraction': 'max_size_fraction'},
        'filter_function': {'filter_channel': 'channel',
                            'threshold': 'threshold',
                            'metric': 'metric'},
    },
)
