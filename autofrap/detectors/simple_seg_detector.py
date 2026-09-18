"""
Simple Otsu + watershed detector with QC visualization.

Built with build_detector for pipeline compatibility:
  - loads channel 0 of survey ND2
  - detects nuclei with autofrap.core.simple_seg.detect_objects
  - stimulation mask: half_object_stim_mask per cell
  - visualization: the loaded DAPI channel itself

Usage::

    autofrap_grid --detector autofrap/detectors/simple_seg_detector.py \
        --detector-arg cell_sigma=16 --detector-arg otsu_frac=0.3
"""
from functools import partial

from autofrap.io.nd2 import read_channel
from autofrap.core.detection import build_detector
from autofrap.core.image.mask import half_object_stim_mask
from autofrap.core.image.segmentation import SimpleSegParams, detect_objects

SURVEY_CHANNEL = 0


def _detect_simple(image, cell_sigma=16.0, otsu_frac=0.3, min_eroded_extent=0.90):
    params = SimpleSegParams(
        cell_sigma=cell_sigma,
        otsu_frac=otsu_frac,
        min_eroded_extent=min_eroded_extent,
    )
    return detect_objects(image, params)


detection_fun = build_detector(
    load_fun=partial(read_channel, channel=SURVEY_CHANNEL),
    detector_fun=_detect_simple,
    stim_mask_fun=lambda labels, image: half_object_stim_mask(labels),
    visualization_fun=lambda image: image,
    parameter_map='auto',
)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='survey nd2 file')
    parser.add_argument('--cell-sigma', type=float, default=None)
    parser.add_argument('--otsu-frac', type=float, default=None)
    parser.add_argument('--min-eroded-extent', type=float, default=None)
    args = parser.parse_args()
    kwargs = {k: v for k, v in [
        ('cell_sigma', args.cell_sigma),
        ('otsu_frac', args.otsu_frac),
        ('min_eroded_extent', args.min_eroded_extent),
    ] if v is not None}
    out = detection_fun(args.file, **kwargs)
    labels = out[0]
    print(f'labels shape {labels.shape}, objects={int(labels.max())}')
