"""
one-off cellpose test: run the cpdino-vitb model on one channel of an
ND2 file and report the segmentation (no normalization: cellpose
normalizes internally)

usage: python test_cellpose.py <nd2_file> [channel] [--server URL]

--server URL runs detection via the remote client (cellpose_server.py)
instead of a local model; useful to A/B local vs. remote on the same
image.

also saves two previews next to the input file:
  <stem>_cellpose_image.png    contrast-stretched input channel
  <stem>_cellpose_labels.png   colored label map (one random color per object)
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import read_channel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('nd2_file',
                   default=r'C:\Users\David\Desktop\nis-automation'
                           r'\test_acquisitions\nuclei_20260901_110410.nd2',
                   nargs='?')
    p.add_argument('channel', type=int, nargs='?', default=0)
    p.add_argument('--server', default=None,
                   help="base URL of a running cellpose_server.py, "
                        "e.g. http://192.168.1.10:8000")
    a = p.parse_args()

    img = read_channel(a.nd2_file, a.channel)
    print(f'image: shape={img.shape} dtype={img.dtype} range={img.min()}..{img.max()}')

    if a.server:
        from detection import remote_detect_objects
        t0 = time.time()
        masks = remote_detect_objects(img, a.server)
        dt = time.time() - t0
        f = a.nd2_file
    else:
        import cellpose.models as models
        t0 = time.time()
        model = models.CellposeModel(pretrained_model='cpdino-vitb')
        print(f'model init: {time.time() - t0:.1f} s')

        t0 = time.time()
        masks, flows, styles = model.eval(img)
        dt = time.time() - t0
        f = a.nd2_file

    vals, counts = np.unique(masks, return_counts=True)
    n_obj = int((vals > 0).sum())
    print(f'inference: {dt:.1f} s, masks shape={masks.shape} dtype={masks.dtype}, '
          f'{n_obj} object(s)')
    for v, c in zip(vals, counts):
        if v > 0:
            print(f'  label {int(v)}: {int(c)} px')

    # previews
    stem = os.path.splitext(f)[0]
    stretched = img.astype(np.float32)
    lo, hi = stretched.min(), np.percentile(stretched, 99.5)
    stretched = 255.0 * np.clip(stretched, lo, hi) / (hi - lo)
    Image.fromarray(stretched.astype(np.uint8)).save(stem + '_cellpose_image.png')

    lab_img = np.zeros(img.shape + (3,), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for i in range(1, int(masks.max()) + 1):
        lab_img[masks == i] = rng.integers(60, 256, 3)
    Image.fromarray(lab_img).save(stem + '_cellpose_labels.png')
    print(f'previews saved: {stem}_cellpose_image.png, {stem}_cellpose_labels.png')


if __name__ == '__main__':
    main()
