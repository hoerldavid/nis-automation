"""
one-off test detector (dry-run plumbing check): loads a plain .tif with
tifffile (not an nd2 survey), runs cellpose on the remote server, and
forwards runtime parameters (diameter, min_size, ...) to the server.

Composed with build_detector(parameter_map='auto') so that
detection_fun(file, diameter=70) -- and the CLI's
--detector-arg diameter=70 -- actually reach the server.

run:
    export CELLPOSE_SERVER_URL=http://localhost:8000   # or the V100
    python - <<'EOF'
    from autofrap.detection import load_detector_file
    fun = load_detector_file('autofrap/autofrap_bitsnpieces/tif_detector.py')
    labels, stim = fun('test_data/0013_ch1.tif', diameter=70)
    print(labels.max(), 'objects')
    EOF
"""
import os
import sys

# Ensure the repo root is on sys.path
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

import tifffile

from autofrap.core.detection import build_detector, remote_detect_objects
from autofrap.core.image.mask import half_object_stim_mask

CELLPOSE_SERVER_URL = os.environ.get(
    'CELLPOSE_SERVER_URL', 'http://10.163.69.12:8000')


def load_tif(path):
    """load a plain .tif as a 2D uint16 image (no nd2 involved)"""
    return tifffile.imread(path).astype('uint16')


def cellpose_server(image, **eval_kwargs):
    """remote cellpose; runtime params (diameter, min_size, ...) pass
    through to the server's model.eval"""
    return remote_detect_objects(image, server_url=CELLPOSE_SERVER_URL,
                                 **eval_kwargs)


detection_fun = build_detector(
    load_fun=load_tif,
    detector_fun=cellpose_server,
    stim_mask_fun=lambda labels, image: half_object_stim_mask(labels),
    parameter_map='auto',  # route runtime kwargs to the sub-functions
)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('file', help='plain .tif image')
    p.add_argument('--diameter', type=float, default=None)
    p.add_argument('--min-size', type=int, default=None)
    a = p.parse_args()

    kwargs = {k: v for k, v in
              (('diameter', a.diameter), ('min_size', a.min_size))
              if v is not None}
    labels, stim = detection_fun(a.file, **kwargs)
    print(f'{labels.max()} objects, {stim.sum()} stim px')
