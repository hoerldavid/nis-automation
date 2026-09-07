"""
one-off contract test for build_detector()'s visualization_fun
(run: python autofrap/autofrap_bitsnpieces/test_detect_viz.py)

Synthetic data only, no nd2 / server needed. Checks the return-tuple
positions (2 = mask, 3 = viz, fixed) for all four combinations of
stim_mask_fun / visualization_fun, and that a failing visualization
degrades to no viz with a warning instead of raising (the labels/mask
contract checks must still raise).
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # autofrap/

import numpy as np

import detection
import mask_utils

FAILURES = []


def check(name, cond, detail=''):
    print(f"{'ok  ' if cond else 'FAIL'} {name}"
          + ('' if cond else f' - {detail}'))
    if not cond:
        FAILURES.append(name)


def load2d(f):
    return np.zeros((64, 48), dtype=np.uint16)


def loadrgb(f):
    return np.zeros((3, 64, 48), dtype=np.uint16)  # (c, y, x)


det = detection.dummy_detect_objects
maskfun = lambda labels, image: mask_utils.half_object_stim_mask(labels)

# 1. no mask, no viz -> (labels,)  [the 1-tuple autofrap() requires]
res = detection.build_detector(load2d, det)('x')
check('no mask/no viz: 1-tuple', isinstance(res, tuple) and len(res) == 1)
check('no mask/no viz: labels shape', res[0].shape == (64, 48))
check('no mask/no viz: labels integer',
      np.issubdtype(res[0].dtype, np.integer))

# 2. mask, no viz -> (labels, mask)
res = detection.build_detector(load2d, det, stim_mask_fun=maskfun)('x')
check('mask/no viz: 2-tuple', isinstance(res, tuple) and len(res) == 2)
check('mask/no viz: mask shape+dtype',
      res[1].shape == (64, 48) and res[1].dtype == bool)
check('mask/no viz: mask non-empty', bool(res[1].any()))

# 3. viz, no mask -> (labels, None, viz)  [position 2 stays the mask]
res = detection.build_detector(load2d, det,
                               visualization_fun=lambda image: image)('x')
check('viz/no mask: 3-tuple', isinstance(res, tuple) and len(res) == 3)
check('viz/no mask: position 2 is None', res[1] is None)
check('viz/no mask: viz shape', res[2].shape == (64, 48))

# 4. mask + viz -> (labels, mask, viz)
res = detection.build_detector(load2d, det, stim_mask_fun=maskfun,
                               visualization_fun=lambda image: image)('x')
check('mask+viz: 3-tuple', isinstance(res, tuple) and len(res) == 3)
check('mask+viz: both present',
      res[1] is not None and res[2] is not None)

# 5. multi-channel (c, y, x) load, 2D grayscale viz from a channel
res = detection.build_detector(loadrgb, det,
                               visualization_fun=lambda image:
                               image[0])('x')
check('c,y,x load: 2D viz', len(res) == 3 and res[2].shape == (64, 48))

# 6. multi-channel load, scientific (c, y, x) -> display (y, x, 3)
res = detection.build_detector(loadrgb, det,
                               visualization_fun=lambda image:
                               np.transpose(image, (1, 2, 0))
                               .astype(np.uint8))('x')
check('c,y,x load: RGB viz', len(res) == 3
      and res[2].shape == (64, 48, 3))

# 7. bad viz shape ((y, 1)) -> warn + drop, run survives
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    res = detection.build_detector(load2d, det, stim_mask_fun=maskfun,
                                   visualization_fun=lambda image:
                                   image[..., :1])('x')
check('bad viz shape: dropped', len(res) == 2)
check('bad viz shape: warned',
      len(w) == 1 and 'visualization' in str(w[0].message))

# 8. raising viz -> warn + drop, run survives
def boom(image):
    raise RuntimeError('nope')


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    res = detection.build_detector(load2d, det, stim_mask_fun=maskfun,
                                   visualization_fun=boom)('x')
check('raising viz: dropped', len(res) == 2)
check('raising viz: warned',
      len(w) == 1 and 'visualization_fun failed' in str(w[0].message))

# 9. the real contract checks still raise (viz must not swallow them)
try:
    detection.build_detector(load2d, lambda image: image.astype(bool),
                             visualization_fun=lambda image: image)('x')
    check('labels dtype check still raises', False)
except ValueError:
    check('labels dtype check still raises', True)

try:
    detection.build_detector(load2d, det,
                             stim_mask_fun=lambda labels, image:
                             np.zeros((10, 10), dtype=bool))('x')
    check('mask shape check still raises', False)
except ValueError:
    check('mask shape check still raises', True)

print()
print(f'{len(FAILURES)} failure(s)')
sys.exit(1 if FAILURES else 0)
