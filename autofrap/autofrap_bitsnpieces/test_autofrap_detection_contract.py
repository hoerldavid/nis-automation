"""
offline test of the autofrap() detection_fun contract (TODOs #21, #22):
  - return shape: bare label map (ndarray) and 1-3 tuples/lists are
    accepted, anything else -> NonRecoverableError
  - failure policy: ANY detection failure (requests ConnectionError /
    Timeout / HTTPError, or a plain Exception) -> NonRecoverableError
    (run-level abort; no per-exception translation in autofrap.py,
    which no longer imports requests)
the nis_util calls are faked; the "survey file" is an empty placeholder
(the fake detectors never read it).

run: python autofrap/autofrap_bitsnpieces/test_autofrap_detection_contract.py
"""
import contextlib
import io
import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

import autofrap
import nis_util

failures = 0


def make_labels():
    """64x64 label map with one 16-px-diameter circle (cell 1)"""
    labels = np.zeros((64, 64), dtype=np.int32)
    yy, xx = np.ogrid[:64, :64]
    labels[(yy - 32) ** 2 + (xx - 32) ** 2 <= 8 ** 2] = 1
    return labels


LABELS = make_labels()
MASK = LABELS > 0  # whole cell is stim-eligible
VIZ = LABELS.astype(np.float32)


def det_bare(sf):
    return LABELS


def det_tuple1(sf):
    return (LABELS,)


def det_tuple2(sf):
    return (LABELS, MASK)


def det_tuple3(sf):
    return (LABELS, MASK, VIZ)


def det_list(sf):
    return [LABELS]


def det_4tuple(sf):
    return (LABELS, MASK, VIZ, None)


def det_string(sf):
    return 'labels'


def make_raiser(exc):
    def f(sf):
        raise exc
    return f


class FakeNIS:
    """fake the nis_util calls used by one autofrap() cycle"""

    def __init__(self):
        self.current_doc = 'Frozen'

    def __enter__(self):
        self.patches = [
            mock.patch.object(nis_util, 'run_current_nd_experiment',
                              self.run_nd),
            mock.patch.object(nis_util, 'open_image', self.open_image),
            mock.patch.object(nis_util, 'get_current_document',
                              self.get_doc),
            mock.patch.object(nis_util, 'add_polygon_roi', self.add_roi),
            mock.patch.object(nis_util, 'set_roi_type',
                              lambda n, i, t: None),
            mock.patch.object(nis_util, 'set_optical_configuration',
                              lambda n, oc: None),
            mock.patch.object(nis_util, 'run_stimulation_experiment',
                              lambda n: None),
            mock.patch.object(nis_util, 'save_current_document',
                              self.save_doc),
            mock.patch.object(nis_util, 'close_current_document',
                              lambda n, save='discard': None),
            mock.patch.object(nis_util, 'delete_roi', lambda n, i: None),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self.patches:
            p.stop()

    def run_nd(self, n, outfile=None, open_after=True, progress_bar=True):
        open(outfile, 'wb').close()  # empty placeholder, never read
        self.current_doc = outfile

    def open_image(self, n, image_path):
        self.current_doc = image_path

    def get_doc(self, n):
        return self.current_doc

    def add_roi(self, n, points, color='green'):
        return 1

    def save_doc(self, n, outfile):
        open(outfile, 'wb').close()


def run_cycle(detection_fun):
    with tempfile.TemporaryDirectory() as tmp:
        with FakeNIS():
            with contextlib.redirect_stdout(io.StringIO()):
                return autofrap.autofrap('fake_nis', tmp, max_cycles=1,
                                         detection_fun=detection_fun,
                                         file_prefix='t')


def check(name, fn, expect):
    """run fn(), check it raises `expect` (or returns when expect is None)"""
    global failures
    try:
        result = fn()
    except Exception as e:
        if expect is not None and isinstance(e, expect):
            print(f'ok   {name}: {type(e).__name__}: {e}')
            return e
        failures += 1
        print(f'FAIL {name}: expected {expect}, got {type(e).__name__}: {e}')
        return None
    if expect is None:
        print(f'ok   {name}: no exception')
        return result
    failures += 1
    print(f'FAIL {name}: expected {expect}, got no exception')
    return None


# 1. return shape: bare label map and 1-3 tuples/lists
print('-- return shape')
for name, det in [('bare label map (ndarray)', det_bare),
                  ('1-tuple (labels,)', det_tuple1),
                  ('2-tuple (labels, mask)', det_tuple2),
                  ('3-tuple (labels, mask, viz)', det_tuple3),
                  ('1-list [labels]', det_list)]:
    results = check(name, lambda d=det: run_cycle(d), None)
    if results is not None and len(results) != 1:
        failures += 1
        print(f'FAIL {name}: expected 1 result, got {results}')

print('-- malformed returns')
check('4-tuple rejected', lambda: run_cycle(det_4tuple),
      autofrap.NonRecoverableError)
check('string rejected', lambda: run_cycle(det_string),
      autofrap.NonRecoverableError)

# 2. failure policy: any detection failure -> NonRecoverableError
print('-- detection failure -> NonRecoverableError')
for name, exc in [
        ('ConnectionError', requests.exceptions.ConnectionError('down')),
        ('Timeout', requests.exceptions.Timeout('slow')),
        ('HTTPError (5xx)', requests.exceptions.HTTPError('500 server error')),
        ('plain RuntimeError', RuntimeError('model crash'))]:
    check(name, lambda e=exc: run_cycle(make_raiser(e)),
          autofrap.NonRecoverableError)

# 3. autofrap.py no longer imports requests
print('-- no requests dependency in autofrap.py')
if hasattr(autofrap, 'requests'):
    failures += 1
    print('FAIL autofrap module has no requests attribute: still imported')
else:
    print('ok   autofrap module has no requests attribute')

print()
print(f'{failures} failure(s)')
sys.exit(1 if failures else 0)
