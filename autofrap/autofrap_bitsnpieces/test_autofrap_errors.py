"""
offline test of the autofrap() error handling (no microscope needed):
the nis_util calls are faked and each failure mode is checked for the
exception class it produces (RecoverableError vs NonRecoverableError),
plus the autofrap_grid continue/abort policy and the remote client retry.

run: python autofrap/autofrap_bitsnpieces/test_autofrap_errors.py
"""
import contextlib
import io
import os
import shutil
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests

import autofrap
import detection
import nis_util

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_SURVEY = os.path.join(ROOT, 'test_acquisitions',
                           'nuclei_20260901_110410.nd2')
NIS = 'fake_nis'

failures = 0


def dummy_detect(survey_file):
    return detection.detect(survey_file, channel=0, detector='dummy')


class FakeNIS:
    """fake the nis_util calls used by autofrap()/autofrap_grid()"""

    def __init__(self, fail_survey=False, fail_save=False, roi_id=1,
                 open_broken=False, abort_add_roi=False, fail_move=False):
        self.fail_survey = fail_survey
        self.fail_save = fail_save
        self.roi_id = roi_id
        self.open_broken = open_broken
        self.abort_add_roi = abort_add_roi
        self.fail_move = fail_move
        self.calls = []
        self.current_doc = 'Frozen'
        self._next_roi = 0

    def __enter__(self):
        self.patches = [
            mock.patch.object(nis_util, 'run_current_nd_experiment', self.run_nd),
            mock.patch.object(nis_util, 'open_image', self.open_image),
            mock.patch.object(nis_util, 'get_current_document', self.get_doc),
            mock.patch.object(nis_util, 'add_polygon_roi', self.add_roi),
            mock.patch.object(nis_util, 'set_roi_type',
                              lambda n, i, t: self._call('set_roi_type', t)),
            mock.patch.object(nis_util, 'set_optical_configuration',
                              lambda n, oc: self._call('set_optical_configuration', oc)),
            mock.patch.object(nis_util, 'run_stimulation_experiment',
                              lambda n: self._call('run_stimulation_experiment')),
            mock.patch.object(nis_util, 'save_current_document', self.save_doc),
            mock.patch.object(nis_util, 'close_current_document',
                              lambda n, save='discard': self._call('close', save)),
            mock.patch.object(nis_util, 'delete_roi',
                              lambda n, i: self._call('delete_roi', i)),
            mock.patch.object(nis_util, 'get_position',
                              lambda n: (1000.0, 2000.0, 0.0)),
            mock.patch.object(nis_util, 'get_resolution',
                              lambda n: (1024, 1024, 13.0, 100.0)),
            mock.patch.object(nis_util, 'set_position', self.set_position),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self.patches:
            p.stop()

    def _call(self, name, *args):
        self.calls.append((name, args))

    def run_nd(self, n, outfile=None, open_after=True, progress_bar=True):
        self._call('run_nd', outfile)
        if not self.fail_survey:
            shutil.copy(REAL_SURVEY, outfile)

    def open_image(self, n, image_path):
        self._call('open_image', image_path)
        if not self.open_broken:
            self.current_doc = image_path

    def get_doc(self, n):
        self._call('get_doc')
        return self.current_doc

    def add_roi(self, n, points, color='green'):
        self._call('add_roi', len(points))
        if self.abort_add_roi:
            raise KeyError('id')
        self._next_roi += 1
        return self.roi_id

    def save_doc(self, n, outfile):
        self._call('save_doc', outfile)
        if not self.fail_save:
            open(outfile, 'wb').close()

    def set_position(self, n, pos_xy=None):
        if self.fail_move:
            self.fail_move = False  # fail only the first move
            raise KeyError('pos')
        self._call('set_position', pos_xy)

    def calls_of(self, name):
        return [c for c in self.calls if c[0] == name]


def check(name, fn, expect):
    """run fn(), check it raises `expect` (or returns when expect is None)"""
    global failures
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
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


_run_counter = 0


def run_autofrap(tmp, **kw):
    global _run_counter
    _run_counter += 1
    out_dir = os.path.join(tmp, f'run{_run_counter}')
    expect = kw.pop('expect', None)
    with FakeNIS(**kw) as fake:
        res = check('autofrap',
                    lambda: autofrap.autofrap(NIS, out_dir, max_cycles=1,
                                              detection_fun=dummy_detect),
                    expect)
    return res, fake


def main():
    global failures
    import tempfile

    R = autofrap.RecoverableError
    NR = autofrap.NonRecoverableError

    with tempfile.TemporaryDirectory() as tmp:
        # 1. happy path
        res, fake = run_autofrap(tmp)
        ok = res is not None and len(res) == 1 and os.path.isfile(res[0][3])
        print(f'{"ok  " if ok else "FAIL"} happy path: results={len(res) if res else None}, '
              f'frap saved={bool(res and os.path.isfile(res[0][3]))}, '
              f'ROIs deleted={len(fake.calls_of("delete_roi"))}')
        if not ok:
            failures += 1

        # 2. NIS did not save the survey
        res, fake = run_autofrap(tmp, fail_survey=True, expect=NR)

        # 3. detection server unreachable
        def down(survey_file):
            raise requests.exceptions.ConnectionError('connection refused')
        with FakeNIS():
            check('server down',
                  lambda: autofrap.autofrap(NIS, os.path.join(tmp, 'run3'),
                                            max_cycles=1, detection_fun=down),
                  NR)

        # 4. detection server error on this image
        def http500(survey_file):
            raise requests.exceptions.HTTPError('500 Internal Server Error')
        with FakeNIS():
            check('server 500',
                  lambda: autofrap.autofrap(NIS, os.path.join(tmp, 'run4'),
                                            max_cycles=1, detection_fun=http500),
                  R)

        # 5. other detection failure (e.g. corrupt survey file)
        def boom(survey_file):
            raise ValueError('corrupt file')
        with FakeNIS():
            check('detection other',
                  lambda: autofrap.autofrap(NIS, os.path.join(tmp, 'run5'),
                                            max_cycles=1, detection_fun=boom),
                  NR)

        # 6. ROI creation failed -> recoverable + cleanup ran
        res, fake = run_autofrap(tmp, roi_id=-1, expect=R)
        ok = fake.calls_of('delete_roi')
        print(f'{"ok  " if ok else "FAIL"} ROI failure cleanup: '
              f'delete_roi calls={[c[1] for c in fake.calls_of("delete_roi")]}')
        if not ok:
            failures += 1

        # 7. NIS macro aborted (empty ini read-back)
        res, fake = run_autofrap(tmp, abort_add_roi=True, expect=NR)

        # 8. FRAP file not saved
        res, fake = run_autofrap(tmp, fail_save=True, expect=NR)

        # 9. survey could not be opened in NIS
        res, fake = run_autofrap(tmp, open_broken=True, expect=NR)

    # 10. grid policy: fov01 recoverable -> skip, fov02 fatal -> abort,
    #     fov03 not visited; stage returns to start
    positions = [(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)]

    def grid_detect(survey_file):
        if 'fov01' in survey_file:
            raise requests.exceptions.HTTPError('500')
        if 'fov02' in survey_file:
            raise requests.exceptions.ConnectionError('refused')
        return dummy_detect(survey_file)

    with tempfile.TemporaryDirectory() as tmp:
        with FakeNIS() as fake:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                results = autofrap.autofrap_grid(NIS, tmp, positions=positions,
                                                 settle_s=0.0,
                                                 detection_fun=grid_detect,
                                                 max_cycles=1)
            log = out.getvalue()
        ok = (len(results) == 2 and results[0][4] is None
              and results[1][4] is None
              and 'ABORTED at FOV 2' in log
              and 'not visited' in log
              and 'moved back to start' in log
              and not any('fov03' in r[3] for r in results))
        print(f'{"ok  " if ok else "FAIL"} grid policy: {len(results)} FOV(s) visited, '
              f'aborted={("ABORTED" in log)}, returned to start={("moved back to start" in log)}')
        if not ok:
            failures += 1

    # 11. grid: stage move fails -> abort at the first FOV, still returns
    #     to start
    with tempfile.TemporaryDirectory() as tmp:
        with FakeNIS(fail_move=True) as fake:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                results = autofrap.autofrap_grid(NIS, tmp, positions=positions,
                                                 settle_s=0.0,
                                                 detection_fun=grid_detect,
                                                 max_cycles=1)
            log = out.getvalue()
        ok = (len(results) == 1 and results[0][4] is None
              and 'stage move failed' in log
              and 'ABORTED at FOV 1' in log
              and 'moved back to start' in log)
        print(f'{"ok  " if ok else "FAIL"} grid stage-move failure: '
              f'{len(results)} entry/ies, aborted={("ABORTED" in log)}, '
              f'returned to start={("moved back to start" in log)}')
        if not ok:
            failures += 1

    # 12. remote client retry
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[4:12, 4:12] = 1

    def fake_response():
        buf = io.BytesIO()
        np.save(buf, labels)

        class R:
            content = buf.getvalue()
            headers = {'X-Inference-Time-S': '1.0', 'X-N-Objects': '1'}

            def raise_for_status(self):
                pass
        return R()

    with tempfile.TemporaryDirectory() as tmp:
        import time as _t
        with mock.patch('requests.post',
                        side_effect=[requests.exceptions.HTTPError('500'),
                                     fake_response()]):
            t0 = _t.time()
            got = check('retry after 500',
                        lambda: detection.remote_detect_objects(
                            labels, 'http://fake', timeout=5, retries=1),
                        None)
        ok = got is not None and np.array_equal(got, labels.astype(np.int32))
        print(f'{"ok  " if ok else "FAIL"} retry after 500: '
              f'{_t.time() - t0:.1f} s (includes 2 s backoff)')
        if not ok:
            failures += 1

        with mock.patch('requests.post',
                        side_effect=requests.exceptions.HTTPError('500')):
            check('500 twice',
                  lambda: detection.remote_detect_objects(
                      labels, 'http://fake', timeout=5, retries=1),
                  requests.exceptions.HTTPError)

        with mock.patch('requests.post',
                        side_effect=[requests.exceptions.ConnectionError('refused'),
                                     fake_response()]):
            got = check('retry after connection error',
                        lambda: detection.remote_detect_objects(
                            labels, 'http://fake', timeout=5, retries=1),
                        None)
        ok = got is not None and np.array_equal(got, labels.astype(np.int32))
        print(f'{"ok  " if ok else "FAIL"} retry after connection error')
        if not ok:
            failures += 1

    print(f'\n{failures} failure(s)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
