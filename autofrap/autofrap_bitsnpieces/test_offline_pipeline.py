"""
Offline assertion suite for the autoFRAP pipeline against FakeNIS
(no microscope, no detector server - the dummy detector is used).

Sections:
  A. fake sanity      - per-FOV source wrapping, FRAP copies, cross-cycle
                        cell matching, per-cycle cleanup, no patch leak
  B. pre-flight       - AbortRunError: invalid name, non-empty run dir,
                        failed setup read
  C. FOV-level policy - consecutive-failure policy: abort at the limit,
                        counter reset on success, one-shot vs systemic
                        failures, stage-move failures, detection failures
  D. clean stop       - summary printed, AutofrapInterruptedException
                        re-raised (CLI exit 130)
  E. detector contract - detection_fun return shapes: accepted forms
                        and wrong-arity rejection

run: python autofrap/autofrap_bitsnpieces/test_offline_pipeline.py
"""
import contextlib
import filecmp
import io
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from autofrap.pipeline import autofrap as af
from autofrap.core.detection import load_detector_file
from autofrap.microscope import fake_nis  # noqa: F401  (imported for parity)
from autofrap.microscope.fake_nis import FakeNIS, PATCHED_FUNCTIONS
import autofrap.microscope.nis as nis_util

detection_fun = load_detector_file(
    os.path.join(ROOT, 'autofrap', 'detectors', 'dummy_detector.py'))

n_failures = 0


def check(name, cond, extra=''):
    global n_failures
    print(f'{"ok  " if cond else "FAIL"} {name}'
          + (f' - {extra}' if extra else ''))
    if not cond:
        n_failures += 1


# remember the originals for the leak check at the end
originals = {name: getattr(nis_util, name)
             for name in PATCHED_FUNCTIONS}

POS5 = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]
POS4 = POS5[:4]
POS2 = POS5[:2]

with tempfile.TemporaryDirectory() as TMP:
    # two fake "survey nd2" sources (opaque contents: the dummy detector
    # never reads the survey file)
    src1 = os.path.join(TMP, 'src1.nd2')
    src2 = os.path.join(TMP, 'src2.nd2')
    for s, tag in ((src1, 1), (src2, 2)):
        with open(s, 'wb') as f:
            f.write(f'fake-nd2-{tag}'.encode())

    def run_outer(out, name, fake, positions, max_cycles=1,
                  detection_fun=detection_fun, **outer_kwargs):
        """autofrap_loop_outer inside FakeNIS; returns (results, log)"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with fake:
                results = af.autofrap_loop_outer(
                    'fake', out, positions, max_cycles=max_cycles,
                    detection_fun=detection_fun,
                    name=name, use_timestamp=False, **outer_kwargs)
        return results, buf.getvalue()

    # --------------------------------------------------------------- #
    # A. fake sanity
    # --------------------------------------------------------------- #
    # A1: one source per FOV (wrapping), FRAP file is a copy of the
    #     survey source
    fake = FakeNIS([src1, src2])
    results, _ = run_outer(os.path.join(TMP, 'a1'), 'wrap', fake, POS4)
    src_of = lambda r: src1 if r[0] % 2 == 1 else src2
    check('A1 source wrapping per FOV + FRAP copy',
          len(results) == 4 and all(r[4] is not None for r in results)
          and all(filecmp.cmp(r[4][0][2], src_of(r), shallow=False)
                  for r in results)
          and all(filecmp.cmp(r[4][0][3], src_of(r), shallow=False)
                  for r in results))

    # A2: cross-cycle cell matching: 2 cells found -> cycle 1 stimulates
    #     cell 1, cycle 2 matches cell 1 and stimulates cell 2, cycle 3
    #     finds nothing new and stops; per-cycle cleanup leaves no state
    fake = FakeNIS([src1])
    results, _ = run_outer(os.path.join(TMP, 'a2'), 'cyc', fake, POS2[:1],
                           max_cycles=3)
    fov = results[0][4]
    check('A2 cross-cycle matching: cells 1 then 2, then stop',
          len(fov) == 2 and [r[1] for r in fov] == [1, 2],
          f'cells={[r[1] for r in fov]}')
    check('A2 cleanup: no docs open, ROIs cleared (ROI batch + per-cycle)',
          fake.open_docs == [] and fake.current == ''
          and len(fake.calls_of('delete_all_rois_in_current_document')) == 7,
          f"open_docs={fake.open_docs} delete_all_rois="
          f"{len(fake.calls_of('delete_all_rois_in_current_document'))}")

    # A3: fov_subdirs layout: each FOV's files in its own sub-directory
    fake = FakeNIS([src1, src2])
    results, _ = run_outer(os.path.join(TMP, 'a3'), 'sub', fake, POS2,
                           fov_subdirs=True)
    check('A3 fov_subdirs layout',
          all(r[4] is not None for r in results)
          and all(os.path.dirname(r[4][0][2]) == r[3] for r in results)
          and 'fov01' in results[0][3] and 'fov02' in results[1][3]
          and filecmp.cmp(results[0][4][0][2], src1, shallow=False)
          and filecmp.cmp(results[1][4][0][2], src2, shallow=False))

    # --------------------------------------------------------------- #
    # B. pre-flight -> AbortRunError
    # --------------------------------------------------------------- #
    # B1: invalid experiment name (checked before any NIS interaction)
    try:
        af.autofrap_loop_outer('fake', os.path.join(TMP, 'b1'), POS2,
                               max_cycles=1, detection_fun=detection_fun,
                               name='bad name!', use_timestamp=False)
        check('B1 invalid name -> AbortRunError', False)
    except af.AbortRunError as e:
        check('B1 invalid name -> AbortRunError',
              'invalid experiment name' in str(e))

    # B2: run directory already exists and is non-empty
    out_b2 = os.path.join(TMP, 'b2', 'x')
    os.makedirs(out_b2, exist_ok=True)
    with open(os.path.join(out_b2, 'leftover.nd2'), 'wb') as f:
        f.write(b'x')
    try:
        af.autofrap_loop_outer('fake', os.path.join(TMP, 'b2'), POS2,
                               max_cycles=1, detection_fun=detection_fun,
                               name='x', use_timestamp=False)
        check('B2 non-empty run dir -> AbortRunError', False)
    except af.AbortRunError as e:
        check('B2 non-empty run dir -> AbortRunError',
              'already exists' in str(e))

    # B3: setup read fails (all retries) -> AbortRunError, not a raw
    #     traceback; the run never starts
    fake = FakeNIS([src1],
                   failures={'get_position':
                             TimeoutError('NIS not responding')})
    buf = io.StringIO()
    raised = None
    with contextlib.redirect_stdout(buf):
        with fake:
            try:
                af.autofrap('fake', os.path.join(TMP, 'b3'), nx=1, ny=1,
                            max_cycles=1, detection_fun=detection_fun)
            except af.AbortRunError as e:
                raised = e
    check('B3 setup failure -> AbortRunError',
          raised is not None and 'microscope setup failed' in str(raised),
          f'{raised!r}' if raised else 'no exception raised')

    # --------------------------------------------------------------- #
    # C. FOV-level failures: consecutive-failure policy
    # --------------------------------------------------------------- #
    # C1: every survey times out -> systemic -> abort at FOV 3
    fake = FakeNIS([src1], failures={
        'run_current_nd_experiment': TimeoutError('macro timed out')})
    results, log = run_outer(os.path.join(TMP, 'c1'), 'tmo', fake, POS5)
    check('C1 all surveys time out -> abort at FOV 3',
          len(results) == 3 and all(r[4] is None for r in results)
          and 'Grid ABORTED at FOV 3 (3 consecutive failures)' in log,
          f'visited={len(results)}')

    # C2: custom limit (2) -> abort at FOV 2
    fake = FakeNIS([src1], failures={
        'run_current_nd_experiment': TimeoutError('macro timed out')})
    results, log = run_outer(os.path.join(TMP, 'c2'), 'lim', fake, POS5,
                             max_consecutive_failures=2)
    check('C2 limit 2 -> abort at FOV 2',
          len(results) == 2 and all(r[4] is None for r in results)
          and 'Grid ABORTED at FOV 2 (2 consecutive failures)' in log)

    # C3: one silent no-save (first survey) -> one failed FOV, run
    #     continues (exercises the pipeline's isfile trust-but-verify)
    fake = FakeNIS([src1],
                   failures={'run_current_nd_experiment': ['skip']})
    results, log = run_outer(os.path.join(TMP, 'c3'), 'nosave', fake, POS5)
    check('C3 one silent no-save -> FOV 1 fails, run continues',
          len(results) == 5 and results[0][4] is None
          and all(r[4] is not None for r in results[1:])
          and 'Grid done: 4/5' in log and 'survey file missing' in log)

    # C4: one failed FRAP save -> FOV 1 fails, run continues
    fake = FakeNIS([src1], failures={
        'save_current_document': [OSError('disk full')]})
    results, log = run_outer(os.path.join(TMP, 'c4'), 'save', fake, POS5)
    check('C4 one failed FRAP save -> FOV 1 fails, run continues',
          len(results) == 5 and results[0][4] is None
          and all(r[4] is not None for r in results[1:])
          and 'Grid done: 4/5' in log and 'disk full' in log)

    # C5: one failed stage move -> counts as a FOV failure, run continues
    fake = FakeNIS([src1],
                   failures={'set_position': [RuntimeError('stage jammed')]})
    results, log = run_outer(os.path.join(TMP, 'c5'), 'move1', fake, POS5)
    check('C5 one failed stage move -> FOV 1 fails, run continues',
          len(results) == 5 and results[0][4] is None
          and all(r[4] is not None for r in results[1:])
          and 'Grid done: 4/5' in log and 'stage jammed' in log)

    # C6: every stage move fails -> systemic -> abort at FOV 3
    fake = FakeNIS([src1],
                   failures={'set_position': RuntimeError('stage jammed')})
    results, log = run_outer(os.path.join(TMP, 'c6'), 'move2', fake, POS5)
    check('C6 every stage move fails -> abort at FOV 3',
          len(results) == 3 and all(r[4] is None for r in results)
          and 'Grid ABORTED at FOV 3 (3 consecutive failures)' in log)

    # C7: ROI creation returns id -1 -> every FOV fails -> abort at FOV 3
    fake = FakeNIS([src1], roi_id=-1)
    results, log = run_outer(os.path.join(TMP, 'c7'), 'roi', fake, POS5)
    check('C7 ROI id -1 -> abort at FOV 3',
          len(results) == 3 and 'cell ROI creation failed (id=-1)' in log
          and 'Grid ABORTED at FOV 3' in log)

    # C8: detector down on odd FOVs only -> failures interspersed with
    #     successes: the counter resets, the run completes
    def det_odd_down(file):
        if any(t in file for t in ('fov01', 'fov03', 'fov05')):
            raise RuntimeError('detector server down')
        return detection_fun(file)
    fake = FakeNIS([src1])
    results, log = run_outer(os.path.join(TMP, 'c8'), 'det', fake, POS5,
                             detection_fun=det_odd_down)
    check('C8 intermittent detector failures -> counter resets, run completes',
          len(results) == 5
          and sum(1 for r in results if r[4] is not None) == 2
          and 'Grid done: 2/5' in log
          and log.count('detector server down') == 3)

    # C9: detector returns the wrong shape -> every FOV fails -> abort
    fake = FakeNIS([src1])
    results, log = run_outer(os.path.join(TMP, 'c9'), 'shape', fake, POS5,
                             detection_fun=lambda f: 'not labels')
    check('C9 bad detector output -> abort at FOV 3',
          len(results) == 3 and 'detection_fun returned str' in log)

    # --------------------------------------------------------------- #
    # D. clean stop: summary printed, exception re-raised (CLI exits 130)
    # --------------------------------------------------------------- #
    stop = {'v': False}

    def det_request_stop(file):
        stop['v'] = True  # request the stop once the first FOV is underway
        return detection_fun(file)

    fake = FakeNIS([src1])
    buf = io.StringIO()
    raised = None
    with contextlib.redirect_stdout(buf):
        with fake:
            try:
                af.autofrap_loop_outer('fake', os.path.join(TMP, 'd1'), POS5,
                                       max_cycles=1,
                                       detection_fun=det_request_stop,
                                       stop_check=lambda: stop['v'],
                                       name='stop', use_timestamp=False)
            except af.AutofrapInterruptedException:
                raised = True
    log = buf.getvalue()
    check('D1 clean stop: re-raised after the summary',
          raised is True and 'Grid stopped by user: 1/5' in log)

    # --------------------------------------------------------------- #
    # E. detector output contract: accepted shapes and rejections
    # --------------------------------------------------------------- #
    labels64 = np.zeros((64, 64), dtype=np.int32)
    yy, xx = np.ogrid[:64, :64]
    labels64[(yy - 32) ** 2 + (xx - 32) ** 2 <= 64] = 1  # one circle, cell 1
    mask64 = labels64 > 0
    viz64 = labels64.astype(np.float32)

    for n, (name, det) in enumerate([
            ('bare label map (ndarray)', lambda f: labels64),
            ('1-tuple (labels,)', lambda f: (labels64,)),
            ('2-tuple (labels, mask)', lambda f: (labels64, mask64)),
            ('3-tuple (labels, mask, viz)', lambda f: (labels64, mask64, viz64)),
            ('1-list [labels]', lambda f: [labels64]),
    ], 1):
        fake = FakeNIS([src1])
        results, _ = run_outer(os.path.join(TMP, f'e{n}'), 'shape', fake,
                               POS2[:1], detection_fun=det)
        check(f'E{n} {name} accepted',
              results[0][4] is not None and len(results[0][4]) == 1)

    # 4-tuple: wrong arity (the str case is covered by C9)
    fake = FakeNIS([src1])
    results, log = run_outer(os.path.join(TMP, 'e6'), 'shape4', fake, POS2[:1],
                             detection_fun=lambda f: (labels64, mask64, viz64, None))
    check('E6 4-tuple rejected',
          results[0][4] is None and 'detection_fun returned tuple' in log)

# A4: no patch leaked outside any of the contexts above
leaked = [n for n in originals if getattr(nis_util, n) is not originals[n]]
check('A4 no patch leaked after context exit', not leaked, f'leaked={leaked}')

print(f'\n{n_failures} failure(s)')
sys.exit(1 if n_failures else 0)
