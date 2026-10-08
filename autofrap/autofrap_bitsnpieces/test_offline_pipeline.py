"""
Offline assertion suite for the autoFRAP pipeline against FakeNIS
(no microscope, no detector server - a dummy detector stub is used).

The pipeline logs via the `autofrap.*` loggers (INFO milestones by
default); this suite captures the log records and asserts on those plus
on-disk artifacts and FakeNIS state. The pipeline no longer returns
per-FOV results - the log is the progress record.

Sections:
  A. fake sanity      - per-FOV source wrapping, FRAP copies, cross-cycle
                        cell matching, per-cycle cleanup, no patch leak
  B. pre-flight       - AbortRunError: invalid name, non-empty run dir,
                        failed setup read
  C. FOV-level policy - consecutive-failure policy: abort at the limit,
                        counter reset on success, one-shot vs systemic
                        failures, stage-move failures, detection failures
  D. clean stop       - summary logged, AutofrapInterruptedException
                        re-raised (CLI exit 130)
  E. detector contract - detection_fun return shapes: accepted forms
                        and wrong-arity rejection

run: python autofrap/autofrap_bitsnpieces/test_offline_pipeline.py
"""
import filecmp
import logging
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from autofrap.pipeline import autofrap as af
from autofrap.microscope import fake_nis  # noqa: F401  (imported for parity)
from autofrap.microscope.fake_nis import FakeNIS, PATCHED_FUNCTIONS
from autofrap.core.image.segmentation import dummy_detect_objects
from autofrap.core.image.mask import half_object_stim_mask
import autofrap.microscope.nis as nis_util

# capture everything the autofrap.* loggers emit (the pipeline logs INFO
# milestones; retry progress comes from autofrap.core.utils.retry)
AUTOPRAP_LOGGER = logging.getLogger('autofrap')
AUTOPRAP_LOGGER.setLevel(logging.DEBUG)


class ListHandler(logging.Handler):
    """collect (levelname, message) records for assertions"""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def msgs(records, level=None):
    """messages of the captured records, optionally filtered by level"""
    return [m for lv, m in records if level is None or lv == level]


def has_msg(records, level, *substrings):
    """True if some record at `level` contains all substrings"""
    return any(all(s in m for s in substrings) for m in msgs(records, level))


def fov_headers(records):
    """the per-FOV header messages ('=== [i/N] ...')"""
    return [m for m in msgs(records, 'INFO') if m.startswith('=== [')]


def detection_fun(survey_file):
    """Detection stub for pipeline tests: the fake survey sources are
    opaque bytes, so the detector must not read the survey file — a
    fixed canvas with two objects (circle + rectangle) and a left-half
    mask. (Detector *files* are covered by test_load_detector_file.)"""
    image = np.zeros((512, 512), dtype=np.uint16)
    labels = dummy_detect_objects(image)
    return labels, half_object_stim_mask(labels)

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
        """autofrap_loop_outer inside FakeNIS with log capture.

        Returns (records, raised): the captured [(level, message), ...]
        records and the exception the run ended with (None on success).
        """
        handler = ListHandler()
        AUTOPRAP_LOGGER.addHandler(handler)
        raised = None
        try:
            with fake:
                af.autofrap_loop_outer(
                    'fake', out, positions, max_cycles=max_cycles,
                    detection_fun=detection_fun,
                    name=name, use_timestamp=False, **outer_kwargs)
        except Exception as e:
            raised = e
        finally:
            AUTOPRAP_LOGGER.removeHandler(handler)
        return handler.records, raised

    def fov_file(out, name, fov, kind, cycle=1, subdir=False):
        """path of a per-cycle file: <out>/<name>[/fovNN]/fovNN_cycleNN_<kind>.nd2"""
        d = os.path.join(out, name, f'fov{fov:02d}') if subdir \
            else os.path.join(out, name)
        return os.path.join(d, f'fov{fov:02d}_cycle{cycle:02d}_{kind}.nd2')

    # --------------------------------------------------------------- #
    # A. fake sanity
    # --------------------------------------------------------------- #
    # A1: one source per FOV (wrapping), FRAP file is a copy of the
    #     survey source
    out = os.path.join(TMP, 'a1')
    fake = FakeNIS([src1, src2])
    recs, raised = run_outer(out, 'wrap', fake, POS4)
    src_of = lambda i: src1 if i % 2 == 1 else src2
    check('A1 source wrapping per FOV + FRAP copy',
          raised is None
          and len(fov_headers(recs)) == 4
          and all(os.path.isfile(fov_file(out, 'wrap', i, k))
                  for i in (1, 2, 3, 4) for k in ('survey', 'frap'))
          and all(filecmp.cmp(fov_file(out, 'wrap', i, k), src_of(i), shallow=False)
                  for i in (1, 2, 3, 4) for k in ('survey', 'frap'))
          and has_msg(recs, 'INFO', 'Run done'),
          f'raised={raised!r}')

    # A2: cross-cycle cell matching: 2 cells found -> cycle 1 stimulates
    #     cell 1, cycle 2 matches cell 1 and stimulates cell 2, cycle 3
    #     finds nothing new and stops; per-cycle cleanup leaves no state
    out = os.path.join(TMP, 'a2')
    fake = FakeNIS([src1])
    recs, raised = run_outer(out, 'cyc', fake, POS2[:1], max_cycles=3)
    check('A2 cross-cycle matching: cells 1 then 2, then stop',
          raised is None
          and has_msg(recs, 'INFO', 'stimulating cell 1')
          and has_msg(recs, 'INFO', 'stimulating cell 2')
          and has_msg(recs, 'INFO', 'all stimulated or no stimulation mask')
          and has_msg(recs, 'INFO', 'FOV done after 3 cycle(s)')
          and os.path.isfile(fov_file(out, 'cyc', 1, 'frap', cycle=2))
          and not os.path.exists(fov_file(out, 'cyc', 1, 'frap', cycle=3)),
          f'raised={raised!r}')
    check('A2 cleanup: no docs open, no current document',
          fake.open_docs == [] and fake.current == '',
          f'open_docs={fake.open_docs} current={fake.current!r}')

    # A3: fov_subdirs layout: each FOV's files in its own sub-directory
    out = os.path.join(TMP, 'a3')
    fake = FakeNIS([src1, src2])
    recs, raised = run_outer(out, 'sub', fake, POS2, fov_subdirs=True)
    check('A3 fov_subdirs layout',
          raised is None
          and all(os.path.isfile(fov_file(out, 'sub', i, k, subdir=True))
                  for i in (1, 2) for k in ('survey', 'frap'))
          and filecmp.cmp(fov_file(out, 'sub', 1, 'survey', subdir=True),
                          src1, shallow=False)
          and filecmp.cmp(fov_file(out, 'sub', 2, 'survey', subdir=True),
                          src2, shallow=False)
          and not os.path.exists(fov_file(out, 'sub', 1, 'survey')),
          f'raised={raised!r}')

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
    raised = None
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
    out = os.path.join(TMP, 'c1')
    fake = FakeNIS([src1], failures={
        'run_current_nd_experiment': TimeoutError('macro timed out')})
    recs, raised = run_outer(out, 'tmo', fake, POS5)
    check('C1 all surveys time out -> abort at FOV 3',
          isinstance(raised, af.AbortRunError)
          and len(fov_headers(recs)) == 3
          and len(msgs(recs, 'WARNING')) == 2
          and has_msg(recs, 'ERROR', 'consecutive failure(s), aborting the run')
          and not os.path.exists(fov_file(out, 'tmo', 1, 'survey')),
          f'raised={raised!r}')

    # C2: custom limit (2) -> abort at FOV 2
    out = os.path.join(TMP, 'c2')
    fake = FakeNIS([src1], failures={
        'run_current_nd_experiment': TimeoutError('macro timed out')})
    recs, raised = run_outer(out, 'lim', fake, POS5,
                             max_consecutive_failures=2)
    check('C2 limit 2 -> abort at FOV 2',
          isinstance(raised, af.AbortRunError)
          and len(fov_headers(recs)) == 2
          and len(msgs(recs, 'WARNING')) == 1
          and has_msg(recs, 'ERROR', '2 consecutive failure(s), aborting the run'),
          f'raised={raised!r}')

    # C3: one silent no-save (first survey) -> one failed FOV, run
    #     continues (exercises the pipeline's isfile trust-but-verify)
    out = os.path.join(TMP, 'c3')
    fake = FakeNIS([src1],
                   failures={'run_current_nd_experiment': ['skip']})
    recs, raised = run_outer(out, 'nosave', fake, POS5)
    check('C3 one silent no-save -> FOV 1 fails, run continues',
          raised is None
          and len(fov_headers(recs)) == 5
          and len(msgs(recs, 'WARNING')) == 1
          and has_msg(recs, 'WARNING', 'survey file missing')
          and not os.path.exists(fov_file(out, 'nosave', 1, 'survey'))
          and all(os.path.isfile(fov_file(out, 'nosave', i, 'frap'))
                  for i in (2, 3, 4, 5))
          and has_msg(recs, 'INFO', 'Run done'),
          f'raised={raised!r}')

    # C4: one failed FRAP save -> FOV 1 fails, run continues
    out = os.path.join(TMP, 'c4')
    fake = FakeNIS([src1], failures={
        'save_current_document': [OSError('disk full')]})
    recs, raised = run_outer(out, 'save', fake, POS5)
    check('C4 one failed FRAP save -> FOV 1 fails, run continues',
          raised is None
          and len(msgs(recs, 'WARNING')) == 1
          and has_msg(recs, 'WARNING', 'disk full')
          and all(os.path.isfile(fov_file(out, 'save', i, 'survey'))
                  for i in (1, 2, 3, 4, 5))
          and not os.path.exists(fov_file(out, 'save', 1, 'frap'))
          and all(os.path.isfile(fov_file(out, 'save', i, 'frap'))
                  for i in (2, 3, 4, 5)),
          f'raised={raised!r}')

    # C5: one failed stage move -> retried by move_stage_with_retry,
    #     the FOV succeeds and the run completes
    out = os.path.join(TMP, 'c5')
    fake = FakeNIS([src1],
                   failures={'set_position': [RuntimeError('stage jammed')]})
    recs, raised = run_outer(out, 'move1', fake, POS5)
    check('C5 one failed stage move -> retried, run completes',
          raised is None
          and has_msg(recs, 'INFO', 'stage jammed', 'retry')
          and all(os.path.isfile(fov_file(out, 'move1', i, 'frap'))
                  for i in (1, 2, 3, 4, 5))
          and has_msg(recs, 'INFO', 'Run done'),
          f'raised={raised!r}')

    # C6: every stage move fails -> systemic -> abort at FOV 3
    out = os.path.join(TMP, 'c6')
    fake = FakeNIS([src1], failures={'set_position': RuntimeError('stage jammed')})
    recs, raised = run_outer(out, 'move2', fake, POS5)
    check('C6 every stage move fails -> abort at FOV 3',
          isinstance(raised, af.AbortRunError)
          and len(fov_headers(recs)) == 3
          and len(msgs(recs, 'WARNING')) == 2
          and has_msg(recs, 'ERROR', 'aborting the run'),
          f'raised={raised!r}')

    # C7: ROI creation returns id -1 -> every FOV fails -> abort at FOV 3
    out = os.path.join(TMP, 'c7')
    fake = FakeNIS([src1], roi_id=-1)
    recs, raised = run_outer(out, 'roi', fake, POS5)
    check('C7 ROI id -1 -> abort at FOV 3',
          isinstance(raised, af.AbortRunError)
          and len(fov_headers(recs)) == 3
          and len([m for m in msgs(recs, 'WARNING') + msgs(recs, 'ERROR')
                   if 'cell ROI creation failed (id=-1)' in m]) == 3,
          f'raised={raised!r}')

    # C8: detector down on odd FOVs only -> failures interspersed with
    #     successes: the counter resets, the run completes
    def det_odd_down(file):
        if any(t in file for t in ('fov01', 'fov03', 'fov05')):
            raise RuntimeError('detector server down')
        return detection_fun(file)
    out = os.path.join(TMP, 'c8')
    fake = FakeNIS([src1])
    recs, raised = run_outer(out, 'det', fake, POS5,
                             detection_fun=det_odd_down)
    check('C8 intermittent detector failures -> counter resets, run completes',
          raised is None
          and len(msgs(recs, 'WARNING')) == 3
          and all('detector server down' in m for m in msgs(recs, 'WARNING'))
          and has_msg(recs, 'INFO', 'Run done')
          and all(os.path.isfile(fov_file(out, 'det', i, 'frap'))
                  for i in (2, 4))
          and not any(os.path.exists(fov_file(out, 'det', i, 'frap'))
                      for i in (1, 3, 5)),
          f'raised={raised!r}')

    # C9: detector returns the wrong shape -> every FOV fails -> abort
    out = os.path.join(TMP, 'c9')
    fake = FakeNIS([src1])
    recs, raised = run_outer(out, 'shape', fake, POS5,
                             detection_fun=lambda f: 'not labels')
    check('C9 bad detector output -> abort at FOV 3',
          isinstance(raised, af.AbortRunError)
          and len(fov_headers(recs)) == 3
          and len([m for m in msgs(recs, 'WARNING') + msgs(recs, 'ERROR')
                   if 'detection_fun returned str' in m]) == 3,
          f'raised={raised!r}')

    # --------------------------------------------------------------- #
    # D. clean stop: summary logged, exception re-raised (CLI exits 130)
    # --------------------------------------------------------------- #
    stop = {'v': False}

    def det_request_stop(file):
        stop['v'] = True  # request the stop once the first FOV is underway
        return detection_fun(file)

    out = os.path.join(TMP, 'd1')
    fake = FakeNIS([src1])
    recs, raised = run_outer(out, 'stop', fake, POS5,
                             detection_fun=det_request_stop,
                             stop_check=lambda: stop['v'])
    check('D1 clean stop: re-raised after the summary',
          isinstance(raised, af.AutofrapInterruptedException)
          and has_msg(recs, 'INFO', 'Run stopped by user: 1/5')
          and os.path.isfile(fov_file(out, 'stop', 1, 'frap')),
          f'raised={raised!r}')

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
        out = os.path.join(TMP, f'e{n}')
        fake = FakeNIS([src1])
        recs, raised = run_outer(out, 'shape', fake, POS2[:1],
                                 detection_fun=det)
        check(f'E{n} {name} accepted',
              raised is None
              and os.path.isfile(fov_file(out, 'shape', 1, 'frap')),
              f'raised={raised!r}')

    # 4-tuple: wrong arity (the str case is covered by C9)
    out = os.path.join(TMP, 'e6')
    fake = FakeNIS([src1])
    recs, raised = run_outer(out, 'shape4', fake, POS2[:1],
                             detection_fun=lambda f: (labels64, mask64, viz64, None))
    check('E6 4-tuple rejected',
          raised is None
          and has_msg(recs, 'WARNING', 'detection_fun returned tuple')
          and not os.path.exists(fov_file(out, 'shape4', 1, 'frap')),
          f'raised={raised!r}')

# A4: no patch leaked outside any of the contexts above
leaked = [n for n in originals if getattr(nis_util, n) is not originals[n]]
check('A4 no patch leaked after context exit', not leaked, f'leaked={leaked}')

# A5: the macro debug dir context was restored after each run
check('A5 macro_debug_dir restored after the runs',
      nis_util._macro_debug_dir is None,
      f'_macro_debug_dir={nis_util._macro_debug_dir!r}')

print(f'\n{n_failures} failure(s)')
sys.exit(1 if n_failures else 0)
