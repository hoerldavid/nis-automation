"""
offline smoke test for autofrap.fake_nis (no microscope, no server):
drives autofrap() and autofrap_grid() against FakeNIS with the dummy
detector and checks the outputs — per-FOV survey copies, FRAP files,
cross-cycle matching, failure flags, and that no patch leaks.

run: python autofrap/autofrap_bitsnpieces/test_fake_nis.py
"""
import filecmp
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import nis_util
import autofrap
from autofrap import NonRecoverableError
from autofrap.detection import load_detector_file
from autofrap import fake_nis
from autofrap.fake_nis import FakeNIS

detection_fun = load_detector_file(os.path.join(
    ROOT, 'autofrap', 'detectors', 'dummy_detector.py'))

failures = 0


def check(name, cond, extra=''):
    global failures
    print(f'{"ok  " if cond else "FAIL"} {name}'
          + (f': {extra}' if extra else ''))
    if not cond:
        failures += 1


# remember the originals for the leak check at the end
originals = {name: getattr(nis_util, name)
             for name in fake_nis.PATCHED_FUNCTIONS}

with tempfile.TemporaryDirectory() as tmp:
    # two fake "survey nd2" sources (opaque contents: the dummy
    # detector never reads the survey file)
    src1 = os.path.join(tmp, 'src1.nd2')
    src2 = os.path.join(tmp, 'src2.nd2')
    for s, tag in ((src1, 1), (src2, 2)):
        with open(s, 'wb') as f:
            f.write(f'fake-nd2-{tag}'.encode())

    # 1. single FOV, 3 cycles: same source for every cycle; the dummy
    #    detector finds 2 objects -> cycle 1 stimulates cell 1, cycle 2
    #    matches cell 1 (centroid) and stimulates cell 2, cycle 3 finds
    #    nothing new and stops
    out1 = os.path.join(tmp, 'run1')
    with FakeNIS([src1]) as fake:
        res = autofrap.autofrap('fake', out1, max_cycles=3,
                                detection_fun=detection_fun)
    ok = (len(res) == 2 and [r[1] for r in res] == [1, 2]
          and all(filecmp.cmp(r[2], src1, shallow=False) for r in res)
          and all(os.path.isfile(r[3]) for r in res))
    check('single fov, 3 cycles: cells 1 then 2, then stop',
          ok, f'results={[(r[0], r[1]) for r in res]}')
    check('surveys are copies of the (single) source',
          all(filecmp.cmp(r[2], src1, shallow=False) for r in res))
    # cleanup ran: both ROIs deleted, no documents left open
    check('cleanup: ROIs deleted, doc state clean',
          len(fake.calls_of('delete_roi')) == 4
          and fake.open_docs == [] and fake.current == 'Frozen',
          f'delete_roi={fake.calls_of("delete_roi")} open_docs={fake.open_docs}')

    # 2. grid 2x2 with 2 sources: one file per FOV, wrapping:
    #    fov01 -> src1, fov02 -> src2, fov03 -> src1, fov04 -> src2
    out2 = os.path.join(tmp, 'run2')
    with FakeNIS([src1, src2]) as fake:
        results = autofrap.autofrap_grid('fake', out2, nx=2, ny=2,
                                         max_cycles=1,
                                         detection_fun=detection_fun,
                                         name='dry', use_timestamp=False)
    run_dir = os.path.join(out2, 'dry')
    survey_of = lambda i, r: r[4][0][2]
    ok = (len(results) == 4
          and all(r[4] is not None for r in results)
          and all(os.path.dirname(survey_of(0, r)) == run_dir
                  for r in results)
          and all(filecmp.cmp(survey_of(0, r),
                              src1 if r[0] % 2 == 1 else src2,
                              shallow=False)
                  for r in results)
          and all(os.path.isfile(r[4][0][3]) for r in results))
    check('grid 2x2, one source per FOV (wrapping)', ok,
          f'run_dir={run_dir}')

    # 3. fov_subdirs layout: the fov tag is still in the file basename
    out3 = os.path.join(tmp, 'run3')
    with FakeNIS([src1, src2]):
        results = autofrap.autofrap_grid('fake', out3, nx=1, ny=2,
                                         max_cycles=1, fov_subdirs=True,
                                         detection_fun=detection_fun,
                                         name='sub', use_timestamp=False)
    ok = (all(r[4] is not None for r in results)
          and all(filecmp.cmp(r[4][0][2],
                              src1 if r[0] == 1 else src2, shallow=False)
                  for r in results))
    check('grid with fov_subdirs: per-FOV source mapping still works', ok)

    # 4. failure flags propagate as the pipeline expects
    out4 = os.path.join(tmp, 'run4')
    with FakeNIS([src1], fail_survey=True):
        try:
            autofrap.autofrap('fake', out4, max_cycles=1,
                              detection_fun=detection_fun)
            check('fail_survey -> NonRecoverableError', False)
        except NonRecoverableError as e:
            check('fail_survey -> NonRecoverableError', True, str(e)[:70])

    out5 = os.path.join(tmp, 'run5')
    with FakeNIS([src1], frap_out='touch', fail_save=True):
        try:
            autofrap.autofrap('fake', out5, max_cycles=1,
                              detection_fun=detection_fun)
            check('fail_save -> NonRecoverableError', False)
        except NonRecoverableError as e:
            check('fail_save -> NonRecoverableError', True, str(e)[:70])

    # stage move failure: the grid handles it internally (abort at FOV 1,
    # no exception), and still returns to start (2nd move succeeds —
    # fail_move is one-shot)
    import contextlib, io
    out6 = os.path.join(tmp, 'run6')
    with FakeNIS([src1], fail_move=True):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            results = autofrap.autofrap_grid('fake', out6, nx=1, ny=1,
                                             max_cycles=1,
                                             detection_fun=detection_fun,
                                             name='mv', use_timestamp=False)
        log = out.getvalue()
    ok = (len(results) == 1 and results[0][4] is None
          and 'ABORTED at FOV 1' in log and 'moved back to start' in log)
    check('fail_move -> grid aborts at FOV 1, returns to start', ok)

    # 5. frap_out='copy': the FRAP file is a copy of the survey source
    out8 = os.path.join(tmp, 'run8')
    with FakeNIS([src1], frap_out='copy'):
        res8 = autofrap.autofrap('fake', out8, max_cycles=1,
                                 detection_fun=detection_fun)
    check("frap_out='copy': FRAP file is a source copy",
          filecmp.cmp(res8[0][3], src1, shallow=False))

# 6. no patch leaked outside the context manager
leaked = [n for n in originals
          if getattr(nis_util, n) is not originals[n]]
check('no patch leaked after context exit', not leaked, f'leaked={leaked}')

print(f'\n{failures} failure(s)')
sys.exit(1 if failures else 0)
