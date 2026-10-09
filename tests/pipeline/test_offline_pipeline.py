"""
Offline test suite for the autoFRAP pipeline against FakeNIS
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

Run from the repo root:  python -m unittest tests.pipeline.test_offline_pipeline -v
(or the whole suite:      python -m unittest discover -s tests -t .)
"""
import filecmp
import logging
import os
import tempfile
import unittest

import numpy as np

from autofrap.pipeline import autofrap as af
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
    mask. (Detector *files* are covered by tests.core.test_detection.)"""
    image = np.zeros((512, 512), dtype=np.uint16)
    labels = dummy_detect_objects(image)
    return labels, half_object_stim_mask(labels)


# remember the originals for the leak check (A4)
ORIGINALS = {name: getattr(nis_util, name) for name in PATCHED_FUNCTIONS}

POS5 = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]
POS4 = POS5[:4]
POS2 = POS5[:2]


def circle_labels():
    """one circle in the middle of a 64x64 canvas (cell 1) + mask + viz"""
    labels = np.zeros((64, 64), dtype=np.int32)
    yy, xx = np.ogrid[:64, :64]
    labels[(yy - 32) ** 2 + (xx - 32) ** 2 <= 64] = 1
    return labels, labels > 0, labels.astype(np.float32)


class TestOfflinePipeline(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        # two fake "survey nd2" sources (opaque contents: the dummy
        # detector never reads the survey file)
        self.src1 = os.path.join(self.tmp, 'src1.nd2')
        self.src2 = os.path.join(self.tmp, 'src2.nd2')
        for s, tag in ((self.src1, 1), (self.src2, 2)):
            with open(s, 'wb') as f:
                f.write(f'fake-nd2-{tag}'.encode())

    # -------------------------------------------------------------- #
    # helpers                                                         #
    # -------------------------------------------------------------- #

    def run_outer(self, out, name, fake, positions, max_cycles=1,
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

    def fov_file(self, out, name, fov, kind, cycle=1, subdir=False):
        """path of a per-cycle file: <out>/<name>[/fovNN]/fovNN_cycleNN_<kind>.nd2"""
        d = os.path.join(out, name, f'fov{fov:02d}') if subdir \
            else os.path.join(out, name)
        return os.path.join(d, f'fov{fov:02d}_cycle{cycle:02d}_{kind}.nd2')

    # -------------------------------------------------------------- #
    # A. fake sanity                                                  #
    # -------------------------------------------------------------- #

    def test_A1_source_wrapping_and_frap_copy(self):
        """A1: one source per FOV (wrapping), FRAP file is a copy of the
        survey source"""
        out = os.path.join(self.tmp, 'a1')
        fake = FakeNIS([self.src1, self.src2])
        recs, raised = self.run_outer(out, 'wrap', fake, POS4)
        src_of = lambda i: self.src1 if i % 2 == 1 else self.src2
        self.assertIsNone(raised)
        self.assertEqual(len(fov_headers(recs)), 4)
        for i in (1, 2, 3, 4):
            for k in ('survey', 'frap'):
                with self.subTest(fov=i, kind=k):
                    path = self.fov_file(out, 'wrap', i, k)
                    self.assertTrue(os.path.isfile(path))
                    self.assertTrue(filecmp.cmp(path, src_of(i), shallow=False))
        self.assertTrue(has_msg(recs, 'INFO', 'Run done'))

    def test_A2_cross_cycle_matching_and_cleanup(self):
        """A2: cross-cycle cell matching: 2 cells found -> cycle 1
        stimulates cell 1, cycle 2 matches cell 1 and stimulates cell 2,
        cycle 3 finds nothing new and stops; per-cycle cleanup leaves no
        state"""
        out = os.path.join(self.tmp, 'a2')
        fake = FakeNIS([self.src1])
        recs, raised = self.run_outer(out, 'cyc', fake, POS2[:1], max_cycles=3)
        self.assertIsNone(raised)
        self.assertTrue(has_msg(recs, 'INFO', 'stimulating cell 1'))
        self.assertTrue(has_msg(recs, 'INFO', 'stimulating cell 2'))
        self.assertTrue(has_msg(recs, 'INFO',
                                'all stimulated or no stimulation mask'))
        self.assertTrue(has_msg(recs, 'INFO', 'FOV done after 3 cycle(s)'))
        self.assertTrue(os.path.isfile(
            self.fov_file(out, 'cyc', 1, 'frap', cycle=2)))
        self.assertFalse(os.path.exists(
            self.fov_file(out, 'cyc', 1, 'frap', cycle=3)))
        # cleanup: no docs open, no current document
        self.assertEqual(fake.open_docs, [])
        self.assertEqual(fake.current, '')

    def test_A3_fov_subdirs_layout(self):
        """A3: fov_subdirs layout: each FOV's files in its own sub-directory"""
        out = os.path.join(self.tmp, 'a3')
        fake = FakeNIS([self.src1, self.src2])
        recs, raised = self.run_outer(out, 'sub', fake, POS2, fov_subdirs=True)
        self.assertIsNone(raised)
        for i in (1, 2):
            for k in ('survey', 'frap'):
                with self.subTest(fov=i, kind=k):
                    self.assertTrue(os.path.isfile(
                        self.fov_file(out, 'sub', i, k, subdir=True)))
        self.assertTrue(filecmp.cmp(
            self.fov_file(out, 'sub', 1, 'survey', subdir=True),
            self.src1, shallow=False))
        self.assertTrue(filecmp.cmp(
            self.fov_file(out, 'sub', 2, 'survey', subdir=True),
            self.src2, shallow=False))
        self.assertFalse(os.path.exists(self.fov_file(out, 'sub', 1, 'survey')))

    def test_A4_no_patch_leaked(self):
        """A4: no FakeNIS patch leaked outside the contexts (valid whenever
        no fake context is active, i.e. after every other test's run)"""
        leaked = [n for n in ORIGINALS
                  if getattr(nis_util, n) is not ORIGINALS[n]]
        self.assertEqual(leaked, [])

    def test_A5_macro_debug_dir_restored(self):
        """A5: the macro debug dir context was restored after the runs"""
        self.assertIsNone(nis_util._macro_debug_dir)

    # -------------------------------------------------------------- #
    # B. pre-flight -> AbortRunError                                  #
    # -------------------------------------------------------------- #

    def test_B1_invalid_name_aborts(self):
        """B1: invalid experiment name (checked before any NIS interaction)"""
        with self.assertRaises(af.AbortRunError) as cm:
            af.autofrap_loop_outer('fake', os.path.join(self.tmp, 'b1'), POS2,
                                   max_cycles=1, detection_fun=detection_fun,
                                   name='bad name!', use_timestamp=False)
        self.assertIn('invalid experiment name', str(cm.exception))

    def test_B2_nonempty_run_dir_aborts(self):
        """B2: run directory already exists and is non-empty"""
        out_b2 = os.path.join(self.tmp, 'b2', 'x')
        os.makedirs(out_b2, exist_ok=True)
        with open(os.path.join(out_b2, 'leftover.nd2'), 'wb') as f:
            f.write(b'x')
        with self.assertRaises(af.AbortRunError) as cm:
            af.autofrap_loop_outer('fake', os.path.join(self.tmp, 'b2'), POS2,
                                   max_cycles=1, detection_fun=detection_fun,
                                   name='x', use_timestamp=False)
        self.assertIn('already exists', str(cm.exception))

    def test_B3_setup_failure_aborts(self):
        """B3: setup read fails (all retries) -> AbortRunError, not a raw
        traceback; the run never starts"""
        fake = FakeNIS([self.src1],
                       failures={'get_position':
                                 TimeoutError('NIS not responding')})
        raised = None
        with fake:
            try:
                af.autofrap('fake', os.path.join(self.tmp, 'b3'), nx=1, ny=1,
                            max_cycles=1, detection_fun=detection_fun)
            except af.AbortRunError as e:
                raised = e
        self.assertIsNotNone(raised, 'no exception raised')
        self.assertIn('microscope setup failed', str(raised))

    # -------------------------------------------------------------- #
    # C. FOV-level failures: consecutive-failure policy               #
    # -------------------------------------------------------------- #

    def test_C1_all_surveys_time_out(self):
        """C1: every survey times out -> systemic -> abort at FOV 3"""
        out = os.path.join(self.tmp, 'c1')
        fake = FakeNIS([self.src1], failures={
            'run_current_nd_experiment': TimeoutError('macro timed out')})
        recs, raised = self.run_outer(out, 'tmo', fake, POS5)
        self.assertIsInstance(raised, af.AbortRunError)
        self.assertEqual(len(fov_headers(recs)), 3)
        self.assertEqual(len(msgs(recs, 'WARNING')), 2)
        self.assertTrue(has_msg(recs, 'ERROR',
                                'consecutive failure(s), aborting the run'))
        self.assertFalse(os.path.exists(self.fov_file(out, 'tmo', 1, 'survey')))

    def test_C2_custom_limit_2(self):
        """C2: custom limit (2) -> abort at FOV 2"""
        out = os.path.join(self.tmp, 'c2')
        fake = FakeNIS([self.src1], failures={
            'run_current_nd_experiment': TimeoutError('macro timed out')})
        recs, raised = self.run_outer(out, 'lim', fake, POS5,
                                      max_consecutive_failures=2)
        self.assertIsInstance(raised, af.AbortRunError)
        self.assertEqual(len(fov_headers(recs)), 2)
        self.assertEqual(len(msgs(recs, 'WARNING')), 1)
        self.assertTrue(has_msg(recs, 'ERROR',
                                '2 consecutive failure(s), aborting the run'))

    def test_C3_one_silent_no_save(self):
        """C3: one silent no-save (first survey) -> one failed FOV, run
        continues (exercises the pipeline's isfile trust-but-verify)"""
        out = os.path.join(self.tmp, 'c3')
        fake = FakeNIS([self.src1],
                       failures={'run_current_nd_experiment': ['skip']})
        recs, raised = self.run_outer(out, 'nosave', fake, POS5)
        self.assertIsNone(raised)
        self.assertEqual(len(fov_headers(recs)), 5)
        self.assertEqual(len(msgs(recs, 'WARNING')), 1)
        self.assertTrue(has_msg(recs, 'WARNING', 'survey file missing'))
        self.assertFalse(os.path.exists(
            self.fov_file(out, 'nosave', 1, 'survey')))
        for i in (2, 3, 4, 5):
            self.assertTrue(os.path.isfile(self.fov_file(out, 'nosave', i, 'frap')))
        self.assertTrue(has_msg(recs, 'INFO', 'Run done'))

    def test_C4_one_failed_frap_save(self):
        """C4: one failed FRAP save -> FOV 1 fails, run continues"""
        out = os.path.join(self.tmp, 'c4')
        fake = FakeNIS([self.src1], failures={
            'save_current_document': [OSError('disk full')]})
        recs, raised = self.run_outer(out, 'save', fake, POS5)
        self.assertIsNone(raised)
        self.assertEqual(len(msgs(recs, 'WARNING')), 1)
        self.assertTrue(has_msg(recs, 'WARNING', 'disk full'))
        for i in (1, 2, 3, 4, 5):
            self.assertTrue(os.path.isfile(self.fov_file(out, 'save', i, 'survey')))
        self.assertFalse(os.path.exists(self.fov_file(out, 'save', 1, 'frap')))
        for i in (2, 3, 4, 5):
            self.assertTrue(os.path.isfile(self.fov_file(out, 'save', i, 'frap')))

    def test_C5_one_failed_stage_move(self):
        """C5: one failed stage move -> retried by move_stage_with_retry,
        the FOV succeeds and the run completes"""
        out = os.path.join(self.tmp, 'c5')
        fake = FakeNIS([self.src1],
                       failures={'set_position': [RuntimeError('stage jammed')]})
        recs, raised = self.run_outer(out, 'move1', fake, POS5)
        self.assertIsNone(raised)
        self.assertTrue(has_msg(recs, 'INFO', 'stage jammed', 'retry'))
        for i in (1, 2, 3, 4, 5):
            self.assertTrue(os.path.isfile(self.fov_file(out, 'move1', i, 'frap')))
        self.assertTrue(has_msg(recs, 'INFO', 'Run done'))

    def test_C6_every_stage_move_fails(self):
        """C6: every stage move fails -> systemic -> abort at FOV 3"""
        out = os.path.join(self.tmp, 'c6')
        fake = FakeNIS([self.src1], failures={'set_position': RuntimeError('stage jammed')})
        recs, raised = self.run_outer(out, 'move2', fake, POS5)
        self.assertIsInstance(raised, af.AbortRunError)
        self.assertEqual(len(fov_headers(recs)), 3)
        self.assertEqual(len(msgs(recs, 'WARNING')), 2)
        self.assertTrue(has_msg(recs, 'ERROR', 'aborting the run'))

    def test_C7_roi_id_minus_1(self):
        """C7: ROI creation returns id -1 -> every FOV fails -> abort at FOV 3"""
        out = os.path.join(self.tmp, 'c7')
        fake = FakeNIS([self.src1], roi_id=-1)
        recs, raised = self.run_outer(out, 'roi', fake, POS5)
        self.assertIsInstance(raised, af.AbortRunError)
        self.assertEqual(len(fov_headers(recs)), 3)
        n_roi_fail = len([m for m in msgs(recs, 'WARNING') + msgs(recs, 'ERROR')
                          if 'cell ROI creation failed (id=-1)' in m])
        self.assertEqual(n_roi_fail, 3)

    def test_C8_intermittent_detector_failures(self):
        """C8: detector down on odd FOVs only -> failures interspersed with
        successes: the counter resets, the run completes"""
        def det_odd_down(file):
            if any(t in file for t in ('fov01', 'fov03', 'fov05')):
                raise RuntimeError('detector server down')
            return detection_fun(file)

        out = os.path.join(self.tmp, 'c8')
        fake = FakeNIS([self.src1])
        recs, raised = self.run_outer(out, 'det', fake, POS5,
                                      detection_fun=det_odd_down)
        self.assertIsNone(raised)
        self.assertEqual(len(msgs(recs, 'WARNING')), 3)
        for m in msgs(recs, 'WARNING'):
            self.assertIn('detector server down', m)
        self.assertTrue(has_msg(recs, 'INFO', 'Run done'))
        for i in (2, 4):
            self.assertTrue(os.path.isfile(self.fov_file(out, 'det', i, 'frap')))
        for i in (1, 3, 5):
            self.assertFalse(os.path.exists(self.fov_file(out, 'det', i, 'frap')))

    def test_C9_bad_detector_output(self):
        """C9: detector returns the wrong shape -> every FOV fails -> abort"""
        out = os.path.join(self.tmp, 'c9')
        fake = FakeNIS([self.src1])
        recs, raised = self.run_outer(out, 'shape', fake, POS5,
                                      detection_fun=lambda f: 'not labels')
        self.assertIsInstance(raised, af.AbortRunError)
        self.assertEqual(len(fov_headers(recs)), 3)
        n_bad = len([m for m in msgs(recs, 'WARNING') + msgs(recs, 'ERROR')
                     if 'detection_fun returned str' in m])
        self.assertEqual(n_bad, 3)

    # -------------------------------------------------------------- #
    # D. clean stop: summary logged, exception re-raised (CLI exit 130)#
    # -------------------------------------------------------------- #

    def test_D1_clean_stop(self):
        """D1: clean stop: re-raised after the summary"""
        stop = {'v': False}

        def det_request_stop(file):
            stop['v'] = True  # request the stop once the first FOV is underway
            return detection_fun(file)

        out = os.path.join(self.tmp, 'd1')
        fake = FakeNIS([self.src1])
        recs, raised = self.run_outer(out, 'stop', fake, POS5,
                                      detection_fun=det_request_stop,
                                      stop_check=lambda: stop['v'])
        self.assertIsInstance(raised, af.AutofrapInterruptedException)
        self.assertTrue(has_msg(recs, 'INFO', 'Run stopped by user: 1/5'))
        self.assertTrue(os.path.isfile(self.fov_file(out, 'stop', 1, 'frap')))

    # -------------------------------------------------------------- #
    # E. detector output contract: accepted shapes and rejections     #
    # -------------------------------------------------------------- #

    def test_E_accepted_detection_output_forms(self):
        """E1-E5: detection_fun return shapes: accepted forms"""
        labels64, mask64, viz64 = circle_labels()
        for n, (name, det) in enumerate([
                ('bare label map (ndarray)', lambda f: labels64),
                ('1-tuple (labels,)', lambda f: (labels64,)),
                ('2-tuple (labels, mask)', lambda f: (labels64, mask64)),
                ('3-tuple (labels, mask, viz)', lambda f: (labels64, mask64, viz64)),
                ('1-list [labels]', lambda f: [labels64]),
        ], 1):
            with self.subTest(form=name):
                out = os.path.join(self.tmp, f'e{n}')
                fake = FakeNIS([self.src1])
                recs, raised = self.run_outer(out, 'shape', fake, POS2[:1],
                                              detection_fun=det)
                self.assertIsNone(raised)
                self.assertTrue(os.path.isfile(
                    self.fov_file(out, 'shape', 1, 'frap')))

    def test_E6_four_tuple_rejected(self):
        """E6: 4-tuple: wrong arity (the str case is covered by C9)"""
        labels64, mask64, viz64 = circle_labels()
        out = os.path.join(self.tmp, 'e6')
        fake = FakeNIS([self.src1])
        recs, raised = self.run_outer(
            out, 'shape4', fake, POS2[:1],
            detection_fun=lambda f: (labels64, mask64, viz64, None))
        self.assertIsNone(raised)
        self.assertTrue(has_msg(recs, 'WARNING', 'detection_fun returned tuple'))
        self.assertFalse(os.path.exists(self.fov_file(out, 'shape4', 1, 'frap')))


if __name__ == '__main__':
    unittest.main()
