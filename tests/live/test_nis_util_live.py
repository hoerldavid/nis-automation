"""
Live smoke test for the nis wrappers.

Run at the microscope with NIS-Elements open. Nothing is opened, saved,
or closed; the only state change is a small stage XY round-trip
(+/-2 um) plus an optional +/-1 um piezo round-trip when a piezo is
present. Covers:

  - every read-only get_* wrapper (temp-.mac -> nis_ar -mw -> temp-.ini
    round-trip through _run_macro)
  - get_fov_from_res (pure computation)
  - set_position round-trip (also exercises the get_position z1=None
    path when no piezo is present, and the set_position pos_piezo
    branch when one is)

Skipped automatically off the microscope workstation.
"""
import os
import time
import unittest

from autofrap.microscope import nis as nis_util

NIS = r'C:\Program Files\NIS-Elements\nis_ar.exe'
LIVE = os.name == 'nt' and os.path.exists(NIS)

MOVE_UM = 2.0      # relative XY move for the set_position round-trip
PIEZO_UM = 1.0     # relative piezo move (only when a piezo is present)
TOL_UM = 0.5       # position read-back tolerance
SETTLE_S = 1.0     # vibration settle after a move (the move itself blocks)


@unittest.skipUnless(LIVE, 'requires the microscope workstation (NIS-Elements)')
class TestNisWrappersLive(unittest.TestCase):

    def test_read_only_wrappers(self):
        """every read-only get_* wrapper runs without raising"""
        nis_util.get_camera_format(NIS)
        pos = nis_util.get_position(NIS)
        res = nis_util.get_resolution(NIS)
        nis_util.get_rotation_matrix(NIS)
        nis_util.get_cam_rotation(NIS)
        ocs = nis_util.get_optical_confs(NIS)
        nis_util.is_color_camera(NIS)
        nis_util.get_current_document(NIS)
        nis_util.get_roi_count(NIS)

        self.assertIsNotNone(pos)
        self.assertIsNotNone(res)
        self.assertIsNotNone(ocs)
        self.assertIn('FRAPPA', ocs)
        # pure computation on the resolution result
        fov = nis_util.get_fov_from_res(res)
        self.assertGreater(fov[0], 0)
        self.assertGreater(fov[1], 0)

    def test_stage_roundtrip(self):
        """set_position round-trip: move out and back, verify read-back"""
        pos = nis_util.get_position(NIS)
        self.assertIsNotNone(pos)
        x0, y0, z0, z1 = pos

        nis_util.set_position(NIS, pos_xy=(MOVE_UM, 0.0), relative_xy=True)
        time.sleep(SETTLE_S)
        p = nis_util.get_position(NIS)
        self.assertLessEqual(abs(p[0] - (x0 + MOVE_UM)), TOL_UM,
                             'move not as commanded')
        self.assertLessEqual(abs(p[1] - y0), TOL_UM,
                             'move not as commanded')

        nis_util.set_position(NIS, pos_xy=(-MOVE_UM, 0.0), relative_xy=True)
        time.sleep(SETTLE_S)
        p = nis_util.get_position(NIS)
        self.assertLessEqual(abs(p[0] - x0), TOL_UM,
                             'original position not restored')
        self.assertLessEqual(abs(p[1] - y0), TOL_UM,
                             'original position not restored')

        if z1 is not None:
            nis_util.set_position(NIS, pos_piezo=PIEZO_UM, relative_piezo=True)
            time.sleep(SETTLE_S)
            nis_util.set_position(NIS, pos_piezo=-PIEZO_UM, relative_piezo=True)
            time.sleep(SETTLE_S)
            p = nis_util.get_position(NIS)
            self.assertLessEqual(abs(p[3] - z1), TOL_UM,
                                 'piezo position not restored')


if __name__ == '__main__':
    unittest.main()
