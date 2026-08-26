"""
Live smoke test for the refactored nis_util wrappers.

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

Usage: python autofrap/autofrap_bitsnpieces/test_nis_util_live.py
"""
import os
import sys
import time

# repo root (for nis_util) — this script lives two levels down in autofrap/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nis_util

NIS = r'C:\Program Files\NIS-Elements\nis_ar.exe'

MOVE_UM = 2.0      # relative XY move for the set_position round-trip
PIEZO_UM = 1.0     # relative piezo move (only when a piezo is present)
TOL_UM = 0.5       # position read-back tolerance
SETTLE_S = 1.0     # vibration settle after a move (the move itself blocks)


def check(name, fn):
    t0 = time.time()
    try:
        res = fn()
    except Exception as e:
        print('FAIL  %-24s %s: %s' % (name, type(e).__name__, e))
        return None
    print('OK    %-24s (%.1f s) %r' % (name, time.time() - t0, res))
    return res


def main():
    ok = True

    # --- read-only wrappers --------------------------------------------
    check('get_camera_format', lambda: nis_util.get_camera_format(NIS))
    pos = check('get_position', lambda: nis_util.get_position(NIS))
    res = check('get_resolution', lambda: nis_util.get_resolution(NIS))
    check('get_rotation_matrix', lambda: nis_util.get_rotation_matrix(NIS))
    check('get_cam_rotation', lambda: nis_util.get_cam_rotation(NIS))
    ocs = check('get_optical_confs', lambda: nis_util.get_optical_confs(NIS))
    check('is_color_camera', lambda: nis_util.is_color_camera(NIS))
    check('get_current_document', lambda: nis_util.get_current_document(NIS))
    check('get_roi_count', lambda: nis_util.get_roi_count(NIS))

    if res:
        print('      FOV = %r um' % (nis_util.get_fov_from_res(res),))

    if ocs is not None and 'FRAPPA' not in ocs:
        print('FAIL  get_optical_confs: FRAPPA missing from %r' % (ocs,))
        ok = False

    # --- set_position round-trip ----------------------------------------
    if pos is None:
        ok = False
    else:
        x0, y0, z0, z1 = pos
        pz = ' piezo %.3f' % z1 if z1 is not None else ''
        print('      start: (%.3f, %.3f, %.3f)%s um' % (x0, y0, z0, pz))

        nis_util.set_position(NIS, pos_xy=(MOVE_UM, 0.0), relative_xy=True)
        time.sleep(SETTLE_S)
        p = nis_util.get_position(NIS)
        d = (p[0] - (x0 + MOVE_UM), p[1] - y0)
        print('      after +%.1f um: (%.3f, %.3f)  delta (%+.3f, %+.3f) um'
              % (MOVE_UM, p[0], p[1], d[0], d[1]))
        if abs(d[0]) > TOL_UM or abs(d[1]) > TOL_UM:
            print('FAIL  set_position round-trip: move not as commanded')
            ok = False

        nis_util.set_position(NIS, pos_xy=(-MOVE_UM, 0.0), relative_xy=True)
        time.sleep(SETTLE_S)
        p = nis_util.get_position(NIS)
        d = (p[0] - x0, p[1] - y0)
        print('      after return:  (%.3f, %.3f)  delta (%+.3f, %+.3f) um'
              % (p[0], p[1], d[0], d[1]))
        if abs(d[0]) > TOL_UM or abs(d[1]) > TOL_UM:
            print('FAIL  set_position round-trip: original position not restored')
            ok = False
        else:
            print('OK    set_position XY round-trip')

        if z1 is not None:
            nis_util.set_position(NIS, pos_piezo=PIEZO_UM, relative_piezo=True)
            time.sleep(SETTLE_S)
            p = nis_util.get_position(NIS)
            print('      piezo +%.1f um: %.3f (was %.3f)' % (PIEZO_UM, p[3], z1))
            nis_util.set_position(NIS, pos_piezo=-PIEZO_UM, relative_piezo=True)
            time.sleep(SETTLE_S)
            p = nis_util.get_position(NIS)
            d = p[3] - z1
            print('      piezo return:  %.3f  delta %+.3f um' % (p[3], d))
            if abs(d) > TOL_UM:
                print('FAIL  set_position piezo round-trip: not restored')
                ok = False
            else:
                print('OK    set_position piezo round-trip')

    print()
    print('ALL LIVE CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
