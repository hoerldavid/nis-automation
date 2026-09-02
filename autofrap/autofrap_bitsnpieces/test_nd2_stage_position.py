"""
verify that the stage position recorded in ND2 metadata
(nd2_helpers.stage_position) matches the commanded positions

Two calibration sets, both with the commanded coordinates on hand:
  1. the 20260901_160216 autofrap_grid run — commanded positions are
     in grid_live_test.log next to the run (2x2 grid, 1 cycle/FOV:
     4 survey + 4 FRAP files)
  2. the 20260819 overview run — commanded positions are in the
     filenames (pNN_<x>_<y>.nd2)

run: python autofrap/autofrap_bitsnpieces/test_nd2_stage_position.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nd2_helpers

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOL_UM = 5.0

failures = 0


def check(tag, nd2_file, cmd_xy):
    global failures
    x, y, z = nd2_helpers.stage_position(nd2_file)
    dx, dy = x - cmd_xy[0], y - cmd_xy[1]
    ok = abs(dx) < TOL_UM and abs(dy) < TOL_UM
    if not ok:
        failures += 1
    print(f"{'ok ' if ok else 'FAIL'} {tag:28s} "
          f"({x:+10.1f}, {y:+10.1f}) um  "
          f"commanded ({cmd_xy[0]:+9.1f}, {cmd_xy[1]:+9.1f})  "
          f"d=({dx:+6.2f}, {dy:+6.2f})  z={z:+9.1f}")


# 1. grid run: parse commanded positions from the log
run_dir = os.path.join(ROOT, 'test_acquisitions', 'autofrap_grid', '20260901_160216')
log = open(os.path.join(ROOT, 'test_acquisitions', 'autofrap_grid',
                        'grid_live_test.log')).read()
pat = re.compile(r'\[(\d+)/\d+\] \(([+-]?\d+\.?\d*), ([+-]?\d+\.?\d*)\) um '
                 r'-> \S*?(fov\d+)')
n_grid = 0
for i, x, y, fov in pat.findall(log):
    for kind in ('survey', 'frap'):
        files = [f for f in os.listdir(os.path.join(run_dir, fov))
                 if f.endswith(f'_{kind}.nd2')]
        assert len(files) == 1, (fov, kind, files)
        check(f'grid {fov} {kind}',
              os.path.join(run_dir, fov, files[0]), (float(x), float(y)))
        n_grid += 1
print(f'grid run: {n_grid} file(s) checked')

# 2. overview run: commanded positions in the filenames
ov_dir = os.path.join(ROOT, 'test_acquisitions', 'overview')
n_ov = 0
for fn in sorted(os.listdir(ov_dir)):
    m = re.match(r'\d{8}_\d{6}_p\d+_(.+)\.nd2$', fn)
    assert m, fn
    x, y = m.group(1).split('_')
    check(f'overview {fn}', os.path.join(ov_dir, fn), (float(x), float(y)))
    n_ov += 1
print(f'overview run: {n_ov} file(s) checked')

print(f'\n{failures} failure(s) (tolerance {TOL_UM} um)')
sys.exit(1 if failures else 0)
