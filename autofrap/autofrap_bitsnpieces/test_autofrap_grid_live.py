"""
Live test of autofrap_grid at the microscope: 2x2 grid, one cell per
FOV (max_cycles=1), real cellpose detector on the GPU server (the
default detection_fun of autofrap).

Prerequisites: NIS GUI running with the survey ND experiment configured
(2 ch x 1024^2) and the sequential stimulation experiment ready; sample
centered on the desired grid center; cellpose server up
(http://10.163.69.12:8000).

Output: test_acquisitions/autofrap_grid/<stamp>/fovNN/
"""
import os
import sys

# repo root (for nis_util) + autofrap/ itself (for autofrap/detection as
# script-dir siblings) — this script lives two levels down in autofrap/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))

import autofrap

NIS_EXE = r'C:\Program Files\NIS-Elements\nis_ar.exe'
OUT_DIR = r'C:\Users\David\Desktop\nis-automation\test_acquisitions\autofrap_grid'

if __name__ == '__main__':
    results = autofrap.autofrap_grid(NIS_EXE, OUT_DIR, nx=2, ny=2,
                                     spacing=1.0, max_cycles=1)

    print('\nSummary:')
    for i, x, y, fov_dir, fov_results in results:
        if fov_results is None:
            print(f'  fov{i:02d} ({x:+.1f}, {y:+.1f}) -> FAILED')
        else:
            cells = ', '.join(f'c{c:02d}=label {lbl}' for c, lbl, _, _ in fov_results)
            print(f'  fov{i:02d} ({x:+.1f}, {y:+.1f}) -> {cells}')
    print(f'output in {OUT_DIR}')
