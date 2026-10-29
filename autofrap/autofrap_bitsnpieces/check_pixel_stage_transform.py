"""
Live check of the closed-form pixel -> stage formula (single FOV):

    stage(px) = p_acq + s * R^T @ F @ (px - c)

with s = pixel_size/mag, R = Get_CalibrationAngleMatrix (applied
TRANSPOSED — row-vector convention in the manual), F = diag(1, -1),
c = image center, p_acq = stage position at acquisition.
Reference: docs/NIS_REFERENCE.md § Pixel <-> stage transform
(live-verified 20261019 with a fitted map; this script checks the
closed-form formula against the GUI directly).

Procedure at the microscope:
  1. acquire a single image (live snap) at the current position — do NOT
     move the stage afterwards (the get_position read below is the
     anchor p_acq only while the stage is where the image was taken);
  2. keep the acquired image as the active view;
  3. run this script — it prints example pixel coords and their
     predicted stage coords;
  4. in the NIS GUI, right-click each printed pixel coordinate (context
     menu readout, whole-micron precision) and compare with the
     predicted stage coordinates.

Caveats:
  - a camera ROI changes the image size; c is then the ROI center in
    saved-image pixel coordinates (live-verified 20261027, off-center
    ROI — see docs/NIS_REFERENCE.md § Pixel <-> stage transform).
  - the flip F = diag(1, -1) is a one-time live measurement of this
    unit; if the camera light path was changed since 20261019, a
    mismatch would show up here as a vertical mirror of the residuals.

Run directly at the microscope (NIS open):
    python autofrap/autofrap_bitsnpieces/check_pixel_stage_transform.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np

from autofrap.microscope import nis as nis_util

NIS_EXE = r'C:\Program Files\NIS-Elements\nis_ar.exe'

# F — pixel -> camera-frame axis flip (pixel y grows down, camera y up)
FLIP = np.array([[1.0, 0.0], [0.0, -1.0]])


def pixel_to_stage(px, p_acq, s, R, center):
    """closed-form pixel -> stage map (docs/NIS_REFERENCE.md)"""
    px = np.asarray(px, dtype=float)
    p_acq = np.asarray(p_acq, dtype=float)
    return p_acq + s * (R.T @ FLIP @ (px - center))


def main():
    pos = nis_util.get_position(NIS_EXE)[:2]  # (x, y); drop z0/z1
    res = nis_util.get_resolution(NIS_EXE)
    rot = nis_util.get_rotation_matrix(NIS_EXE)
    roi_enabled, roi = nis_util.get_camera_roi(NIS_EXE)

    xres, yres, pixel_size, mag = res
    center = np.array([xres / 2.0, yres / 2.0])
    s = pixel_size / mag
    R = np.asarray(rot, dtype=float).reshape(2, 2)
    fov = nis_util.get_fov_from_res(res)

    print('stage position   : x=%.1f  y=%.1f (um)' % (pos[0], pos[1]))
    print('resolution       : %d x %d px' % (xres, yres))
    print('pixel size / mag : %.4f / %.1f -> s = %.4f um/px'
          % (pixel_size, mag, s))
    print('FOV              : %.1f x %.1f um' % fov)
    print('rotation matrix R:\n%s' % R)
    print('det(R)           : %.3f' % np.linalg.det(R))
    print('camera ROI       : %s %s' % ('ACTIVE - anchor assumption untested!'
                                        if roi_enabled else 'off', roi))
    print()

    # example pixels: center, corners (in-bounds), quarter points
    px_max = (xres - 1, yres - 1)
    examples = [
        ('center', (int(center[0]), int(center[1]))),
        ('top-left', (0, 0)),
        ('top-right', (px_max[0], 0)),
        ('bottom-left', (0, px_max[1])),
        ('bottom-right', (px_max[0], px_max[1])),
        ('quarter (1/4, 1/4)', (xres // 4, yres // 4)),
        ('quarter (3/4, 3/4)', (3 * xres // 4, 3 * yres // 4)),
    ]

    print('%-22s %-16s %-26s' % ('point', 'pixel (col,row)', 'stage (x, y) [um]'))
    print('-' * 66)
    for label, px in examples:
        st = pixel_to_stage(px, pos, s, R, center)
        print('%-22s (%4d, %4d)     (x=%9.2f, y=%9.2f)'
              % (label, px[0], px[1], st[0], st[1]))
    print()
    print('right-click each pixel in the GUI (context-menu stage readout,')
    print('whole-micron) and compare with the predicted stage coords above.')


if __name__ == '__main__':
    main()
