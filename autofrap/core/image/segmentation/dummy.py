"""
Dummy detector for testing.

Fixed circle + rectangle, no dependencies.
"""

import numpy as np


def dummy_detect_objects(image):
    """
    dummy object detector: places one circle and one rectangle

    Parameters
    ----------
    image: 2D np.ndarray (y, x) or (c, y, x)
        input image (pixel values are ignored, only the shape is used)

    Returns
    -------
    labels: 2D np.ndarray (y, x), int
        0 = background, 1 = circle, 2 = rectangle
    """
    h, w = image.shape[-2:]  # works for 2D (y, x) and (c, y, x)
    labels = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.ogrid[:h, :w]

    # keep the objects small: FRAP bleaching is a laser scan, so the
    # stimulation time scales with ROI area. The two sizes differ by
    # ~3x on purpose, so a run over both objects also tests that the
    # stimulation duration tracks the ROI area.

    # object 1: circle, center in the upper-left third, radius 1/16 of min. axis
    cy, cx, r = h // 3, w // 3, min(h, w) // 16
    labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1

    # object 2: rectangle, in the lower-right quadrant, 1/16 of the image per side
    labels[3 * h // 4 - h // 16:3 * h // 4,
           3 * w // 4 - w // 16:3 * w // 4] = 2

    return labels
