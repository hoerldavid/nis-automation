"""
Object detection for the grid survey pipeline.

Contract: read a saved ND2 file -> 2D label array of the same (y, x)
shape as the image, where 0 = background and 1, 2, ... label the
detected objects.

For now this is a DUMMY detector (places fixed circle + rectangle);
swap in the real detector keeping the same interface.
"""
import numpy as np
import nd2


def read_channel(nd2_file, channel=0):
    """
    read one channel of an ND2 file as a 2D (y, x) array

    Parameters
    ----------
    nd2_file: str
        path to the ND2 file
    channel: int
        channel index to read

    Returns
    -------
    image: np.ndarray
        2D image array (y, x)
    """
    with nd2.ND2File(nd2_file) as f:
        return f.asarray()[channel]


def dummy_detect_objects(image):
    """
    dummy object detector: places one circle and one rectangle

    Parameters
    ----------
    image: 2D np.ndarray (y, x)
        input image (pixel values are ignored, only the shape is used)

    Returns
    -------
    labels: 2D np.ndarray (y, x), int
        0 = background, 1 = circle, 2 = rectangle
    """
    h, w = image.shape
    labels = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.ogrid[:h, :w]

    # keep the objects small: FRAP bleaching is a laser scan, so the
    # stimulation time scales with the ROI area

    # object 1: circle, center in the upper-left third, radius 1/16 of min. axis
    cy, cx, r = h // 3, w // 3, min(h, w) // 16
    labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1

    # object 2: rectangle, in the lower-right quadrant, 1/8 of the image per side
    labels[3 * h // 4 - h // 8:3 * h // 4, 3 * w // 4 - w // 8:3 * w // 4] = 2

    return labels


def detect(nd2_file, channel=0):
    """
    run detection on a saved ND2 file

    Parameters
    ----------
    nd2_file: str
        path to the ND2 file
    channel: int
        channel index to detect on

    Returns
    -------
    labels: 2D np.ndarray (y, x), int
        0 = background, 1..N = objects
    """
    return dummy_detect_objects(read_channel(nd2_file, channel))


def label_to_polygon(labels, label_id, tolerance=2.0):
    """
    convert one label of a label map to polygon vertices in pixel coordinates

    Parameters
    ----------
    labels: 2D np.ndarray (y, x), int
        label map (0 = background)
    label_id: int
        label to convert
    tolerance: float
        Douglas-Peucker simplification tolerance [px]

    Returns
    -------
    polygon: list of (x, y) tuples
        pixel coordinates (x right, y down, (0,0) at top-left corner);
        empty list if the label is not present
    """
    from skimage.measure import find_contours, approximate_polygon

    contours = find_contours(labels == label_id, 0.5)
    if not contours:
        return []

    contour = max(contours, key=len)  # largest / outermost contour
    poly = np.column_stack((contour[:, 1], contour[:, 0]))  # (row, col) -> (x, y)
    if len(poly) > 3:
        poly = approximate_polygon(poly, tolerance=tolerance)

    return [(float(x), float(y)) for x, y in poly]


if __name__ == '__main__':
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else \
        r'C:\Users\David\Desktop\nis-automation\overview\20260819_173530_p01_-0262.4_-0270.0.nd2'
    lab = detect(f)
    vals, counts = np.unique(lab, return_counts=True)
    print(f'{f}')
    print(f'label shape: {lab.shape} (y, x), dtype: {lab.dtype}')
    for v, c in zip(vals, counts):
        print(f'  label {v}: {c} px')
    for v in vals[1:]:
        poly = label_to_polygon(lab, v)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        print(f'  label {v}: {len(poly)} vertices, bbox x:[{min(xs):.1f},{max(xs):.1f}] y:[{min(ys):.1f},{max(ys):.1f}]')
