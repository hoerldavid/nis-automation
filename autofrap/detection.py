"""
Object detection for the grid survey pipeline.

Contract: read a saved ND2 file -> 2D label array of the same (y, x)
shape as the image, where 0 = background and 1, 2, ... label the
detected objects, plus a binary stimulation mask indicating which
areas are eligible for photostimulation.

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


def split_mask_along_axis_equal_area(mask, axis=0):
    """
    Split a binary mask into two halves of (approximately) equal area along a given axis.
    This assumes a single connected object in the mask and may produce weird results
    for masks containing multiple connected components.

    Parameters
    ----------
    label_mask: np.ndarray
        Binary mask of the object (0/1 or bool dtype)
    axis: int
        Axis along which to split (must be 0..ndim-1)

    Returns
    -------
    (first_half, second_half)
        Two boolean masks of the same shape as mask; disjoint, covering the object
    """
    mask = mask.astype(bool)
    ndim = mask.ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis must be 0..{ndim-1} for {ndim}-D array (got {axis})")

    total_area = mask.sum()
    if total_area == 0:
        # TODO: just return zeros and not error?
        raise ValueError('Empty mask - no object found')

    # Sum over all axes except the split axis, giving a 1D profile
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    sums_along_axis = mask.sum(axis=other_axes)
    
    # cumulative sum along profile
    cumsum = np.cumsum(sums_along_axis)

    # we want to be as close to half area as possible
    target = total_area / 2

    # Find the first index where cumsum >= target
    idx = np.searchsorted(cumsum, target, side='left')
    
    if idx > 0:
        diff_at = np.abs(cumsum[idx] - target)
        diff_prev = np.abs(cumsum[idx - 1] - target)
        # choose one index before if the difference is smaller
        idx = idx - 1 if diff_prev < diff_at else idx

    first = np.zeros_like(mask)
    second = np.zeros_like(mask)

    # Set the slice for the target axis; keep all other axes unchanged
    slices = [slice(None)] * ndim
    
    slices[axis] = slice(0, idx + 1)
    first[tuple(slices)] = mask[tuple(slices)]

    slices[axis] = slice(idx + 1, None)
    second[tuple(slices)] = mask[tuple(slices)]

    return first, second


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
    stimulation_mask: 2D np.ndarray (y, x), bool
        binary mask of areas eligible for stimulation (True = eligible);
        for the dummy: the left half of each detected object
    """
    h, w = image.shape
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

    # dummy: only the left half of each object is stimulation-eligible
    # (equal-area split), mimicking a real FRAP experiment in which part
    # of the cell is bleached and diffusion from the rest is recorded
    stim_mask = np.zeros((h, w), dtype=np.bool_)
    for lbl in np.unique(labels):
        if lbl > 0:
            left, _ = split_mask_along_axis_equal_area(labels == lbl, axis=1)
            stim_mask |= left

    return labels, stim_mask


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
    stimulation_mask: 2D np.ndarray (y, x), bool
        binary mask of areas eligible for photostimulation
    """
    return dummy_detect_objects(read_channel(nd2_file, channel))


def detect_stim_mask(labels, stimulation_mask, cell_id):
    """
    compute the combined ROI mask for a specific cell

    The result is the intersection of the cell region with the
    stimulation mask, so it only covers areas that are both inside
    the cell and eligible for photostimulation.

    Parameters
    ----------
    labels: 2D np.ndarray
        label map (0 = background, 1..N = objects)
    stimulation_mask: 2D np.ndarray
        binary stimulation mask
    cell_id: int
        the cell label to extract

    Returns
    -------
    combined_mask: 2D np.ndarray
        binary mask where (labels == cell_id) & stimulation_mask
    """
    return (labels == cell_id) & stimulation_mask


def detect_polygon_stim_mask(labels, stimulation_mask, cell_id, tolerance=2.0):
    """
    convert a cell's stimulation-eligible region to polygon vertices

    This is the same as ``label_to_polygon`` but uses the intersection
    of the cell with the stimulation mask, so the resulting polygon
    only covers areas eligible for photostimulation.

    Parameters
    ----------
    labels: 2D np.ndarray
        label map (0 = background)
    stimulation_mask: 2D np.ndarray
        binary stimulation mask
    cell_id: int
        cell label to extract
    tolerance: float
        Douglas-Peucker simplification tolerance [px]

    Returns
    -------
    polygon: list of (x, y) tuples
        pixel coordinates (x right, y down, (0,0) at top-left corner);
        empty list if the combined region is empty
    """
    from skimage.measure import find_contours, approximate_polygon

    combined_mask = detect_stim_mask(labels, stimulation_mask, cell_id)
    contours = find_contours(combined_mask, 0.5)
    if not contours:
        return []

    contour = max(contours, key=len)  # largest / outermost contour
    poly = np.column_stack((contour[:, 1], contour[:, 0]))  # (row, col) -> (x, y)
    if len(poly) > 3:
        poly = approximate_polygon(poly, tolerance=tolerance)

    return [(float(x), float(y)) for x, y in poly]


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
    labels, stim = lab
    vals, counts = np.unique(labels, return_counts=True)
    print(f'{f}')
    print(f'label shape: {labels.shape} (y, x), dtype: {labels.dtype}')
    for v, c in zip(vals, counts):
        print(f'  label {v}: {c} px')
    print(f'stim_mask shape: {stim.shape}, nonzero: {np.sum(stim)}')
