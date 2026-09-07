"""
General-purpose mask / label-map utilities (no pipeline-specific logic).

Mask convention: binary arrays (bool or 0/1); label maps are integer
arrays (0 = background, 1..N = objects).
"""
import numpy as np


def split_mask_equal_area(mask, axis=0):
    """
    Split a binary mask into two halves of (approximately) equal area along a given axis.
    This assumes a single connected object in the mask and may produce weird results
    for masks containing multiple connected components.

    Parameters
    ----------
    mask: np.ndarray
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


def half_object_stim_mask(labels):
    """
    default stimulation mask: the left half of each detected object

    Each object is split into two equal-area halves along the
    horizontal axis (split_mask_equal_area) and the left
    half is marked as stimulation-eligible, mimicking a real FRAP
    experiment in which part of the cell is bleached and diffusion
    from the rest is recorded.

    Parameters
    ----------
    labels: 2D np.ndarray (y, x), int
        label map (0 = background, 1..N = objects)

    Returns
    -------
    stimulation_mask: 2D np.ndarray (y, x), bool
        binary mask of areas eligible for photostimulation
    """
    stim_mask = np.zeros(labels.shape, dtype=np.bool_)
    for lbl in np.unique(labels):
        if lbl > 0:
            # TODO: only do it in object bbox for speedup (use regionprops?)
            left, _ = split_mask_equal_area(labels == lbl, axis=1)
            stim_mask |= left
    return stim_mask


# TODO: add helpers for enforcing one-stimulation-area-per-labels?
# e.g. - only keep largest stimulation area per label - only keep most central per label?
# would also go in


def shuffle_labels(labels, seed=None):
    """
    randomly permute the object labels of a label map

    Detectors usually number objects in raster order (top-left first),
    which can bias downstream processing that treats labels in order.
    This function renumbers 1..N with a random permutation; the
    background (0) is left unchanged.

    Parameters
    ----------
    labels: np.ndarray
        label map (0 = background, 1..N = objects)
    seed: int, optional
        seed for the random permutation (for reproducibility)

    Returns
    -------
    shuffled: np.ndarray
        label map of the same shape and dtype with the labels 1..N
        randomly permuted
    """
    import fastremap

    n = int(labels.max()) if labels.size else 0
    if n == 0:
        return labels.copy()

    rng = np.random.default_rng(seed)
    perm = np.concatenate([[0], rng.permutation(np.arange(1, n + 1))])
    remap = dict(zip(range(n + 1), perm))  # old -> new, background stays 0

    return fastremap.remap(labels, remap)


def relabel_by_distance(labels, reference=None):
    """
    relabel objects by increasing distance from a reference point

    The object whose centroid (skimage `regionprops`) is closest to
    `reference` becomes label 1, the next closest label 2, and so on;
    the background (0) is left unchanged. The default reference is the
    image center, i.e. the optical axis of the microscope, which has the
    least distortion/vignetting and should therefore be processed first.

    Parameters
    ----------
    labels: np.ndarray
        label map (0 = background, 1..N = objects), 2D or 3D
    reference: array-like of pixel coordinates, optional
        point to measure distances from; one value per axis, e.g. (y, x)
        for 2D or (z, y, x) for 3D. Defaults to the image center.

    Returns
    -------
    relabeled: np.ndarray
        label map of the same shape and dtype with 1..N renumbered by
        increasing centroid distance to the reference
    """
    import fastremap
    from skimage.measure import regionprops

    n = int(labels.max()) if labels.size else 0
    if n == 0:
        return labels.copy()

    if reference is None:
        # per-axis center, e.g. (h/2, w/2) for 2D or (d/2, h/2, w/2) for 3D
        reference = np.array(labels.shape, dtype=float) / 2.0
    ref = np.asanyarray(reference, dtype=float)

    props = regionprops(labels)
    # centroids are in array-index order, e.g. (y, x) or (z, y, x)
    centroids = np.array([p.centroid for p in props])
    ids = np.array([p.label for p in props])

    # squared Euclidean distance from each centroid to the reference
    dist2 = np.sum((centroids - ref) ** 2, axis=1)
    order = np.argsort(dist2, kind="stable")

    # new labels 1..N assigned to the objects in increasing-distance order
    remap = {0: 0}
    for new_lbl, idx in enumerate(order, start=1):
        remap[int(ids[idx])] = new_lbl

    return fastremap.remap(labels, remap)


def mask_to_polygon(mask, tolerance=2.0):
    """
    convert a binary mask to polygon vertices in pixel coordinates

    Uses the largest (outermost) contour of the mask, simplified with
    Douglas-Peucker (`approximate_polygon`). For a single region this
    is the outer boundary (hole contours are ignored). If the mask
    contains several disconnected regions (a detector contract
    violation, see detect), the largest region is selected.

    Parameters
    ----------
    mask: 2D np.ndarray (y, x), bool or 0/1
        binary mask
    tolerance: float
        Douglas-Peucker simplification tolerance [px]

    Returns
    -------
    polygon: list of (x, y) tuples
        pixel coordinates (x right, y down, (0,0) at top-left corner);
        empty list if the mask is empty
    """
    from skimage.measure import find_contours, approximate_polygon

    contours = find_contours(mask, 0.5)
    if not contours:
        return []

    contour = max(contours, key=len)  # largest / outermost contour
    poly = np.column_stack((contour[:, 1], contour[:, 0]))  # (row, col) -> (x, y)
    if len(poly) > 3:
        poly = approximate_polygon(poly, tolerance=tolerance)

    return [(float(x), float(y)) for x, y in poly] 

