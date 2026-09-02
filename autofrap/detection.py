"""
Object detection for the grid survey pipeline.

Contract: read a saved ND2 file -> 2D label array of the same (y, x)
shape as the image, where 0 = background and 1, 2, ... label the
detected objects, plus a binary stimulation mask indicating which
areas are eligible for photostimulation.

Detectors:
  - 'dummy'          fixed circle + rectangle (testing, no dependencies)
  - 'cellpose-remote' cellpose on a separate server (cellpose_server.py);
                      this machine only ships the image over HTTP
Both detectors return a label map; detect() adds the default
stimulation mask (left half of each object).
"""
import numpy as np

import nd2_helpers


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


def default_stimulation_mask(labels):
    """
    default stimulation mask: the left half of each detected object

    Each object is split into two equal-area halves along the
    horizontal axis (split_mask_along_axis_equal_area) and the left
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
            left, _ = split_mask_along_axis_equal_area(labels == lbl, axis=1)
            stim_mask |= left
    return stim_mask


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


def remote_detect_objects(image, server_url, timeout=60, retries=1,
                          **eval_kwargs):
    """
    run cellpose on a remote server (see cellpose_server.py)

    The image is serialized with np.save and POSTed to the server's
    /detect endpoint; the label map comes back in the same format.
    A failed request (connection error, timeout, or HTTP error) is
    retried `retries` times with a short backoff before propagating.

    Parameters
    ----------
    image: 2D np.ndarray (y, x)
        input image
    server_url: str
        base URL of the cellpose server, e.g. 'http://192.168.1.10:8000'
    timeout: float
        request timeout in seconds; the V100 server answers in ~2 s, so
        60 s leaves room for connection latency and queued requests
    retries: int
        number of retries after a failed request, with a 2 s backoff
    eval_kwargs: dict
        optional cellpose model.eval() parameters, sent as query params:
        diameter, min_size, cellprob_threshold, flow_threshold,
        max_size_fraction (see cellpose_server.py for defaults)

    Returns
    -------
    labels: 2D np.ndarray (y, x), int32
        0 = background, 1..N = objects
    """
    import io
    import time

    import requests

    buf = io.BytesIO()
    np.save(buf, image)
    for attempt in range(retries + 1):
        try:
            r = requests.post(f'{server_url}/detect', data=buf.getvalue(),
                              headers={'Content-Type': 'application/x-numpy'},
                              params=eval_kwargs or None,
                              timeout=timeout)
            r.raise_for_status()
            break
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            if attempt == retries:
                raise
            time.sleep(2.0 * (attempt + 1))
    labels = np.load(io.BytesIO(r.content), allow_pickle=False)
    print(f'remote detection: {r.headers.get("X-Inference-Time-S", "?")} s '
          f'({r.headers.get("X-N-Objects", "?")} objects) on {server_url}')

    # CP4 returns uint16 masks; the rest of the pipeline uses int32
    return np.ascontiguousarray(labels, dtype=np.int32)


def detect(nd2_file, channel=0, detector='dummy', server_url=None,
           relabel='distance', **detector_kwargs):
    """
    run detection on a saved ND2 file

    Objects touching the image border are discarded before relabelling
    (clear_border removes the whole label, not just the border pixels)
    and the remaining labels are renumbered to a gap-free 1..N.

    Parameters
    ----------
    nd2_file: str
        path to the ND2 file
    channel: int
        channel index to detect on
    detector: str
        'dummy' (default) or 'cellpose-remote'
    server_url: str, optional
        base URL of the cellpose server (required for 'cellpose-remote')
    relabel: str or None
        relabelling mode applied to the detector's label map:
        'distance' (default) - relabel 1..N by increasing centroid
            distance to the image center (optical axis, process first),
            via relabel_by_distance
        'shuffle' - randomly permute 1..N (avoids raster-order bias),
            via shuffle_labels
        None - return the detector's labels as-is
    detector_kwargs: dict
        detector-specific options; for 'cellpose-remote' these are the
        cellpose model.eval() parameters (diameter, min_size, ...)

    Returns
    -------
    labels: 2D np.ndarray (y, x), int
        0 = background, 1..N = objects
    stimulation_mask: 2D np.ndarray (y, x), bool
        binary mask of areas eligible for photostimulation;
        default convention (default_stimulation_mask): the left half
        of each detected object
    """
    from skimage.segmentation import clear_border, relabel_sequential

    image = nd2_helpers.read_channel(nd2_file, channel)
    if detector == 'dummy':
        labels = dummy_detect_objects(image)
    elif detector == 'cellpose-remote':
        if server_url is None:
            raise ValueError('server_url is required for detector="cellpose-remote"')
        labels = remote_detect_objects(image, server_url, **detector_kwargs)
    else:
        raise ValueError(f'unknown detector {detector!r}')

    # discard objects touching the image border (clear_border removes the
    # whole label, not just the border pixels) and close the gaps this
    # leaves in the numbering; done before any relabelling
    labels = clear_border(labels)
    labels, _, _ = relabel_sequential(labels)  # (labels, forward, inverse)

    if relabel == 'distance':
        labels = relabel_by_distance(labels)
    elif relabel == 'shuffle':
        labels = shuffle_labels(labels)
    elif relabel is not None:
        raise ValueError(f'unknown relabel mode {relabel!r}')

    return labels, default_stimulation_mask(labels)


def cell_mask(labels, cell_id, stimulation_mask=None):
    """
    binary mask of one cell of a label map

    Without a stimulation mask: the whole cell (``labels == cell_id``).
    With one: the intersection of the cell with the stimulation mask,
    i.e. only the areas that are both inside the cell and eligible for
    photostimulation.

    Parameters
    ----------
    labels: 2D np.ndarray
        label map (0 = background, 1..N = objects)
    cell_id: int
        the cell label to extract
    stimulation_mask: 2D np.ndarray, optional
        binary stimulation mask; if given, the cell is intersected with it

    Returns
    -------
    mask: 2D np.ndarray, bool
        binary mask of the cell (or its stimulation-eligible part)
    """
    if stimulation_mask is None:
        return labels == cell_id
    return (labels == cell_id) & stimulation_mask


def mask_to_polygon(mask, tolerance=2.0):
    """
    convert a binary mask to polygon vertices in pixel coordinates

    Uses the largest (outermost) contour of the mask, simplified with
    Douglas-Peucker (`approximate_polygon`).

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
