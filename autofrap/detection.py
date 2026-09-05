"""
Object detection for the grid survey pipeline.

autofrap() calls a detection_fun: survey_file -> (labels[,
stimulation_mask[, visualization]]), where labels is a 2D integer
array of the same (y, x) shape as the image (0 = background,
1..N = objects); without a mask the whole cell is FRAPed. The mask
holds at most one connected region per cell (cells without a region
are skipped downstream); picking *which* region a cell gets is the
detector's job (DESIGN_GOALS_AUTOFRAP.md, step 6).

detect() composes such a detection_fun from parts and applies the
stable housekeeping + contract checks (composition contract and
examples in its docstring). Parts:

  - nd2_helpers.read_channel    read one survey channel (2D)
  - dummy_detect_objects        fixed circle + rectangle (testing,
                                no dependencies)
  - remote_detect_objects       cellpose on a separate server
                                (cellpose_server.py); this machine
                                only ships the image over HTTP
  - default_stimulation_mask    left half of each object (pass as
                                stim_mask_fun=lambda labels, image:
                                default_stimulation_mask(labels))

Writing your own detector: pass your own detector_fun (and load_fun
/ stim_mask_fun) to detect() for anything that fits image -> labels;
for a fully custom survey_file -> (labels[, mask[, viz]]) callable
(e.g. one that also returns a visualization), pass it straight to
autofrap() instead.
"""
import warnings

import numpy as np


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


def _warn_multi_region(labels, stimulation_mask):
    """
    warn about cells whose stimulation mask has more than one connected
    region

    Detector contract violation (see the detect docstring). This is a
    warning, not an error: mask_to_polygon still works and implicitly
    selects the largest region, so a violating detector degrades the
    run instead of aborting it.

    Connectivity is 4-neighborhood (cross), the same convention
    find_contours uses for boundaries: blobs touching only at a corner
    count as two regions (label's default is full connectivity, which
    would merge them).
    """
    from skimage.measure import label
    for cell_id in np.unique(labels)[1:]:
        cell_stim = stimulation_mask & (labels == cell_id)
        if not cell_stim.any():
            continue  # no FRAP region: allowed, the cell is skipped
        n_regions = label(cell_stim, connectivity=1).max()
        if n_regions > 1:
            warnings.warn(
                f'cell {cell_id} has {n_regions} connected FRAP regions '
                '(detector contract: at most one); the largest region '
                'will be used', stacklevel=2)


def detect(load_fun, detector_fun, relabel='distance',
           clear_border=True, stim_mask_fun=None):
    """
    compose a detection_fun for autofrap()

    The experiment-specific parts - which data to load, which
    detector to run, which areas are FRAP-eligible - are passed in
    as callables; detect() applies only the stable housekeeping and
    the contract checks:

        image  = load_fun(survey_file)        2D (y, x) or (y, x, c)
        labels = detector_fun(image)          2D (y, x), int
        mask   = stim_mask_fun(labels, image) 2D (if given)

    Housekeeping on labels, in this order:
      - clear_border=True: discard objects touching the image border
        (clear_border removes the whole label, not just the border
        pixels) and renumber to a gap-free 1..N
      - relabel: 'distance' (default - relabel 1..N by increasing
        centroid distance to the image center), 'shuffle', or None

    Contract checks (ValueError):
      - labels: 2D, integer, same (y, x) as the image
      - mask: 2D, same shape as labels
    plus a *warning* (not an error): cells with more than one
    connected FRAP region (see _warn_multi_region).

    Returns
    -------
    detection_fun: callable
        survey_file -> (labels, stimulation_mask), or (labels,) if
        stim_mask_fun is None (whole-cell FRAP downstream). No
        visualization is produced here: a detector that returns one
        bypasses detect() and passes its own
        survey_file -> (labels[, mask[, viz]]) to autofrap() directly.

    Examples
    --------
    Built-in parts (channel 0, cellpose on the server, left-half
    mask) - as used by autofrap():

        detect(partial(nd2_helpers.read_channel, channel=0),
               partial(remote_detect_objects, server_url=...),
               stim_mask_fun=lambda labels, image:
                   default_stimulation_mask(labels))

    Multi-channel: detect cells in channel 0, keep only the ones
    expressing the marker in channel 1, FRAP the whole cell:

        def load(f):
            return np.stack([nd2_helpers.read_channel(f, 0),
                             nd2_helpers.read_channel(f, 1)],
                            axis=-1)

        def detect_expressing(img):
            labels = cellpose(img[..., 0])
            expressing = per-cell means of img[..., 1] above threshold
            return np.where(expressing, labels, 0)

        detect(load, detect_expressing, relabel=None)

    Parameters
    ----------
    load_fun: callable
        survey_file -> image, 2D (y, x) or (y, x, c) with one plane
        per channel; which channel(s) to read is the caller's choice
        (e.g. partial(nd2_helpers.read_channel, channel=...))
    detector_fun: callable
        image -> 2D (y, x) int label map (0 = background, 1..N);
        e.g. dummy_detect_objects, partial(remote_detect_objects,
        server_url=...), or your own model
    relabel: str or None
        'distance' (default), 'shuffle', or None (no relabelling)
    clear_border: bool
        discard border-touching objects and renumber gap-free
        (default: True)
    stim_mask_fun: callable or None
        (labels, image) -> 2D binary mask of areas eligible for
        photostimulation (see default_stimulation_mask); receives the
        labels *after* clear_border/relabelling; None: no mask,
        whole-cell FRAP
    """
    if relabel not in ('distance', 'shuffle', None):
        raise ValueError(f'unknown relabel mode {relabel!r}')

    def _detect(survey_file):
        image = load_fun(survey_file)
        labels = detector_fun(image)
        if labels.ndim != 2:
            raise ValueError(
                f'detector_fun returned {labels.ndim}D labels, '
                'expected 2D (y, x)')
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f'labels must be integer, got {labels.dtype}')
        if labels.shape != image.shape[:2]:
            raise ValueError(f'labels/image shape mismatch: '
                             f'{labels.shape} vs {image.shape}')

        if clear_border:
            from skimage.segmentation import (clear_border as _clear,
                                              relabel_sequential)
            labels = _clear(labels)
            labels, _, _ = relabel_sequential(labels)  # (l, fwd, inv)

        if relabel == 'distance':
            labels = relabel_by_distance(labels)
        elif relabel == 'shuffle':
            labels = shuffle_labels(labels)

        if stim_mask_fun is None:
            return labels
        mask = stim_mask_fun(labels, image)
        if mask.ndim != 2 or mask.shape != labels.shape:
            raise ValueError(
                f'stimulation mask must be 2D with the labels shape, '
                f'got {getattr(mask, "shape", None)}')
        mask = mask.astype(bool)
        _warn_multi_region(labels, mask)
        return labels, mask

    return _detect


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
