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
    from skimage.measure import regionprops
    stim_mask = np.zeros(labels.shape, dtype=bool)
    for rp in regionprops(labels):
        if rp.label == 0:
            continue
        minr, minc, maxr, maxc = rp.bbox
        crop = labels[minr:maxr, minc:maxc] == rp.label
        left_crop, _ = split_mask_equal_area(crop, axis=1)
        stim_mask[minr:maxr, minc:maxc] |= left_crop
    return stim_mask


def _disk(radius):
    """
    filled disk of `radius` pixel offsets (integer offsets with Euclidean
    norm <= radius), centered in a (2*radius+1) x (2*radius+1) bool array
    """
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def random_circle_stim_mask(labels, area_fraction=0.25, seed=None):
    """
    stimulation mask: one randomly placed circle fully inside each object

    Alternative to half_object_stim_mask: instead of a fixed left half,
    each object gets a single circular stimulation region at a random
    position. The circle covers a fixed fraction of the object's area,
    so the relative bleach size is comparable across objects of
    different sizes.

    The circle is placed fully inside the object: a center is sampled
    uniformly from the erosion of the object with a disk of the target
    radius (i.e. the set of all valid centers). Objects too small for
    the requested radius instead get their largest inscribed circle;
    objects smaller than a 3-px-wide disk are left without a
    stimulation region (the pipeline skips cells without
    stimulation-eligible pixels).

    At most one connected region per cell holds by construction.

    Parameters
    ----------
    labels: 2D np.ndarray (y, x), int
        label map (0 = background, 1..N = objects)
    area_fraction: float, optional
        circle area as a fraction of the object's area (default 0.25)
    seed: int, optional
        seed for the random center placement (for reproducibility)

    Returns
    -------
    stimulation_mask: 2D np.ndarray (y, x), bool
        binary mask of areas eligible for photostimulation
    """
    from scipy import ndimage
    from skimage.measure import regionprops

    rng = np.random.default_rng(seed)
    stim_mask = np.zeros(labels.shape, dtype=bool)

    for rp in regionprops(labels):
        if rp.label == 0:
            continue
        minr, minc, maxr, maxc = rp.bbox
        crop = labels[minr:maxr, minc:maxc] == rp.label
        area = int(crop.sum())
        if area == 0:
            continue
        r = max(1, int(round(np.sqrt(area_fraction * area / np.pi))))

        # Use distance transform to find valid centers: dist >= r
        # Pad with a single False border so EDT is well-defined even if object fills bbox
        pad = 1
        crop_padded = np.pad(crop, pad_width=pad, mode='constant', constant_values=False)
        dist = ndimage.distance_transform_edt(crop_padded)
        # valid centres in padded coordinates, then map back to original crop
        valid = dist >= r
        if not np.any(valid):
            continue
        # pick a random valid center
        ys, xs = np.nonzero(valid)
        i = int(rng.integers(len(ys)))
        cy_c = int(ys[i] - pad)
        cx_c = int(xs[i] - pad)

        # place disk in the full-image mask, clipped to object
        cy = minr + cy_c
        cx = minc + cx_c
        # draw disk directly with skimage to handle bounds
        from skimage.draw import disk as sk_disk
        rr, cc = sk_disk((cy, cx), r, shape=stim_mask.shape)
        stim_mask[rr, cc] |= (labels[rr, cc] == rp.label)

    return stim_mask


def _mask_per_label(labels, mask):
    """
    Yield (label_id, (label == id) & mask) for each foreground label.

    Parameters
    ----------
    labels: np.ndarray
        label map (0 = background, 1..N = objects)
    mask: np.ndarray, same shape as labels
        binary mask

    Yields
    ------
    (label_id, region_mask)
        region_mask = (labels == label_id) & mask
    """
    for lbl in np.unique(labels):
        if lbl > 0:
            yield int(lbl), ((labels == lbl) & mask)


def largest_region_per_label(labels, mask):
    """
    Keep only the largest connected region per label.

    Each object (label) may have multiple disconnected regions. This
    function selects the largest region per object and discards the
    rest. Empty masks and labels with a single region pass through
    unchanged.

    Parameters
    ----------
    labels: np.ndarray
        label map (0 = background, 1..N = objects)
    mask: np.ndarray, same shape as labels
        binary mask (one or more connected regions per label)

    Returns
    -------
    reduced_mask: np.ndarray, same shape and dtype as mask
        binary mask with at most one region per label (the largest)
    """
    from skimage.measure import label as _label, regionprops

    result = np.zeros(labels.shape, dtype=bool)
    for rp in regionprops(labels):
        if rp.label == 0:
            continue
        minr, minc, maxr, maxc = rp.bbox
        region_crop = (labels[minr:maxr, minc:maxc] == rp.label) & mask[minr:maxr, minc:maxc]
        if not np.any(region_crop):
            continue
        components = _label(region_crop, connectivity=1)
        if components.max() == 0:
            continue
        if components.max() == 1:
            result[minr:maxr, minc:maxc] |= region_crop
        else:
            areas = np.bincount(components.ravel())[1:]
            keep = components == np.argmax(areas) + 1
            result[minr:maxr, minc:maxc] |= keep
    return result


def most_central_region_per_label(labels, mask):
    """
    Keep only the most-central connected region per label.

    Each object (label) may have multiple disconnected regions. This
    function selects the region whose centroid is closest to the
    centroid of the entire object (label) and discards the rest.

    Parameters
    ----------
    labels: np.ndarray
        label map (0 = background, 1..N = objects)
    mask: np.ndarray, same shape as labels
        binary mask (one or more connected regions per label)

    Returns
    -------
    reduced_mask: np.ndarray, same shape and dtype as mask
        binary mask with at most one region per label (the most central)
    """
    from skimage.measure import label as _label, regionprops

    result = np.zeros(labels.shape, dtype=bool)
    # pre-compute label centroids once
    label_props = {p.label: p for p in regionprops(labels) if p.label != 0}
    for rp in regionprops(labels):
        if rp.label == 0:
            continue
        minr, minc, maxr, maxc = rp.bbox
        region_crop = (labels[minr:maxr, minc:maxc] == rp.label) & mask[minr:maxr, minc:maxc]
        if not np.any(region_crop):
            continue
        components = _label(region_crop, connectivity=1)
        if components.max() == 0:
            continue
        comp_props = regionprops(components)
        if len(comp_props) == 1:
            result[minr:maxr, minc:maxc] |= region_crop
            continue
        # reference centroid in crop coordinates
        ref = np.array(rp.centroid) - np.array([minr, minc])
        best = min(comp_props, key=lambda p: np.sum((p.centroid - ref) ** 2))
        result[minr:maxr, minc:maxc] |= (components == best.label)
    return result


def clusters_in_object(image, obj_mask, min_cluster_area=15, contrast=1.5,
                       max_cluster_frac=0.2):
    """
    mask of small bright clusters within one labeled object

    The object's own pixel values are Otsu-thresholded (per object, so
    the threshold adapts to its overall brightness), and the resulting
    connected components (4-connectivity) are the candidate clusters. A
    candidate is kept when its area is at least `min_cluster_area` and
    its mean intensity is at least `contrast` times the object median.
    If the TOTAL kept area exceeds `max_cluster_frac` of the object
    area, the object is considered diffuse (e.g. a coarse-diffuse
    protein distribution) and the returned mask is empty.

    Example: a punctate GFP-tagged nuclear protein (GFP-DNMT1) — the
    clusters are the bright puncta inside the nuclei.

    Parameters
    ----------
    image: 2D np.ndarray (y, x)
        intensity image
    obj_mask: 2D np.ndarray (y, x), bool, same shape as image
        one object (e.g. a single cell / nucleus)
    min_cluster_area: int
        minimum cluster area in px (below this is noise)
    contrast: float
        a cluster's mean must be at least `contrast` times the object
        median intensity
    max_cluster_frac: float
        total cluster area as a fraction of the object area; above this
        the object is discarded (empty mask)

    Returns
    -------
    cluster_mask: 2D np.ndarray (y, x), bool, same shape as image
        True on the kept cluster pixels (empty for uniform / diffuse
        objects)
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label as _label

    obj_mask = obj_mask.astype(bool)
    cluster = np.zeros(image.shape, dtype=bool)
    vals = image[obj_mask]
    if vals.size == 0 or vals.max() == vals.min():
        return cluster  # empty object or perfectly flat -> nothing

    median = float(np.median(vals))
    thr = float(threshold_otsu(vals))
    lab = _label((image > thr) & obj_mask)
    total = 0
    for i in range(1, int(lab.max()) + 1):
        comp = lab == i
        area = int(comp.sum())
        if area >= min_cluster_area and image[comp].mean() >= contrast * median:
            cluster |= comp
            total += area
    if total > max_cluster_frac * vals.size:
        return np.zeros(image.shape, dtype=bool)
    return cluster


def cluster_stim_mask(labels, image, min_cluster_area=15, contrast=1.5,
                      max_cluster_frac=0.2, pick='largest', channel=0):
    """
    stimulation mask from small bright clusters within the objects

    Per object, clusters_in_object() finds the bright clusters; the
    results are unioned over all objects. Objects with a uniform
    distribution — or a diffuse one (total cluster area above
    `max_cluster_frac`) — get no mask pixels, so the pipeline's
    next_stimulatable_cell() skips them automatically.

    Example: a punctate GFP-tagged nuclear protein (GFP-DNMT1) — bleach
    one cluster per nucleus and watch whether it recovers (diffusion
    from the other clusters) or stays bleached (tight binding).

    The raw mask can carry one region per cluster, i.e. more than one
    region per object; `pick` enforces the one-region-per-object
    pipeline contract:
      'largest'  -> largest_region_per_label
      'central'  -> most_central_region_per_label
      None       -> keep all clusters (the pipeline will warn)

    Parameters
    ----------
    labels: 2D np.ndarray (y, x), int
        label map (0 = background, 1..N = objects)
    image: 2D np.ndarray (y, x) or 3D np.ndarray (c, y, x)
        intensity image (same shape as labels); if 3D, the channel is
        selected with `channel`
    min_cluster_area: int
        see clusters_in_object
    contrast: float
        see clusters_in_object
    max_cluster_frac: float
        see clusters_in_object
    pick: str or None
        'largest' / 'central' / None (see above)
    channel: int
        channel index to select from a (c, y, x) image; ignored for 2D
        input (assumes the correct channel was already loaded)

    Returns
    -------
    stimulation_mask: 2D np.ndarray (y, x), bool
        binary mask of areas eligible for photostimulation
    """
    # select channel if image is multi-channel
    if image.ndim == 3:
        if channel < 0 or channel >= image.shape[0]:
            raise ValueError(
                f'channel {channel} out of bounds for image with {image.shape[0]} channels')
        image = image[channel]
    elif image.ndim != 2:
        raise ValueError(
            f'cluster_stim_mask expects 2D (y, x) or 3D (c, y, x) image, got {image.ndim}D')

    mask = np.zeros(labels.shape, dtype=bool)
    for lbl in np.unique(labels):
        if lbl > 0:
            mask |= clusters_in_object(image, labels == lbl,
                                       min_cluster_area, contrast,
                                       max_cluster_frac)
    if pick == 'largest':
        return largest_region_per_label(labels, mask)
    if pick == 'central':
        return most_central_region_per_label(labels, mask)
    if pick is None:
        return mask
    raise ValueError(f'unknown pick={pick!r} (use "largest", "central" or None)')


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

    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []
    # crop to bounding box for speed, with 1-px margin for contour detection
    coords = np.argwhere(mask)
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0) + 1
    margin = 1
    minr_p = max(0, minr - margin)
    minc_p = max(0, minc - margin)
    maxr_p = min(mask.shape[0], maxr + margin)
    maxc_p = min(mask.shape[1], maxc + margin)
    crop = mask[minr_p:maxr_p, minc_p:maxc_p]

    contours = find_contours(crop, 0.5)
    if not contours:
        return []

    contour = max(contours, key=len)  # largest / outermost contour
    contour[:, 0] += minr_p
    contour[:, 1] += minc_p
    poly = np.column_stack((contour[:, 1], contour[:, 0]))  # (row, col) -> (x, y)
    if len(poly) > 3:
        poly = approximate_polygon(poly, tolerance=tolerance)

    return [(float(x), float(y)) for x, y in poly]


def filter_intensity_inside(labels, image, channel=0, metric='mean', threshold=0.0):
    """
    Return the label IDs whose mean/median intensity inside the object is > threshold.

    Parameters
    ----------
    labels: 2D np.ndarray (y, x), int
        label map, 0 = background
    image: 2D np.ndarray (y, x) or 3D np.ndarray (c, y, x)
        intensity image
    channel: int
        channel index if image is 3D
    metric: str
        'mean' or 'median'
    threshold: float
        keep labels with metric > threshold

    Returns
    -------
    list of int
        label IDs to keep
    """
    from skimage.measure import regionprops

    if image.ndim == 3:
        img = image[channel]
    else:
        img = image

    if metric not in ('mean', 'median'):
        raise ValueError("metric must be 'mean' or 'median'")

    good = []
    for rp in regionprops(labels, intensity_image=img):
        if rp.label == 0:
            continue
        if metric == 'mean':
            # regionprops uses deprecated attribute name in older versions
            val = getattr(rp, 'mean_intensity', getattr(rp, 'intensity_mean'))
        else:
            mask = labels == rp.label
            val = float(np.median(img[mask]))
        if val > threshold:
            good.append(int(rp.label))
    return good


def filter_intensity_surround(labels, image, distance_px=5, channel=0,
                              metric='mean', threshold=0.0, include_center=False):
    """
    Return the label IDs whose mean/median intensity in the surrounding ring
    of radius distance_px is > threshold.

    The ring is computed exactly with an Euclidean distance transform on a
    cropped bounding box around each object, so the cost is O(crop size) and
    independent of the radius.

    Parameters
    ----------
    labels: 2D np.ndarray (y, x), int
        label map, 0 = background
    image: 2D np.ndarray (y, x) or 3D np.ndarray (c, y, x)
        intensity image
    distance_px: int
        radius of the surrounding ring in pixels
    channel: int
        channel index if image is 3D
    metric: str
        'mean' or 'median'
    threshold: float
        keep labels with metric > threshold
    include_center: bool
        if True, use the whole dilated neighbourhood; if False, use the annulus
        0 < dist <= distance_px

    Returns
    -------
    list of int
        label IDs to keep
    """
    from skimage.measure import regionprops
    from scipy.ndimage import distance_transform_edt

    if image.ndim == 3:
        img = image[channel]
    else:
        img = image

    if metric not in ('mean', 'median'):
        raise ValueError("metric must be 'mean' or 'median'")

    good = []
    for rp in regionprops(labels):
        if rp.label == 0:
            continue
        minr, minc, maxr, maxc = rp.bbox
        r0 = max(0, minr - distance_px)
        c0 = max(0, minc - distance_px)
        r1 = min(img.shape[0], maxr + distance_px)
        c1 = min(img.shape[1], maxc + distance_px)

        img_crop = img[r0:r1, c0:c1]
        mask_crop = (labels[r0:r1, c0:c1] == rp.label)

        dist = distance_transform_edt(~mask_crop)
        if include_center:
            ring = dist <= distance_px
        else:
            ring = (dist > 0) & (dist <= distance_px)

        if not np.any(ring):
            continue
        vals = img_crop[ring]
        val = float(vals.mean()) if metric == 'mean' else float(np.median(vals))
        if val > threshold:
            good.append(int(rp.label))
    return good 



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
