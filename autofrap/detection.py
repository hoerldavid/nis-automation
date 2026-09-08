"""
Object detection for the grid survey pipeline.

autofrap() calls a detection_fun: survey_file -> (labels[,
stimulation_mask[, visualization]]) or a bare label map, where
labels is a 2D integer array of the same (y, x) shape as the image
(0 = background, 1..N = objects); without a mask the whole cell is
FRAPed. The mask
holds at most one connected region per cell (cells without a region
are skipped downstream); picking *which* region a cell gets is the
detector's job (DESIGN_GOALS_AUTOFRAP.md, step 6).

build_detector() composes such a detection_fun from parts and applies
the stable housekeeping + contract checks (composition contract and
examples in its docstring). Parts:

  - nd2_helpers.read_channel    read one survey channel (2D)
  - dummy_detect_objects        fixed circle + rectangle (testing,
                                no dependencies)
  - remote_detect_objects       cellpose on a separate server
                                (cellpose_server.py); this machine
                                only ships the image over HTTP
  - mask_utils.half_object_stim_mask  left half of each object (pass as
                                stim_mask_fun=lambda labels, image:
                                half_object_stim_mask(labels))
  - visualization_fun           image -> 2D grayscale or (y, x, 3/4)
                                RGB(A) for the QC overlay (e.g.
                                lambda image: image, or a channel
                                picker for a multi-channel load)

Image convention: multi-channel images are (c, y, x) (scientific
format); the only (y, x, 3/4) array in the pipeline is the RGB(A)
visualization (display format).

Writing your own detector: pass your own detector_fun / load_fun /
stim_mask_fun / visualization_fun to build_detector() for anything
that fits image -> labels (+ mask / viz as above); for a fully custom
survey_file -> (labels[, mask[, viz]]) callable (e.g. a visualization
that depends on the labels, or an input that is not a survey nd2
file), pass it straight to autofrap() instead.
"""
import warnings

from autofrap.mask_utils import half_object_stim_mask, relabel_by_distance, shuffle_labels
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


def build_detector(load_fun, detector_fun, relabel='distance',
                   clear_border=True, stim_mask_fun=None,
                   visualization_fun=None):
    """
    compose a detection_fun for autofrap()

    The experiment-specific parts - which data to load, which
    detector to run, which areas are FRAP-eligible, how the image is
    shown in the QC overlay - are passed in as callables;
    build_detector() applies only the stable housekeeping and the
    contract checks:

        image  = load_fun(survey_file)        2D (y, x) or (c, y, x)
        labels = detector_fun(image)          2D (y, x), int
        mask   = stim_mask_fun(labels, image) 2D (if given)
        viz    = visualization_fun(image)     2D or (y, x, 3/4)
                                                 (if given)

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
    connected FRAP region (see _warn_multi_region), and a failing or
    malformed visualization (see visualization_fun).

    Returns
    -------
    detection_fun: callable
        survey_file -> (labels[, stimulation_mask[, viz]]): the
        positions are fixed (2 = mask, 3 = viz), so with a viz but no
        mask the result is (labels, None, viz). (labels,) if nothing
        else is given (whole-cell FRAP downstream).

    Examples
    --------
    Built-in parts (channel 0, cellpose on the server, left-half
    mask, the channel itself as visualization) - as used by
    autofrap():

        build_detector(partial(nd2_helpers.read_channel, channel=0),
                       partial(remote_detect_objects, server_url=...),
                       stim_mask_fun=lambda labels, image:
                           half_object_stim_mask(labels),
                       visualization_fun=lambda image: image)

    Multi-channel: detect cells in channel 0, keep only the ones
    expressing the marker in channel 1, FRAP the whole cell, show
    channel 0 in the overlay:

        def load(f):
            return np.stack([nd2_helpers.read_channel(f, 0),
                             nd2_helpers.read_channel(f, 1)],
                            axis=0)  # (c, y, x)

        def detect_expressing(img):
            labels = cellpose(img[..., 0])
            expressing = per-cell means of img[..., 1] above threshold
            return np.where(expressing, labels, 0)

        build_detector(load, detect_expressing, relabel=None,
                       visualization_fun=lambda image: image[0])

    Parameters
    ----------
    load_fun: callable
        survey_file -> image, 2D (y, x) or (c, y, x) with one plane
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
        photostimulation (see half_object_stim_mask); receives the
        labels *after* clear_border/relabelling; None: no mask,
        whole-cell FRAP
    visualization_fun: callable or None
        image -> 2D (y, x) or (y, x, 3/4) RGB(A) image for the QC
        overlay; receives the same loaded image the detector got
        (2D or (c, y, x)), independent of the detection result.
        Note the convention: scientific images are (c, y, x), only
        the visualization is (y, x, 3/4) (display format).
        None: no visualization (the overlay is drawn on a blank
        canvas). Best effort: any failure (exception or wrong output
        shape, e.g. a 2-channel image) only warns and drops the
        visualization - it is cosmetic and must not break the run.
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
        if labels.shape != image.shape[-2:]:  # 2D (y, x) or (c, y, x)
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

        mask = None
        if stim_mask_fun is not None:
            mask = stim_mask_fun(labels, image)
            if mask.ndim != 2 or mask.shape != labels.shape:
                raise ValueError(
                    f'stimulation mask must be 2D with the labels '
                    f'shape, got {getattr(mask, "shape", None)}')
            mask = mask.astype(bool)
            _warn_multi_region(labels, mask)

        viz = None
        if visualization_fun is not None:
            viz = _make_viz(visualization_fun, image, labels.shape)

        if mask is None and viz is None:
            return (labels,)
        if mask is None:
            return labels, None, viz
        if viz is None:
            return labels, mask
        return labels, mask, viz

    return _detect


def _make_viz(visualization_fun, image, shape):
    """
    best-effort visualization: run visualization_fun and check the
    output shape; on failure (exception or wrong shape) warn and
    return None

    The visualization is cosmetic (QC overlay only) and must not
    break the run - a bad visualization_fun is a warning, not an
    error, unlike the labels/mask contract checks.
    """
    try:
        viz = visualization_fun(image)
    except Exception as e:
        warnings.warn(
            f'visualization_fun failed: {e!r}; continuing without a '
            'visualization', stacklevel=2)
        return None
    ok = (isinstance(viz, np.ndarray)
          and ((viz.ndim == 2 and viz.shape == shape)
               or (viz.ndim == 3 and viz.shape[:2] == shape
                   and viz.shape[2] in (3, 4))))
    if not ok:
        warnings.warn(
            f'visualization_fun returned {type(viz).__name__} of '
            f'shape {getattr(viz, "shape", None)}; expected 2D '
            f'{shape} or RGB(A) ({shape[0]}, {shape[1]}, 3/4); '
            'continuing without a visualization', stacklevel=2)
        return None
    return viz


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


if __name__ == '__main__':
    import sys
    from functools import partial

    import nd2_helpers

    f = sys.argv[1] if len(sys.argv) > 1 else \
        r'C:\Users\David\Desktop\nis-automation\overview\20260819_173530_p01_-0262.4_-0270.0.nd2'
    labels, stim, viz = build_detector(
        partial(nd2_helpers.read_channel, channel=0),
        dummy_detect_objects,
        stim_mask_fun=lambda labels, image:
            half_object_stim_mask(labels),
        visualization_fun=lambda image: image)(f)
    vals, counts = np.unique(labels, return_counts=True)
    print(f'{f}')
    print(f'label shape: {labels.shape} (y, x), dtype: {labels.dtype}')
    for v, c in zip(vals, counts):
        print(f'  label {v}: {c} px')
    print(f'stim_mask shape: {stim.shape}, nonzero: {np.sum(stim)}')
    print(f'viz shape: {viz.shape}')
