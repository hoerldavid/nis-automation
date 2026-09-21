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

  - autofrap.io.nd2.read_channel    read one survey channel (2D)
  - autofrap.core.image.segmentation.dummy_detect_objects        fixed circle + rectangle (testing,
                                no dependencies)
  - autofrap.core.image.segmentation.remote_detect_objects       cellpose on a separate server
                                (cellpose_server.py); this machine
                                only ships the image over HTTP
  - autofrap.core.image.mask.half_object_stim_mask  left half of each object (pass as
                                stim_mask_fun=lambda labels, image:
                                half_object_stim_mask(labels))
  - autofrap.core.image.mask.random_circle_stim_mask  one random circle per object,
                                covering a fixed area fraction (pass as
                                stim_mask_fun=lambda labels, image:
                                random_circle_stim_mask(labels))
  - autofrap.core.image.mask.cluster_stim_mask  small bright clusters within each
                                object (Otsu per object + size/contrast
                                filters); uniform / diffuse objects get
                                no pixels (pass as
                                stim_mask_fun=cluster_stim_mask)
  - visualization_fun           image -> 2D grayscale or (y, x, 3/4)
                                RGB(A) for the QC overlay (e.g.
                                lambda image: image, or a channel
                                picker for a multi-channel load)
  - filter_function             (labels, image) -> set/list of "good"
                                label IDs; labels not in this set
                                are zeroed out (e.g. expression
                                filtering — see below)

Image convention: multi-channel images are (c, y, x) (scientific
format); the only (y, x, 3/4) array in the pipeline is the RGB(A)
visualization (display format).

Writing your own detector: pass your own detector_fun / load_fun /
stim_mask_fun / visualization_fun to build_detector() for anything
that fits image -> labels (+ mask / viz as above); for a fully custom
survey_file -> (labels[, mask[, viz]]) callable (e.g. a visualization
that depends on the labels, or an input that is not a survey nd2
file), pass it straight to autofrap() instead.

filter_function is called *after* the detector and *before*
clear_border/relabelling — it works on the raw detector IDs so the
caller can use the exact label map from the detector (e.g. reference
the detector's label IDs when computing per-cell marker intensity).

**Custom detector file** (for ``python -m autofrap.pipeline --detector FILE``):
any ``.py`` file that defines ``detection_fun`` — a callable with
the ``build_detector`` return signature
(`survey_file -> (labels[, mask[, viz]])`). The runner imports the
file and uses ``detection_fun`` directly. See
``autofrap/detectors/example_detector.py``.

**Extra detector parameters** (``--detector-arg key=value``):
the runner passes additional keyword arguments to ``detection_fun``
at call time (``detection_fun(survey_file, **kwargs)``). A function
assembled by ``build_detector`` routes them to its sub-functions
according to the ``parameter_map`` setting (see
:func:`build_detector`); a fully custom ``detection_fun`` can simply
accept them via ``**kwargs`` or by name.
"""
import warnings
import inspect

from autofrap.core.image.mask import relabel_by_distance, shuffle_labels
from autofrap.core.image.qc import default_visualization as _default_visualization
import numpy as np

def _accepts_kwargs(func):
    """Check if a function accepts variable keyword arguments (**kwargs)."""
    try:
        sig = inspect.signature(func)
        return any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    except (ValueError, TypeError):
        return False

def _warn_multi_region(labels, stimulation_mask):
    """
    Warn about cells whose stimulation mask has more than one connected
    region. This is a detector contract violation (see the detect docstring)
    and results in warning, not an error: downstream mask_to_polygon still works
    and implicitly selects the largest region.

    Connectivity is 4-neighborhood (cross), the same convention
    find_contours uses for boundaries during polygon generation.
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
                   clear_border=True, filter_function=None,
                   stim_mask_fun=None, visualization_fun=None,
                   parameter_map=None):
    """
    compose a detection_fun for autofrap()

    The experiment-specific parts - which data to load, which
    detector to run, which areas are FRAP-eligible, how the image is
    shown in the QC overlay - are passed in as callables;
    build_detector() applies only the stable housekeeping and the
    contract checks:

        image   = load_fun(survey_file)         2D (y, x) or (c, y, x)
        labels  = detector_fun(image)           2D (y, x), int
        mask    = stim_mask_fun(labels, image)  2D (if given)
        viz     = visualization_fun(image)      2D or (y, x, 3/4)
                                                  (if given)

    Housekeeping on labels, in this order:
      - filter_function (if given): keep only the label IDs returned
        by the callable; labels not in the set are zeroed
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

        from autofrap.core.image.segmentation import remote_detect_objects
        build_detector(partial(autofrap.io.nd2.read_channel, channel=0),
                       partial(remote_detect_objects, server_url=...),
                       stim_mask_fun=lambda labels, image:
                           half_object_stim_mask(labels),
                       visualization_fun=lambda image: image)

    Multi-channel: detect cells in channel 0, keep only the ones
    expressing the marker in channel 1, FRAP the whole cell, show
    channel 0 in the overlay:

        def load(f):
            return np.stack([autofrap.io.nd2.read_channel(f, 0),
                             autofrap.io.nd2.read_channel(f, 1)],
                            axis=0)  # (c, y, x)

        def detect_expressing(img):
            from autofrap.core.image.segmentation import remote_detect_objects
            labels = remote_detect_objects(img[..., 0], server_url=...)
            expressing = per-cell means of img[..., 1] above threshold
            return np.where(expressing, labels, 0)

        build_detector(load, detect_expressing, relabel=None,
                       visualization_fun=lambda image: image[0])

    Expression filter (keep only cells with marker intensity above
    threshold — filter_function gets the raw detector labels + the
    loaded image, uses regionprops to check intensity, returns
    "good" label IDs):

        from skimage.measure import regionprops

        def load(f):
            return np.stack([
                autofrap.io.nd2.read_channel(f, 0),  # DAPI
                autofrap.io.nd2.read_channel(f, 1),  # marker
            ], axis=0)

        def filter_expressing(labels, image):
            good = []
            for rp in regionprops(labels, intensity_image=image[1]):
                if rp.mean_intensity > 500:
                    good.append(rp.label)
            return good

        from autofrap.core.image.segmentation import remote_detect_objects
        detection_fun = build_detector(
            load,
            partial(remote_detect_objects, server_url=...),
            filter_function=filter_expressing,
            stim_mask_fun=lambda labels, image:
                half_object_stim_mask(labels),
            visualization_fun=lambda image: image[0])

    Parameters
    ----------
    load_fun: callable
        survey_file -> image, 2D (y, x) or (c, y, x) with one plane
        per channel; which channel(s) to read is the caller's choice
        (e.g. partial(autofrap.io.nd2.read_channel, channel=...))
    detector_fun: callable
        image -> 2D (y, x) int label map (0 = background, 1..N);
        e.g. autofrap.core.image.segmentation.dummy_detect_objects,
        partial(autofrap.core.image.segmentation.remote_detect_objects,
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
    filter_function: callable or None
        (labels, image) -> set or list of "good" label IDs; labels
        not in this set are zeroed out (relabel_sequential then
        renumbers the remaining IDs gap-free). Called *after* the
        detector and *before* clear_border/relabelling — the caller
        gets the exact raw detector labels. None: no filtering.
    visualization_fun: callable, None, or False
        image -> 2D (y, x) or (y, x, 3/4) RGB(A) image for the QC
        overlay; receives the same loaded image the detector got
        (2D or (c, y, x)), independent of the detection result.
        Note the convention: scientific images are (c, y, x), only
        the visualization is (y, x, 3/4) (display format).
        None: use the default visualization from
        autofrap.core.image.qc.default_visualization (grayscale pass-through
        for 2-D, RGB composite for 3-D). False: explicitly disable
        visualization, the overlay is drawn on a blank canvas.
        Best effort: any failure (exception or wrong output shape, e.g. a
        2-channel image) only warns and drops the visualization - it is
        cosmetic and must not break the run.
    parameter_map: str or dict or None, optional
        Controls how extra keyword arguments are routed to the
        sub-functions (``load_fun``, ``detector_fun``, etc.) when
        ``detection_fun`` is called with additional keyword arguments
        (e.g. from ``python -m autofrap.pipeline --detector-arg key=value``).

        - ``None`` (default): no extra arguments are passed to any
          sub-function.
        - ``'auto'``: each sub-function receives the extra arguments
          that it can actually accept — i.e. the ones whose names
          appear in its signature, or all of them if it accepts
          ``**kwargs``.
        - ``dict``: an explicit mapping of the form
          ``{<build_detector_arg_name>: {<runtime_key>: <internal_name>}}``.
          Only the listed runtime keys are passed, and they are
          renamed to ``<internal_name>`` before being passed to the
          sub-function. Keys not listed in the mapping are not
          passed to that sub-function.

        Example (explicit mapping, resolving name collisions):

            build_detector(
                load_fun=my_load,          # my_load(f, channel=...)
                detector_fun=my_detect,    # my_detect(img, channel=...)
                parameter_map={
                    'load_fun':     {'load_channel': 'channel'},
                    'detector_fun': {'det_channel': 'channel'},
                })

            # called as: detection_fun(f, load_channel=0, det_channel=1)
            # -> my_load(f, channel=0)
            # -> my_detect(img, channel=1)
    """
    if relabel not in ('distance', 'shuffle', None):
        raise ValueError(f'unknown relabel mode {relabel!r}')

    def _route(func, arg_name, runtime_kwargs):
        """Extract the subset of runtime_kwargs for a sub-function."""
        if parameter_map is None or not runtime_kwargs:
            return {}

        # TODO: remove auto? might only cause confusion?
        if parameter_map == 'auto':
            if _accepts_kwargs(func):
                return runtime_kwargs
            try:
                sig = inspect.signature(func)
                return {k: v for k, v in runtime_kwargs.items()
                        if k in sig.parameters}
            except (ValueError, TypeError):
                return {}
        # explicit dict mapping
        func_map = parameter_map.get(arg_name, {})
        return {func_map[rk]: rv for rk, rv in runtime_kwargs.items()
                if rk in func_map}

    def _detect(survey_file, **runtime_kwargs):
        
        image = load_fun(survey_file, **_route(load_fun, 'load_fun', runtime_kwargs))
        labels = detector_fun(image, **_route(detector_fun, 'detector_fun', runtime_kwargs))
        
        # check detection output - should be integer label map with same yx shape as input
        if labels.ndim != 2:
            raise ValueError(
                f'detector_fun returned {labels.ndim}D labels, '
                'expected 2D (y, x)')
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f'labels must be integer, got {labels.dtype}')
        if labels.shape != image.shape[-2:]:  # 2D (y, x) or (c, y, x)
            raise ValueError(f'labels/image shape mismatch: '
                             f'{labels.shape} vs {image.shape}')

        # filter: keep only labels in the set returned by
        # filter_function; labels not in the set are zeroed.
        if filter_function is not None:
            good = filter_function(
                labels, image, **_route(filter_function, 'filter_function',
                                        runtime_kwargs))
            # np.isin needs a list (sets produce object-dtype arrays
            # that don't match integer label maps).
            labels = np.isin(labels, list(good)) * labels

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
            mask = stim_mask_fun(
                labels, image, **_route(stim_mask_fun, 'stim_mask_fun',
                                        runtime_kwargs))
            if mask.ndim != 2 or mask.shape != labels.shape:
                raise ValueError(
                    f'stimulation mask must be 2D with the labels '
                    f'shape, got {getattr(mask, "shape", None)}')
            mask = mask.astype(bool)
            _warn_multi_region(labels, mask)

        viz = None
        # Use default visualization if none was provided; False disables visualization
        if visualization_fun is False:
            viz = None
        elif visualization_fun is None:
            viz = _apply_and_check_viz(
                _default_visualization, image, labels.shape)
        else:
            viz = _apply_and_check_viz(
                visualization_fun, image, labels.shape,
                **_route(visualization_fun, 'visualization_fun',
                         runtime_kwargs))

        if mask is None and viz is None:
            return (labels,)
        if mask is None:
            return labels, None, viz
        if viz is None:
            return labels, mask
        return labels, mask, viz

    return _detect


def _apply_and_check_viz(visualization_fun, image, shape, **kwargs):
    """
    Run visualization_fun and check that the output can be used by downstream plotting. 
    On failure (exception or wrong shape) warn and return None.

    The visualization is cosmetic (QC overlay only) and must not break the run.
    Therefore a bad visualization_fun result is a warning, not an
    error, unlike the labels/mask contract checks.
    """
    try:
        viz = visualization_fun(image, **kwargs)
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


def load_detector_file(path):
    """
    import a user-supplied detector file and return its ``detection_fun``

    The file must define ``detection_fun: survey_file -> (labels[, mask[, viz]])``
    with the same return signature as :func:`build_detector`.

    Parameters
    ----------
    path: str
        path to a ``.py`` file that defines ``detection_fun``

    Returns
    -------
    detection_fun: callable
        the ``detection_fun`` exported from the file

    Examples
    --------
    A simple detector file (see ``autofrap/detectors/example_detector.py``):

        # my_detector.py
        from autofrap.core.image.segmentation import dummy_detect_objects
        from autofrap.core.image.mask import half_object_stim_mask

        def detection_fun(f):
            labels = dummy_detect_objects(np.zeros((100, 100)))
            mask = half_object_stim_mask(labels)
            return labels, mask

    Usage from the CLI:

        python -m autofrap.pipeline --detector my_detector.py ...
    """
    import importlib.util
    import os

    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'could not load detector file: {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    detection_fun = getattr(mod, 'detection_fun', None)
    if detection_fun is None or not callable(detection_fun):
        raise ValueError(
            f'detector file {path} must define a callable "detection_fun"; '
            f'found {type(detection_fun).__name__}')
    return detection_fun


# TODO: remove complicated wrapper for deprecated functions
def __getattr__(name):
    if name == "dummy_detect_objects":
        import warnings
        warnings.warn(
            "autofrap.core.detection.dummy_detect_objects is deprecated, import from autofrap.core.image.segmentation instead",
            DeprecationWarning, stacklevel=2,
        )
        from autofrap.core.image.segmentation import dummy_detect_objects as _f
        return _f
    if name == "remote_detect_objects":
        import warnings
        warnings.warn(
            "autofrap.core.detection.remote_detect_objects is deprecated, import from autofrap.core.image.segmentation instead",
            DeprecationWarning, stacklevel=2,
        )
        from autofrap.core.image.segmentation import remote_detect_objects as _f
        return _f
    if name == "cell_mask":
        import warnings
        warnings.warn(
            "autofrap.core.detection.cell_mask is deprecated, import from autofrap.core.image.mask instead",
            DeprecationWarning, stacklevel=2,
        )
        from autofrap.core.image.mask import cell_mask as _f
        return _f
    if name == "default_visualization":
        import warnings
        warnings.warn(
            "autofrap.core.detection.default_visualization is deprecated, import from autofrap.core.image.qc instead",
            DeprecationWarning, stacklevel=2,
        )
        from autofrap.core.image.qc import default_visualization as _f
        return _f
    raise AttributeError(name)
