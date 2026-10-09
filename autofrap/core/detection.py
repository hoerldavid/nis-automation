"""
Modular object detection / FRAP region generation.

autofrap() calls a detection_fun: survey_file_path -> (labels[,stimulation_mask[, visualization]]),
where labels is an integer label map of the same (y, x) shape as the image
(0 = background, 1..N = objects) and stimulation_mask a binary array indicating wich parts to FRAP - without a mask the whole cell is
FRAPed. the 2D/RGB(A) visualization image is used as the background for QC visualization of detected regions.

build_detector() composes such a detection_fun from parts
this detector also applies housekeeping + contract checks.

Image convention: multi-channel images are (c, y, x) (scientific
format); the only (y, x, 3/4) array in the pipeline is the RGB(A)
visualization (display format).
"""

import warnings
import importlib.util
import os

import numpy as np
from skimage.measure import label, regionprops

from autofrap.core.image.mask import relabel_by_distance, shuffle_labels
from autofrap.core.image.qc import default_visualization as _default_visualization


def _warn_multi_region(labels, stimulation_mask):
    """
    Warn about cells whose stimulation mask has more than one connected
    region. Connectivity is 4-neighborhood (cross), the same convention
    find_contours uses for boundaries during polygon generation.

    Only warns, as downstream mask_to_polygon still works and implicitly selects the largest region.
    """

    for region in regionprops(labels):
        cell_stim_mask = stimulation_mask[region.slice]
        if not cell_stim_mask.any():
            continue  # no FRAP region: allowed, the cell is skipped
        n_regions = label(cell_stim_mask, connectivity=1).max()
        if n_regions > 1:
            warnings.warn(
                f"cell {region.label} has {n_regions} connected FRAP regions "
                "(detector contract: at most one); the largest region "
                "will be used",
                stacklevel=2,
            )


def _apply_and_check_viz(visualization_fun, image, shape, **kwargs):
    """
    Run visualization_fun and check that the output is 2D or RGB(A).
    On failure warn and return None -> visualization will be plotted on blank background.
    """
    try:
        viz = visualization_fun(image, **kwargs)
    except Exception as e:
        warnings.warn(
            f"visualization_fun failed: {e!r}; continuing without a visualization",
            stacklevel=2,
        )
        return None
    ok = isinstance(viz, np.ndarray) and (
        (viz.ndim == 2 and viz.shape == shape)
        or (viz.ndim == 3 and viz.shape[:2] == shape and viz.shape[2] in (3, 4))
    )
    if not ok:
        warnings.warn(
            f"visualization_fun returned {type(viz).__name__} of "
            f'shape {getattr(viz, "shape", None)}; expected 2D '
            f"{shape} or RGB(A) ({shape[0]}, {shape[1]}, 3/4); "
            "continuing without a visualization",
            stacklevel=2,
        )
        return None
    return viz


def build_detector(
    load_fun,
    detector_fun,
    relabel="distance",
    clear_border=True,
    filter_function=None,
    stim_mask_fun=None,
    visualization_fun=None,
    parameter_map=None,
):
    """
    Compose a detection_fun for autofrap()

    The experiment-specific parts - which data to load, which
    detector to run, which labels to keep, which areas are FRAP-eligible, how the image is
    shown in the QC overlay - are passed in as callables;
    build_detector() composes into one detection funtion with housekeeping and the
    contract checks:

        image   = load_fun(survey_file)         2D (y, x) or (c, y, x)
        labels  = detector_fun(image)           2D (y, x), int
        good_labels = filter_function(labels, image)    set of int
        mask    = stim_mask_fun(labels, image)  2D (if given)
        viz     = visualization_fun(image)      2D or (y, x, 3/4)
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
        survey_file_path -> (labels[, stimulation_mask[, viz]]): the
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
                if rp.intensity_mean > 500:
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
    parameter_map: dict or None, optional
        Controls how extra keyword arguments are routed to the
        sub-functions (``load_fun``, ``detector_fun``, etc.) when
        ``detection_fun`` is called with additional keyword arguments
        (e.g. from ``python -m autofrap.pipeline --detector-arg key=value``).

        - ``None`` (default): no extra arguments are passed to any
          sub-function.
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
    if relabel not in ("distance", "shuffle", None):
        raise ValueError(f"unknown relabel mode {relabel!r}")
    if parameter_map is not None and not isinstance(parameter_map, dict):
        raise ValueError(
            "parameter_map must be None or an explicit dict, got "
            f"{parameter_map!r} - the 'auto' mode was removed; use an "
            "explicit {step: {runtime_key: internal_name}} mapping "
            "(see WRITING_DETECTOR.md)"
        )

    def _route(function_name, runtime_kwargs):
        """Extract the subset of runtime_kwargs for a sub-function."""
        if parameter_map is None or not runtime_kwargs:
            return {}
        param_name_map_for_fun = parameter_map.get(function_name, {})
        return {
            param_name_map_for_fun[rk]: rv
            for rk, rv in runtime_kwargs.items()
            if rk in param_name_map_for_fun
        }

    def _detect(survey_file, **runtime_kwargs):

        image = load_fun(survey_file, **_route("load_fun", runtime_kwargs))
        labels = detector_fun(image, **_route("detector_fun", runtime_kwargs))

        # check detection output - should be integer label map with same yx shape as input
        if labels.ndim != 2:
            raise ValueError(
                f"detector_fun returned {labels.ndim}D labels, expected 2D (y, x)"
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"labels must be integer, got {labels.dtype}")
        if labels.shape != image.shape[-2:]:  # 2D (y, x) or (c, y, x)
            raise ValueError(
                f"labels/image shape mismatch: {labels.shape} vs {image.shape}"
            )

        # filter: keep only labels in the set returned by
        # filter_function; labels not in the set are zeroed.
        if filter_function is not None:
            good = filter_function(
                labels, image, **_route("filter_function", runtime_kwargs)
            )
            # np.isin needs a list (sets produce object-dtype arrays
            # that don't match integer label maps).
            labels = np.isin(labels, list(good)) * labels

        if clear_border:
            from skimage.segmentation import clear_border as _clear, relabel_sequential

            labels = _clear(labels)
            labels, _, _ = relabel_sequential(labels)  # (l, fwd, inv)

        if relabel == "distance":
            labels = relabel_by_distance(labels)
        elif relabel == "shuffle":
            labels = shuffle_labels(labels)

        mask = None
        if stim_mask_fun is not None:
            mask = stim_mask_fun(
                labels, image, **_route("stim_mask_fun", runtime_kwargs)
            )
            if mask.ndim != 2 or mask.shape != labels.shape:
                raise ValueError(
                    f"stimulation mask must be 2D with the labels "
                    f'shape, got {getattr(mask, "shape", None)}'
                )
            mask = mask.astype(bool)
            _warn_multi_region(labels, mask)

        viz = None
        # Use default visualization if none was provided; False disables visualization
        if visualization_fun is False:
            viz = None
        elif visualization_fun is None:
            viz = _apply_and_check_viz(_default_visualization, image, labels.shape)
        else:
            viz = _apply_and_check_viz(
                visualization_fun,
                image,
                labels.shape,
                **_route("visualization_fun", runtime_kwargs),
            )

        if mask is None and viz is None:
            return (labels,)
        if mask is None:
            return labels, None, viz
        if viz is None:
            return labels, mask
        return labels, mask, viz

    return _detect


def load_detector_file(path):
    """
    Import a user-supplied detector .py file and return its ``detection_fun``

    The file must define ``detection_fun: survey_file -> (labels[, mask[, viz]])``
    (same return signature as functions assembled by :func:`build_detector`.)

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
        from autofrap.io.nd2 import read_channel
        from autofrap.core.image.segmentation import dummy_detect_objects
        from autofrap.core.image.mask import half_object_stim_mask

        def detection_fun(f):
            image = read_channel(f, channel=0)
            labels = dummy_detect_objects(image)
            mask = half_object_stim_mask(labels)
            return labels, mask

    Usage from the CLI:

        python -m autofrap.pipeline --detector my_detector.py ...
    """

    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load detector file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    detection_fun = getattr(mod, "detection_fun", None)
    if detection_fun is None or not callable(detection_fun):
        raise ValueError(
            f'detector file {path} must define a callable "detection_fun"; '
            f"found {type(detection_fun).__name__}"
        )
    return detection_fun


def unpack_detection_result(result):
    """
    Normalize a detection_fun return value to (labels, stimulation_mask, viz).

    Accepts a bare label map (normalized to a 1-tuple) or a 1-3
    tuple/list ``(labels[, stimulation_mask[, visualization]])`` in the
    contract order; the missing entries are None. This is the single
    home of the unpacking logic - the pipeline and the detector runner
    (:mod:`autofrap.detectors.cli`) both use it.

    Parameters
    ----------
    result: np.ndarray or tuple/list
        what ``detection_fun`` returned

    Returns
    -------
    (labels, stimulation_mask, viz)
        stimulation_mask / viz are None when the detector omitted them

    Raises
    ------
    ValueError
        anything else (0-tuple, 4+-tuple, other type)
    """
    if isinstance(result, np.ndarray):
        result = (result,)
    if not isinstance(result, (tuple, list)) or not 1 <= len(result) <= 3:
        raise ValueError(
            f"detection_fun returned {type(result).__name__} "
            f"(length {len(result) if isinstance(result, (tuple, list)) else '-'}); "
            "expected (labels[, stimulation_mask[, visualization]])"
        )
    labels = result[0]
    stimulation_mask = result[1] if len(result) > 1 else None
    viz = result[2] if len(result) > 2 else None
    return labels, stimulation_mask, viz


def parse_detector_args(pairs):
    """
    Parse repeated KEY=VALUE strings (``--detector-arg``) into a kwargs dict.

    A value becomes an int (or float if it contains a ``.``) when it
    parses as one, otherwise it stays a string. This is the single home
    of the ``--detector-arg`` value parsing - the pipeline and the
    detector runner use it, so the two CLIs cannot drift apart.

    Parameters
    ----------
    pairs: iterable of str
        the raw KEY=VALUE arguments

    Returns
    -------
    dict
        keyword arguments for ``detection_fun``

    Raises
    ------
    ValueError
        an entry without ``=``
    """
    kwargs = {}
    for arg in pairs:
        if "=" not in arg:
            raise ValueError(f"expects KEY=VALUE, got: {arg!r}")
        key, val = arg.split("=", 1)
        try:
            val = float(val) if "." in val else int(val)
        except ValueError:
            pass
        kwargs[key] = val
    return kwargs
