"""
Deprecated shim for autofrap.core.simple_seg

The simple segmentation building blocks have moved to
autofrap.core.image.segmentation.simple. This module is kept for
backwards compatibility and will emit a DeprecationWarning on import.

Migrate to:

    from autofrap.core.image.segmentation import SimpleSegParams, detect_objects, segment_nuclei_otsu_watershed
"""
import warnings

warnings.warn(
    "autofrap.core.simple_seg is deprecated, use autofrap.core.image.segmentation instead",
    DeprecationWarning,
    stacklevel=2,
)

from autofrap.core.image.segmentation.simple import (
    SimpleSegParams,
    detect_objects,
    segment_nuclei_otsu_watershed,
)

__all__ = [
    "SimpleSegParams",
    "detect_objects",
    "segment_nuclei_otsu_watershed",
]
