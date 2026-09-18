"""
Image segmentation building blocks.

Public API:
  - simple.SimpleSegParams, simple.detect_objects, simple.segment_nuclei_otsu_watershed
  - dummy.dummy_detect_objects
  - remote.remote_detect_objects
"""

from .simple import SimpleSegParams, detect_objects, segment_nuclei_otsu_watershed
from .dummy import dummy_detect_objects
from .remote import remote_detect_objects

__all__ = [
    "SimpleSegParams",
    "detect_objects",
    "segment_nuclei_otsu_watershed",
    "dummy_detect_objects",
    "remote_detect_objects",
]
