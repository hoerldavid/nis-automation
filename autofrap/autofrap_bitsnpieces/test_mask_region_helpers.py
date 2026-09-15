"""
Tests for mask_utils one-stimulation-region-per-label helpers (TODO #29).

run: python autofrap/autofrap_bitsnpieces/test_mask_region_helpers.py
"""
import sys

import numpy as np

from autofrap.core.image.mask import (
    largest_region_per_label,
    most_central_region_per_label,
)

FAILURES = []


def check(name, condition):
    if not condition:
        FAILURES.append(name)
        print(f"FAIL {name}")
    else:
        print(f"ok   {name}")


def test_empty():
    """empty mask → empty mask"""
    labels = np.zeros((32, 32), dtype=np.int32)
    stim = np.zeros((32, 32), dtype=bool)
    result = largest_region_per_label(labels, stim)
    check("empty mask → empty", result.sum() == 0)
    check("shape preserved", result.shape == (32, 32))


def test_single_region_per_label():
    """no-op when each label already has at most one region"""
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[40:50, 40:50] = 2
    stim = labels.astype(bool).copy()
    result = largest_region_per_label(labels, stim)
    check("unchanged", np.array_equal(result, stim))


def test_two_regions_per_label_largest():
    """largest region wins when a label has multiple regions"""
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[10:20, 10:20] = 1  # label 1 at top-left

    # two disconnected regions within label 1's bbox (10:20, 10:20)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:16, 10:16] = True   # 6×6 = 36 px (top-left)
    mask[17:20, 17:20] = True   # 3×3 = 9 px  (bottom-right, gap=1)

    result = largest_region_per_label(labels, mask)
    check("keeps largest region (36 px)",
          result[13, 13])
    check("drops smaller region (9 px)",
          not result[18, 18])


def test_two_regions_per_label_central():
    """most-central region wins"""
    labels = np.zeros((64, 64), dtype=np.int32)
    # label 1 centered roughly at (25, 25)
    labels[10:40, 10:40] = 1

    # two regions: one at top (far), one at bottom-right (closer to centroid)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:15, 10:15] = True    # far top-left, centroid ~12.5
    mask[30:38, 30:38] = True    # closer to label centroid ~25, centroid ~34

    result = most_central_region_per_label(labels, mask)
    check("keeps most central region",
          result[34, 34])
    check("drops far region",
          not result[12, 12])


def test_multi_label_mixed():
    """multiple labels, some with multiple regions"""
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[40:50, 40:50] = 2

    # label 1: two disconnected regions (both within bbox 10:20, 10:20)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:16, 10:16] = True    # 6×6 = 36 px (top-left)
    mask[17:20, 17:20] = True    # 3×3 = 9 px  (bottom-right)
    # label 2: one region
    mask[40:48, 40:48] = True    # 8×8 = 64 px

    result = largest_region_per_label(labels, mask)
    check("label 1 keeps largest",
          int(result[14, 14]))
    check("label 1 drops small",
          not result[18, 18])
    check("label 2 untouched",
          result[44, 44])


def test_all_regions_same_label():
    """all pixels of a label have multiple regions"""
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[:] = 1  # everything is label 1

    # checkerboard pattern → many regions
    mask = np.zeros((64, 64), dtype=bool)
    mask[::2, ::2] = True

    result = largest_region_per_label(labels, mask)
    # ~1024 px in result (largest single 1×1 block in checkerboard is 1 px, but there are ~2048 regions of 1 px each)
    # Actually each 1px block is a region, all equal size → largest picks one
    check("single region remains",
          result.sum() == 1)


def test_all_regions_same_label_central():
    """checkerboard → picks region closest to image centroid"""
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[:] = 1

    mask = np.zeros((64, 64), dtype=bool)
    mask[::2, ::2] = True

    result = most_central_region_per_label(labels, mask)
    # closest to center (32,32) among checkerboard 1px regions
    check("single central-ish region",
          result.sum() == 1)


def main():
    test_empty()
    test_single_region_per_label()
    test_two_regions_per_label_largest()
    test_two_regions_per_label_central()
    test_multi_label_mixed()
    test_all_regions_same_label()
    test_all_regions_same_label_central()

    print(f"\n{len(FAILURES)} failure(s)")
    if FAILURES:
        print(f"Failed: {', '.join(FAILURES)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
