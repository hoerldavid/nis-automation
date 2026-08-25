"""
Split a binary mask into two halves of (approximately) equal area along a chosen axis.

The function works as follows:
1. Compute the total number of foreground pixels.
2. Compute the cumulative sum of pixels along the requested axis.
3. Find the cut coordinate where the cumulative sum is closest to half the total.
4. Build two masks that cover the two halves.

The masks are guaranteed to be disjoint and to cover the whole object.
"""

import numpy as np
from skimage.measure import regionprops


def split_mask_along_axis_equal_area(label_mask: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a binary mask into two halves of roughly equal area along the given axis.

    Parameters
    ----------
    label_mask : np.ndarray
        2‑D binary mask of the object.
    axis : int, optional
        0 → split along rows (vertical cut → left/right halves)
        1 → split along columns (horizontal cut → top/bottom halves)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (first_half, second_half) – each is a mask of the same shape as ``label_mask``.
    """
    if label_mask.ndim != 2:
        raise ValueError("label_mask must be 2‑D")

    # Ensure mask is boolean
    mask = label_mask.astype(bool)

    # Total area of the object
    total_area = mask.sum()
    if total_area == 0:
        raise ValueError("Empty mask – no object found")

    # Compute cumulative sum along the chosen axis
    if axis == 0:  # vertical split → cut along rows
        # Sum per row
        row_sums = mask.sum(axis=1)
        # Cumulative sum
        cum = np.cumsum(row_sums)
        # Target half area
        target = total_area / 2
        # Find row where cumulative sum is closest to target
        cut_row = np.searchsorted(cum, target)
        # Build halves
        first = np.zeros_like(mask, dtype=bool)
        first[:cut_row, :] = mask[:cut_row, :]
        second = np.zeros_like(mask, dtype=bool)
        second[cut_row:, :] = mask[cut_row:, :]
    elif axis == 1:  # horizontal split → cut along columns
        col_sums = mask.sum(axis=0)
        cum = np.cumsum(col_sums)
        target = total_area / 2
        cut_col = np.searchsorted(cum, target)
        first = np.zeros_like(mask, dtype=bool)
        first[:, :cut_col] = mask[:, :cut_col]
        second = np.zeros_like(mask, dtype=bool)
        second[:, cut_col:] = mask[:, cut_col:]
    else:
        raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")

    return first, second

# Demo ---------------------------------------------------------------
if __name__ == "__main__":
    # Create a random irregular shape
    rng = np.random.default_rng(42)
    shape = (20, 30)
    mask = rng.integers(0, 2, size=shape).astype(bool)
    # Force it to have at least one object
    if not mask.any():
        mask[10, 15] = True

    f, s = split_mask_along_axis_equal_area(mask, axis=0)
    print("Total area:", mask.sum())
    print("First half area:", f.sum())
    print("Second half area:", s.sum())
    print("Difference:", abs(f.sum() - s.sum()))
""