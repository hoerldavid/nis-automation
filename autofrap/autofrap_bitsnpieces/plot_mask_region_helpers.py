"""Visual tests for mask_utils region helpers (TODO #29).

Saves PNGs next to this file showing:
  - the label map
  - the original stimulation mask
  - the reduced mask (largest / most-central)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autofrap.core.image.mask import (
    largest_region_per_label,
    most_central_region_per_label,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _save(labels, stim, reduced, title):
    """Save a 3-row figure: labels / stim / reduced."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), dpi=100)

    # 1. label map
    ax = axes[0]
    ax.imshow(labels, cmap="tab10")
    ax.set_title(f"{title}\n(label map, {labels.max()} objects)")
    ax.axis("off")

    # 2. original stimulation mask
    ax = axes[1]
    ax.imshow(stim, cmap="Reds", vmin=0, vmax=1)
    ax.set_title("Original stimulation mask")
    ax.axis("off")

    # 3. reduced mask
    ax = axes[2]
    ax.imshow(reduced, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Reduced mask (one region per label)")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, title.replace(" ", "_") + ".png"), dpi=100)
    print(f"Saved: {title.replace(' ', '_')}.png")
    plt.close(fig)


# --- Case 1: two regions per label, largest wins ---
labels1 = np.zeros((64, 64), dtype=np.int32)
labels1[10:20, 10:20] = 1  # label 1 at top-left
stim1 = np.zeros((64, 64), dtype=bool)
stim1[10:16, 10:16] = True   # 6×6 = 36 px (larger)
stim1[17:20, 17:20] = True   # 3×3 = 9 px  (smaller)
reduced1 = largest_region_per_label(labels1, stim1)
_save(labels1, stim1, reduced1, "largest_region_per_label — two regions")


# --- Case 2: two regions per label, most-central wins ---
labels2 = np.zeros((64, 64), dtype=np.int32)
labels2[10:40, 10:40] = 1  # label 1, centroid ~(25, 25)
stim2 = np.zeros((64, 64), dtype=bool)
stim2[10:15, 10:15] = True   # far top-left (centroid ~12.5)
stim2[30:38, 30:38] = True   # closer to label centroid (centroid ~34)
reduced2 = most_central_region_per_label(labels2, stim2)
_save(labels2, stim2, reduced2, "most_central_region_per_label — two regions")


# --- Case 3: multi-label mixed ---
labels3 = np.zeros((64, 64), dtype=np.int32)
labels3[10:20, 10:20] = 1
labels3[40:50, 40:50] = 2
stim3 = np.zeros((64, 64), dtype=bool)
stim3[10:16, 10:16] = True   # label 1, 36 px
stim3[17:20, 17:20] = True   # label 1, 9 px
stim3[40:48, 40:48] = True   # label 2, 64 px (single region)
reduced3 = largest_region_per_label(labels3, stim3)
_save(labels3, stim3, reduced3, "largest_region — multi-label mixed")


# --- Case 4: single region per label (no-op) ---
labels4 = np.zeros((64, 64), dtype=np.int32)
labels4[10:20, 10:20] = 1
labels4[40:50, 40:50] = 2
stim4 = np.zeros((64, 64), dtype=bool)
stim4[10:20, 10:20] = True
stim4[40:50, 40:50] = True
reduced4 = largest_region_per_label(labels4, stim4)
_save(labels4, stim4, reduced4, "no-op — single region per label")


# --- Case 5: empty ---
labels5 = np.zeros((64, 64), dtype=np.int32)
stim5 = np.zeros((64, 64), dtype=bool)
reduced5 = largest_region_per_label(labels5, stim5)
_save(labels5, stim5, reduced5, "empty mask")


print("All plots saved.")
