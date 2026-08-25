"""
Test the merge_label_slices function from calmutils.
This reproduces the scenario described in the conversation.
"""

import numpy as np
from calmutils.segmentation import merge_label_slices

# Initial segmentation: labels 1 and 2
seg0 = np.array([
    [0,1,1,0],
    [0,1,1,0],
    [0,0,0,2],
    [0,0,0,2],
])

# After photostimulation, labels are renumbered but positions unchanged
seg1 = np.array([
    [0,0,3,3],
    [0,0,3,3],
    [0,0,0,4],
    [0,0,0,4],
])

# Merge the two slices to get a consistent labeling
merged = merge_label_slices([seg0, seg1])
seg0_remapped, seg1_remapped = merged[0], merged[1]

print('seg0_remapped:\n', seg0_remapped)
print('seg1_remapped:\n', seg1_remapped)

# Determine mapping from seg0 labels to seg1 labels
# For each label in seg0, find the most common label in seg1 at overlapping pixels
mapping = {}
for lbl0 in np.unique(seg0_remapped):
    if lbl0 == 0:
        continue
    mask0 = seg0_remapped == lbl0
    lbl1_counts = {}
    for lbl1 in np.unique(seg1_remapped[mask0]):
        if lbl1 == 0:
            continue
        lbl1_counts[lbl1] = lbl1_counts.get(lbl1, 0) + 1
    if lbl1_counts:
        mapping[lbl0] = max(lbl1_counts, key=lbl1_counts.get)

print('Mapping seg0 -> seg1:', mapping)

# Check that mapping is as expected
assert mapping == {1:1, 2:2}, 'Mapping incorrect'
print('Test passed: labels mapped correctly.')
""