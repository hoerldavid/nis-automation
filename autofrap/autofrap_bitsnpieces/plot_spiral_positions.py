"""Visual plot of spiral_positions for visual verification (TODO #15).

Saves a PNG showing the first N positions with their visit order.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grid_utils import spiral_positions

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_spiral(max_positions, fov, spacing, center, title):
    """Plot the spiral pattern."""
    positions = spiral_positions(center, fov, spacing, max_positions)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)

    # Plot all positions
    ax.scatter(xs, ys, s=20, c='steelblue', zorder=3)

    # Annotate first few positions
    for i, (x, y) in enumerate(positions[:min(25, len(positions))]):
        ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center',
                    color='white', fontweight='bold', zorder=4)

    # Annotate center
    ax.annotate(f"0", (center[0], center[1]), fontsize=10, ha='center', va='center',
                color='red', fontweight='bold', zorder=5)

    # Draw layer boundaries
    step = spacing * fov
    for n in range(1, 5):
        square = plt.Rectangle(
            (center[0] - n * step, center[1] - n * step),
            2 * n * step, 2 * n * step,
            fill=False, edgecolor='gray', linestyle='--', linewidth=0.5, zorder=1
        )
        ax.add_patch(square)

    # Draw arrows showing first few steps
    for i in range(min(len(positions) - 1, 15)):
        ax.annotate('', xy=positions[i + 1], xytext=positions[i],
                    arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.5),
                    zorder=2)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.invert_yaxis()  # image convention: y increases downward

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"spiral_{max_positions}_{title.replace(' ', '_')}.png")
    fig.savefig(path, dpi=100)
    print(f"Saved: {path}")
    plt.close(fig)


# --- Plots ---

plot_spiral(25, fov=10.0, spacing=1.0, center=(0, 0),
            title="25 positions (5x5)")

plot_spiral(65, fov=10.0, spacing=1.0, center=(0, 0),
            title="65 positions (layers 0-3)")

plot_spiral(200, fov=10.0, spacing=1.0, center=(0, 0),
            title="200 positions (layers 0-5)")

# Overlapping FOV example
plot_spiral(25, fov=10.0, spacing=0.8, center=(0, 0),
            title="25 positions with overlap (spacing=0.8)")

print("All plots saved.")
