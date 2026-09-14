"""
Pure grid-geometry helpers for tiled acquisitions (no NIS / hardware
dependency).

Moved out of nis_util.py: gen_grid is not NIS-specific; it is used by
the old wing-scanner code at the repo root (automation.py,
NIS_Macro_Acquisition.ipynb).
"""
from math import ceil


def spiral_positions(position, fov, spacing, max_positions=None):
    """
    Generate stage positions in a square spiral around a center point.

    Positions are spaced by ``spacing`` FOV units, starting at
    ``position`` and spiraling outward counter-clockwise.  This provides
    a center-out visit order that visits nearby cells before distant
    ones — useful for experiments where time matters (e.g. cell
    viability degrades over time).

    Parameters
    ----------
    position : array-like (x, y[, z])
        Starting position (µm); the spiral center.
    fov : float
        Field of view size in µm (assumes square FOV; for rectangular
        FOV, pass the mean of fov_x and fov_y).
    spacing : float
        Distance between adjacent positions in FOV units:
        1 = touching, <1 = overlapping, >1 = gap.
    max_positions : int, optional
        Maximum number of positions to generate.  None = unlimited
        (spiral grows until the caller stops iterating).

    Returns
    -------
    positions : list of 2-tuples (x, y)
        Stage coordinates in µm.

    Examples
    --------
    5×5 FOV grid (spacing=1):

        positions = spiral_positions(start, fov, 1.0, max_positions=25)

    With overlap (spacing=0.8, 13 positions):

        positions = spiral_positions(start, fov, 0.8)

    The first few positions form this pattern (n = layer number):

        n=0   (center)
        n=1   8 positions around center
        n=2   16 positions around n=1
        ...
    """
    x0, y0 = position[:2]
    step = spacing * fov

    positions = []
    count = 0

    # Layer 0: just the center
    positions.append((x0, y0))
    count += 1
    if max_positions is not None and count >= max_positions:
        return positions

    # Layers 1, 2, 3, ...
    n = 1
    while True:
        # Right edge, going up: (n, -(n-1)) → (n, n)
        for y in range(-(n - 1), n + 1):
            if max_positions is not None and count >= max_positions:
                return positions
            positions.append((x0 + n * step, y0 + y * step))
            count += 1

        # Top edge, going left: (n-1, n) → (-n, n)
        for x in range(n - 1, -n - 1, -1):
            if max_positions is not None and count >= max_positions:
                return positions
            positions.append((x0 + x * step, y0 + n * step))
            count += 1

        # Left edge, going down: (-n, n-1) → (-n, -n)
        for y in range(n - 1, -n - 1, -1):
            if max_positions is not None and count >= max_positions:
                return positions
            positions.append((x0 - n * step, y0 + y * step))
            count += 1

        # Bottom edge, going right: (-n+1, -n) → (n, -n)
        for x in range(-n + 1, n + 1):
            if max_positions is not None and count >= max_positions:
                return positions
            positions.append((x0 + x * step, y0 - n * step))
            count += 1

        n += 1


def gen_grid(fov, min_, max_, overlap, snake, half_fov_offset=True, center=True):
    """
    generate a grid of coordinates at which to do a tiled acquisition

    Parameters
    ----------
    fov: array-like
        field-of-view in units
    min_: array-like
        minimum of bbox to scan
    max_: array-like
        maximum of bbox to scan
    overlap: scalar \\in (0,1)
        percent overlap
    snake: boolean
        whether to alternate in x or not
    half_fov_offset: boolean
        whether to correct for NIS 'centering' on locations (-> half FOV offset)
    center: boolean
        whether to center the grid on the bounding box or not (in this case, the object will be in the upper left corner)

    Returns
    -------
    grid: list of 2-tuples
        (x,y) - coordinates at which to image
    """

    # whether coordinates are increasing or decreasing in a dimension
    direction = [1 if max_[0] > min_[0] else -1, 1 if max_[1] > min_[1] else -1]

    # number of tiles
    tilesX = (abs(max_[0] - min_[0]) - fov[0]) / (fov[0] * (1 - overlap))
    tilesY = (abs(max_[1] - min_[1]) - fov[1]) / (fov[1] * (1 - overlap))
    tilesX = max(0, int(ceil(tilesX))) + 1
    tilesY = max(0, int(ceil(tilesY))) + 1

    # re-center grid on bbox
    if center:
        totalX = fov[0] + (tilesX - 1) * (fov[0] * (1 - overlap))
        totalY = fov[1] + (tilesY - 1) * (fov[1] * (1 - overlap))

        #print('{} {}'.format(totalX, totalY))
        extraX = totalX - abs(max_[0] - min_[0])
        extraY = totalY - abs(max_[1] - min_[1])

        #print('{} {}'.format(extraX, extraY))
        min_ = [min_[0] - 0.5 * extraX * direction[0], min_[1] - 0.5 * extraY * direction[1]]

    # correct for NIS's half FOV offset
    if half_fov_offset:
        min_ = [min_[0] + 0.5 * fov[0] * direction[0], min_[1] + 0.5 * fov[1] * direction[1]]

    # steps: increasing or decreasing
    stepX = fov[0] * (1 - overlap) if direction[0] == 1 else - (fov[0] * (1 - overlap))
    stepY = fov[1] * (1 - overlap) if direction[1] == 1 else - (fov[1] * (1 - overlap))

    res = []
    for y in range(tilesY):
        row = [(min_[0] + x * stepX, min_[1] + y * stepY) for x in range(tilesX)]
        if snake and (y % 2 != 0):
            row.reverse()
        res.extend(row)

    return res, tilesX, tilesY, overlap


if __name__ == '__main__':
    print(gen_grid([.6, .6], [1, 0], [0, 1], 0.0, True, True, True))
