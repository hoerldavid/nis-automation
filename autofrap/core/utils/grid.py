"""
Grid-geometry helpers for tiled acquisitions.
"""
from math import ceil


def spiral_positions(position, fov, max_positions, spacing=1.0):
    """
    Generate stage positions in a square spiral around a center point.

    Positions are spaced by ``spacing`` FOV units, starting at
    ``position`` and spiraling outward counter-clockwise.

    The first few positions form this pattern (n = layer number):

        n=0   (center)
        n=1   8 positions around center
        n=2   16 positions around n=1
        ...

    Parameters
    ----------
    position : array-like (x, y)
        Starting position; the spiral center.
    fov : fov: (fov_x, fov_y)
        field of view per axis
    max_positions : int
        Maximum number of positions to generate.
    spacing : float, optional
        Distance between adjacent positions in FOV units:
        1 (default) = touching, <1 = overlapping, >1 = gap.

    Returns
    -------
    positions : list of 2-tuples (x, y)
        Stage coordinates.
    """

    gen = _spiral_positions_generator(position, fov, spacing)
    return [next(gen) for _ in range(max_positions)]


def _spiral_positions_generator(position, fov, spacing=1.0): 
    """
    Generator for (infinite) spiral positions.
    """
    
    fov_x, fov_y = fov
    x0, y0 = position
    step_x = spacing * fov_x
    step_y = spacing * fov_y

    # Layer 0: just the center
    yield (x0, y0)

    # Layers 1, 2, 3, ...
    n = 1
    while True:
        # Right edge, going up: (n, -(n-1)) → (n, n)
        for y in range(-(n - 1), n + 1):
            yield (x0 + n * step_x, y0 + y * step_y)

        # Top edge, going left: (n-1, n) → (-n, n)
        for x in range(n - 1, -n - 1, -1):
            yield (x0 + x * step_x, y0 + n * step_y)

        # Left edge, going down: (-n, n-1) → (-n, -n)
        for y in range(n - 1, -n - 1, -1):
            yield (x0 - n * step_x, y0 + y * step_y)

        # Bottom edge, going right: (-n+1, -n) → (n, -n)
        for x in range(-n + 1, n + 1):
            yield (x0 + x * step_x, y0 - n * step_y)

        n += 1


def grid_positions(position, fov, nx=2, ny=2, spacing=1.0):
    """
    compute a grid of stage positions centered on the given position

    Parameters
    ----------
    position: (x, y)
        center of the grid
    fov: (fov_x, fov_y)
        field of view per axis
    nx, ny: int
        number of grid positions in x and y
    spacing: float
        distance between neighboring positions in units of FOV size:
        1 -> touching (non-overlapping) FOVs,
        <1 -> overlapping FOVs,
        >1 -> non-overlapping FOVs with a gap

    Returns
    -------
    positions: list of 2-tuples
        (x, y) stage positions, row-major order
    """
    fov_x, fov_y = fov
    x0, y0 = position
    step_x = spacing * fov_x
    step_y = spacing * fov_y

    return [(x0 + (i - (nx - 1) / 2) * step_x,
             y0 + (j - (ny - 1) / 2) * step_y)
            for j in range(ny) for i in range(nx)]


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
