"""
Pure grid-geometry helpers for tiled acquisitions (no NIS / hardware
dependency).

Moved out of nis_util.py: gen_grid is not NIS-specific; it is used by
the old wing-scanner code at the repo root (automation.py,
NIS_Macro_Acquisition.ipynb).
"""
from math import ceil


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
