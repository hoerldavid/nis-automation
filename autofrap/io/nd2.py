"""
Helpers for reading ND2 files (nd2 library).
"""
import numpy as np

import nd2


def read_channel(nd2_file, channel=0, z_projection=None):
    """
    read one or more channels of a survey ND2 file

    Survey images only: the supported dimensions are C, Y, X (and Z
    when a projection is requested). Files with other dimensions
    (T, P, ...) raise ValueError - FRAP time series etc. belong to
    downstream analysis, not the acquisition pipeline.

    Channel selection is independent of the file's axis order (NIS
    writes e.g. (C, Y, X) and (Z, C, Y, X)): the file is loaded as an
    axis-labeled xarray (ND2File.to_xarray) and selected by dimension
    name, so nd2/xarray own the axis handling.

    Parameters
    ----------
    nd2_file: str
        path to the ND2 file
    channel: int, tuple of int, or 'all'
        one channel index -> 2D (y, x) array; several channel
        indices -> (k, y, x) array in the given order. 'all' loads all
        channels and returns (C, Y, X), promoting single-channel files
        to (1, Y, X). NIS omits the C dimension of single-channel files:
        pass channel=0 for those. Default is 0.
    z_projection: None or 'max'
        None (default): files with a Z dimension raise ValueError;
        'max': max projection along Z (per channel)

    Returns
    -------
    image: np.ndarray
        2D (y, x) for a single channel, (k, y, x) for multiple,
        (C, y, x) for channel='all'

    Raises
    ------
    ValueError
        on dimensions other than C/Y/X(/Z), a Z dimension without
        z_projection, or a channel selection that does not fit the
        file (e.g. a channel index for a single-channel file)
    """
    if z_projection not in (None, 'max'):
        raise ValueError(f'unknown z_projection {z_projection!r} '
                         '(use None or \'max\')')

    with nd2.ND2File(nd2_file) as f:
        axes = list(f.sizes)  # axis names in array axis order
        unexpected = set(axes) - {'C', 'Y', 'X', 'Z'}
        if unexpected:
            raise ValueError(
                f'{nd2_file}: unsupported dimensions {sorted(unexpected)} '
                f'(axes: {axes}); read_channel reads survey images '
                '(C/Y/X, Z with a projection) - time series etc. belong '
                'to downstream analysis')
        if 'Z' in axes and z_projection is None:
            raise ValueError(
                f'{nd2_file}: file has a Z dimension (axes: {axes}); '
                "pass z_projection='max' to project it")

        # axis-labeled array; squeeze=True (the default) drops singleton
        # dims (e.g. a size-1 C), so the channel logic below keys on the
        # post-squeeze dims rather than on f.sizes
        arr = f.to_xarray(delayed=False)

        if channel == 'all':
            channels = (list(range(arr.sizes['C'])) if 'C' in arr.dims
                        else [0])
        elif isinstance(channel, (tuple, list)):
            channels = list(channel)
        else:
            channels = [channel]

        if 'C' in arr.dims:
            # one index -> 2D (y, x); several -> (k, y, x) in order
            arr = arr.isel(C=channels[0] if len(channels) == 1
                           else channels)
        elif channels != [0]:
            raise ValueError(
                f'{nd2_file}: single-channel file (no C dimension); '
                "pass channel=0 or channel='all'")

        if 'Z' in arr.dims and z_projection == 'max':
            arr = arr.max(dim='Z')

        image = np.asarray(arr)
        if channel == 'all' and 'C' not in arr.dims:
            return image[None, ...]  # promote (Y, X) -> (1, Y, X)
        return image


# TODO: not really needed - remove?
def stage_position(nd2_file):
    """
    read the (x, y, z) stage position [um] recorded in the ND2 metadata

    NIS writes the coarse XY stage + Z position into the per-frame
    metadata (dXPos/dYPos/dZPos); nd2 exposes it as
    frame_metadata(...).channels[...].position.stagePositionUm. One
    value per file (first frame; the stage does not move within a file).

    Gotcha: the raw metadata also contains per-channel "XY device"
    slots (pDeviceSetting m_iXYUseN/m_sXYKeyN/m_dXYPositionXN). Those
    are *not* the stage position: slot 0 is unused and holds stale
    values, and the only in-use slot on this microscope is 'XYDrive'
    (the Ti XY piezo), whose position hovers around 0.

    Parameters
    ----------
    nd2_file: str
        path to the ND2 file

    Returns
    -------
    (x, y, z): tuple of float
        stage position in um
    """
    with nd2.ND2File(nd2_file) as f:
        p = f.frame_metadata(0).channels[0].position.stagePositionUm
        return (p.x, p.y, p.z)
