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
    writes e.g. (C, Y, X) and (Z, C, Y, X)).

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

        # Determine channel selection
        if channel == 'all':
            if 'C' in axes:
                num_c = f.sizes['C']
                channels = list(range(num_c))
            else:
                # single-channel file: will be promoted to (1,Y,X) after extraction
                channels = [0]
        elif isinstance(channel, (tuple, list)):
            channels = list(channel)
        else:
            channels = [channel]

        if 'C' not in axes:
            if channels != [0]:
                raise ValueError(
                    f'{nd2_file}: single-channel file (no C dimension); '
                    'pass channel=0 or channel=\'all\'')

        arr = _extract(f.asarray(), axes, channels, z_projection)

        # Promote to (C,Y,X) for channel='all' on single-channel files
        if channel == 'all' and 'C' not in axes:
            return arr[None, ...]
        return arr


def _extract(arr, axes, channels, z_projection):
    """
    axis-order-independent extraction on a validated survey array
    (axes ⊆ {C, Z, Y, X}): normalize to (C, Z, Y, X) as present,
    select the channel(s) (a single index gives a 2D result), project
    Z if requested. (Factored out for testing the axis logic without
    a matching nd2 file.)
    """
    order = [a for a in ('C', 'Z', 'Y', 'X') if a in axes]
    arr = np.transpose(arr, [axes.index(a) for a in order])
    z_axis = 0
    if 'C' in axes:
        if len(channels) > 1:
            arr = arr[channels]  # (k, Z?, Y, X)
            z_axis = 1
        else:
            arr = arr[channels[0]]  # (Z?, Y, X)
    if 'Z' in axes and z_projection == 'max':
        arr = arr.max(axis=z_axis)
    return arr


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
