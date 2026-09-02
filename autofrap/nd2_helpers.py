"""
Helpers for reading ND2 files (nd2 library).
"""
import nd2


def read_channel(nd2_file, channel=0):
    """
    read one channel of an ND2 file as a 2D (y, x) array

    Parameters
    ----------
    nd2_file: str
        path to the ND2 file
    channel: int
        channel index to read

    Returns
    -------
    image: np.ndarray
        2D image array (y, x)
    """
    with nd2.ND2File(nd2_file) as f:
        return f.asarray()[channel]


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
