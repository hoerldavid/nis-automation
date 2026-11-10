"""
Remote Cellpose detector.

Runs cellpose on a separate server via HTTP.
"""

import io
import json
import time

import numpy as np
import requests


def remote_detect_objects(image, server_url, timeout=60, retries=1,
                          channel=0, **eval_kwargs):
    """
    run cellpose on a remote server (see cellpose_server.py)

    The image is serialized with np.save and POSTed to the server's
    /detect endpoint; the label map comes back in the same format.
    A failed request (connection error, timeout, or HTTP error) is
    retried `retries` times with a short backoff before propagating.

    Parameters
    ----------
    image: 2D np.ndarray (y, x) or 3D np.ndarray (c, y, x)
        input image; if 3D, the channel is selected with `channel`.
        This client enforces the 2-D contract: the array sent to the
        server is always 2-D (the server itself accepts any shape)
    server_url: str
        base URL of the cellpose server, e.g. 'http://192.168.1.10:8000'
    timeout: float
        request timeout in seconds; the V100 server answers in ~2 s, so
        60 s leaves room for connection latency and queued requests
    retries: int
        number of retries after a failed request, with a 2 s backoff
    channel: int
        channel index to select from a (c, y, x) image; ignored for 2D
        input (assumes the correct channel was already loaded)
    eval_kwargs: dict
        optional cellpose model.eval() parameters, sent to the server as
        one JSON query param (any parameter, incl. dict-valued ones like
        normalize); accepted names and cellpose's own defaults are listed
        by the server's /health endpoint

    Returns
    -------
    labels: 2D np.ndarray (y, x), int32
        0 = background, 1..N = objects
    """
    # select channel if image is multi-channel
    if image.ndim == 3:
        if channel < 0 or channel >= image.shape[0]:
            raise ValueError(
                f'channel {channel} out of bounds for image with {image.shape[0]} channels')
        image = image[channel]
    elif image.ndim != 2:
        raise ValueError(
            f'remote_detect_objects expects 2D (y, x) or 3D (c, y, x) image, got {image.ndim}D')

    buf = io.BytesIO()
    np.save(buf, image)
    params = {'eval_kwargs': json.dumps(eval_kwargs)} if eval_kwargs else None
    for attempt in range(retries + 1):
        try:
            r = requests.post(f'{server_url}/detect', data=buf.getvalue(),
                              headers={'Content-Type': 'application/x-numpy'},
                              params=params,
                              timeout=timeout)
            r.raise_for_status()
            break
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            if attempt == retries:
                raise
            time.sleep(2.0 * (attempt + 1))
    labels = np.load(io.BytesIO(r.content), allow_pickle=False)
    print(f'remote detection: {r.headers.get("X-Inference-Time-S", "?")} s '
          f'({r.headers.get("X-N-Objects", "?")} objects) on {server_url}')

    # CP4 returns uint16 masks; the rest of the pipeline uses int32
    # TODO: this is unlikely to cause problems, stick to uint16?
    return np.ascontiguousarray(labels, dtype=np.int32)
