"""
cellpose inference server (runs on a separate, ideally GPU, machine).

Endpoints:

    POST /detect
        request body  : one numpy array, any shape, serialized with
                        np.save (Content-Type: application/x-numpy);
                        2-D for plain cellpose, 3-D with do_3D in
                        eval_kwargs for volumetric input
        query params  : eval_kwargs - a JSON object holding any
                        CellposeModel.eval() keyword arguments (all
                        optional; cellpose's own defaults apply), e.g.
                        ?eval_kwargs={"diameter": 30, "min_size": 50,
                                      "normalize": {"percentile": [1, 99]}}
                        'progress' (a live object) and compute_masks=False
                        (no label map would come back) are rejected with
                        a 400, as are unknown keys; /health lists the
                        accepted parameter names
        response body : the cellpose label map (numpy array, same
                        serialization); extras in headers
                        (X-Inference-Time-S, X-N-Objects, X-Shape)
    GET /health
        {"status": "ok", "model": "<name>", "cuda": ..., "device": ...,
         "eval_params": [<accepted model.eval() kwarg names>]}

The model is loaded once at startup; inference requests are serialized
with a lock (single GPU).

Wire format: raw np.save bytes (magic + shape/dtype header + data).
A 1024x1024 uint16 image is ~2 MB - no compression needed on a LAN.

Plain HTTP, no auth: intended for a trusted lab network. Add a token
check (FastAPI dependency on /detect) if it ever leaves that network.

Setup and run: see README.md in this directory (requirements.txt holds
the pip dependencies; install a CUDA-matched torch first, and dinov3
for the default model).
"""
import argparse
import inspect
import io
import json
import re
import threading
import time

import numpy as np
import torch
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import Response


_DEVICE_RE = re.compile(r'^(auto|cpu|mps|cuda(?::\d+)?)$')


def _device_arg(s):
    """argparse type for --device: any cuda index is allowed (cuda, cuda:0,
    cuda:1, ...) so multi-GPU hosts are not limited to a fixed list"""
    if not _DEVICE_RE.match(s):
        raise argparse.ArgumentTypeError(
            f"invalid device {s!r}: expected 'auto', 'cpu', 'mps' or 'cuda[:N]'")
    return s


def _resolve_device(device):
    """resolve 'auto' to the best available torch device; validate an
    explicit cuda index against the devices torch actually sees"""
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)
    if device.type == 'cuda':
        n = torch.cuda.device_count()
        if n == 0:
            raise ValueError('CUDA device requested, but torch sees no CUDA devices')
        if device.index is not None and device.index >= n:
            avail = ', '.join(f'cuda:{i} ({torch.cuda.get_device_name(i)})'
                              for i in range(n))
            raise ValueError(f'cuda:{device.index} does not exist; available: {avail}')
    elif device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS device requested, but torch reports MPS is not available')
    return device


def create_app(model_name: str, device: str = 'auto') -> FastAPI:
    import cellpose.models as models

    # explicit device: 'auto' = cuda if available, then mps (Apple
    # Silicon), then cpu
    device = _resolve_device(device)

    app = FastAPI(title='cellpose inference server')
    model = models.CellposeModel(pretrained_model=model_name, device=device)  # once, at startup
    lock = threading.Lock()  # serialize inference (single device)
    # accepted model.eval() kwargs, for /health (excluding self and the image)
    eval_params = sorted(p for p in inspect.signature(models.CellposeModel.eval).parameters
                         if p not in ('self', 'x'))

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': model_name,
                'cuda': device.type == 'cuda',
                'device': (torch.cuda.get_device_name(device)
                           if device.type == 'cuda' else device.type),
                'eval_params': eval_params}

    @app.post('/detect')
    def detect(image: bytes = Body(..., description='np.save-serialized numpy array'),
               eval_kwargs: str = Query(None,
                                        description='JSON object of CellposeModel.eval() '
                                                    'keyword arguments')):
        # sync def on purpose: FastAPI runs it in a worker thread, so the
        # blocking model.eval doesn't stall the event loop
        # (explicit Body(...): a bare `bytes` annotation is read as a query
        #  param in current FastAPI versions)
        arr = np.load(io.BytesIO(image), allow_pickle=False)

        if eval_kwargs is None:
            kwargs = {}
        else:
            try:
                kwargs = json.loads(eval_kwargs)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f'eval_kwargs is not valid JSON: {e}')
            if not isinstance(kwargs, dict):
                raise HTTPException(400, 'eval_kwargs must be a JSON object')
        if 'progress' in kwargs:
            raise HTTPException(400, "'progress' is a live object and cannot be "
                                     'passed over HTTP')
        if kwargs.get('compute_masks') is False:
            raise HTTPException(400, 'compute_masks=False would return no label map')

        t0 = time.time()
        with lock:
            try:
                masks, flows, styles = model.eval(arr, **kwargs)
            except (TypeError, ValueError) as e:
                # unknown kwarg or invalid parameter combination -> client error
                raise HTTPException(400, f'bad eval_kwargs: {e}')
        dt = time.time() - t0
        n_obj = len(np.unique(masks)) - 1

        buf = io.BytesIO()
        np.save(buf, masks)
        return Response(
            content=buf.getvalue(),
            media_type='application/x-numpy',
            headers={'X-Inference-Time-S': f'{dt:.3f}',
                     'X-N-Objects': str(int(n_obj)),
                     'X-Shape': ','.join(map(str, masks.shape))},
        )

    return app


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='cpdino-vitb',
                   help='cellpose pretrained model name')
    p.add_argument('--device', default='auto', type=_device_arg, metavar='DEVICE',
                   help="torch device: 'auto', 'cpu', 'mps' or 'cuda[:N]' "
                        "(e.g. cuda:1 for the second GPU); 'auto' = cuda if "
                        "available, then mps (Apple Silicon), then cpu "
                        "[default: %(default)s]")
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    a = p.parse_args()

    try:
        _resolve_device(a.device)  # fail fast on a bad device, before the model loads
    except ValueError as e:
        p.error(str(e))

    import uvicorn
    uvicorn.run(create_app(a.model, a.device), host=a.host, port=a.port)
