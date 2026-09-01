"""
cellpose inference server (runs on a separate, ideally GPU, machine).

Endpoints:

    POST /detect
        request body  : one 2D numpy array, serialized with np.save
                        (Content-Type: application/x-numpy)
        query params  : main model.eval() knobs, all optional (defaults
                        in brackets = cellpose's own defaults):
                          diameter          (None: estimate from image)
                          min_size          (15) - discard smaller objects [px]
                          cellprob_threshold (0.0) - discard objects with lower
                                                     cell probability
                          flow_threshold    (0.4) - discard objects with lower
                                                     flow coherence
                          max_size_fraction (0.4) - discard objects larger
                                                     than this fraction of the image
        response body : the cellpose label map (2D numpy array), same
                        serialization; extras in headers
                        (X-Inference-Time-S, X-N-Objects)
    GET /health
        {"status": "ok", "model": "<name>"}

The model is loaded once at startup; inference requests are serialized
with a lock (single GPU).

Setup on the server machine:
    pip install fastapi uvicorn cellpose
    # pretrained weights are downloaded on first model load, or copy
    # the local model cache over to skip the download

Run:
    python cellpose_server.py --model cpdino-vitb --host 0.0.0.0 --port 8000

Wire format: raw np.save bytes (magic + shape/dtype header + data).
A 1024x1024 uint16 image is ~2 MB - no compression needed on a LAN.

Plain HTTP, no auth: intended for a trusted lab network. Add a token
check (FastAPI dependency on /detect) if it ever leaves that network.
"""
import argparse
import io
import threading
import time

import numpy as np
import torch
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import Response


def create_app(model_name: str) -> FastAPI:
    import cellpose.models as models

    # explicit device: GPU if available, CPU otherwise
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    app = FastAPI(title='cellpose inference server')
    model = models.CellposeModel(pretrained_model=model_name, device=device)  # once, at startup
    lock = threading.Lock()  # serialize inference (single GPU)

    @app.get('/health')
    def health():
        cuda = torch.cuda.is_available()
        return {'status': 'ok', 'model': model_name,
                'cuda': cuda,
                'device': torch.cuda.get_device_name(0) if cuda else 'cpu'}

    @app.post('/detect')
    def detect(image: bytes = Body(..., description='np.save-serialized 2-D array'),
               diameter: float = Query(None,
                                       description='expected object diameter [px]; '
                                                   'null = estimate from image'),
               min_size: int = Query(15, ge=0,
                                     description='discard objects smaller than this [px]'),
               cellprob_threshold: float = Query(0.0, ge=0.0,
                                                 description='discard objects with '
                                                             'lower cell probability'),
               flow_threshold: float = Query(0.4, ge=0.0,
                                             description='discard objects with lower '
                                                         'flow coherence'),
               max_size_fraction: float = Query(0.4, gt=0.0, le=1.0,
                                                description='discard objects larger '
                                                            'than this fraction of '
                                                            'the image')):
        # sync def on purpose: FastAPI runs it in a worker thread, so the
        # blocking model.eval doesn't stall the event loop
        # (explicit Body(...): a bare `bytes` annotation is read as a query
        #  param in current FastAPI versions)
        arr = np.load(io.BytesIO(image), allow_pickle=False)
        if arr.ndim != 2:
            raise HTTPException(400, f'expected a 2-D array, got {arr.ndim}-D')

        eval_kwargs = dict(min_size=min_size,
                           cellprob_threshold=cellprob_threshold,
                           flow_threshold=flow_threshold,
                           max_size_fraction=max_size_fraction)
        if diameter is not None:
            eval_kwargs['diameter'] = diameter

        t0 = time.time()
        with lock:
            masks, flows, styles = model.eval(arr, **eval_kwargs)
        dt = time.time() - t0
        n_obj = len(np.unique(masks)) - 1

        buf = io.BytesIO()
        np.save(buf, masks)
        return Response(
            content=buf.getvalue(),
            media_type='application/x-numpy',
            headers={'X-Inference-Time-S': f'{dt:.3f}',
                     'X-N-Objects': str(int(n_obj))},
        )

    return app


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='cpdino-vitb',
                   help='cellpose pretrained model name')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    a = p.parse_args()

    import uvicorn
    uvicorn.run(create_app(a.model), host=a.host, port=a.port)
