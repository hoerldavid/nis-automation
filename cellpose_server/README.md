# Cellpose inference server

FastAPI server that runs cellpose model inference on a separate machine
(ideally with a GPU; Apple Silicon works too). The microscope-side
detectors (`autofrap/detectors/cellpose_remote_*.py`) are its clients:
they POST one survey image and get back a label map.

The server imports nothing from the `autofrap` package — it is a
self-contained deployment unit: copy this directory to the server
machine and run it there.

## Setup

Install a **CUDA-matched torch first**. cellpose depends on torch, and
pip will install a default build if it is missing — that build does not
always match your CUDA version:

```bash
# pick the wheel for your CUDA version, see
# https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124   # example: CUDA 12.4
```

Then the server dependencies:

```bash
pip install -r requirements.txt      # fastapi, uvicorn, cellpose
```

We default to using the `cpdino-vitb` model, which is smaller and faster than cellpose's default SAM. To use this DINOv3-based model, we need one more
package that is not on PyPI:

```bash
pip install git+https://github.com/facebookresearch/dinov3
```

Pretrained weights are downloaded on first model load, or copy the local
model cache over to skip the download.

## Run

```bash
python cellpose_server.py --model cpdino-vitb --host 0.0.0.0 --port 8000
```

Device: `--device auto` (default) picks cuda, then mps (Apple Silicon),
then cpu; override with `--device cuda` / `mps` / `cpu`. (mps support
depends on the installed cellpose version — if a model errors on mps,
retry with `--device cpu` or a newer cellpose.)

## Client side

The detector files default to `DEFAULT_CELLPOSE_SERVER_URL` (defined at
the top of each `cellpose_remote_*` detector file — edit it there if the
server moves), or pass it per run with
`--detector-arg server_url=http://<server>:8000`.

Plain HTTP, no auth: intended for a trusted lab network. Add a token
check (FastAPI dependency on `/detect`) if it ever leaves that network.

## Endpoints

Full contract in the module docstring of `cellpose_server.py`:

    POST /detect    one 2-D numpy array (np.save bytes) -> label map
                    (query params: the model.eval() knobs)
    GET  /health    {"status": "ok", "model": "<name>"}
