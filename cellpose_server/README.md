# Cellpose inference server

FastAPI server that runs cellpose model inference on a separate machine
(ideally with a GPU; Apple Silicon works too). The microscope-side
detectors (`autofrap/detectors/cellpose_remote_*.py`) are its clients:
they POST one survey image and get back a label map.

The server imports nothing from the `autofrap` package — it is a
self-contained deployment unit: copy this directory to the server
machine and run it there.

## Setup

Install **torch first**. Installing the requirements (cellpose) below would transiently install it as well but it's a good idea to install a package that explicitly matches your GPU environment (e.g. to prevent CUDA version mismatches):

```bash
# pick the wheel for your CUDA version, see e.g. https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu126   # example: CUDA 12.6
```

Then install the server dependencies:

```bash
pip install -r requirements.txt      # fastapi, uvicorn, cellpose, DINOv3
```

Pretrained model weights will be downloaded on first model load.

## Run

```bash
python cellpose_server.py --model cpdino-vitb --host 0.0.0.0 --port 8000
```

The `--model`, `--host` and `--port` arguments in the example are the defaults, so you could also leave them out (or adjust).

Device: `--device auto` (default) picks cuda, then mps (Apple Silicon),
then cpu; override with `--device cuda` / `mps` / `cpu` / `cuda:N`
(e.g. `cuda:1` to pin a specific GPU on a multi-GPU host). An
out-of-range index or an unavailable backend exits with a clear error
(code 2) before the model loads.

## Client side

The detector files default to `DEFAULT_CELLPOSE_SERVER_URL` (defined at
the top of each `cellpose_remote_*` detector file — edit it there if the
server moves), or pass it per run with
`--detector-arg server_url=http://<server>:8000`.

Plain HTTP, no auth: intended for a trusted lab network. Add a token
check (FastAPI dependency on `/detect`) if it ever leaves that network.

## Endpoints

Full contract in the module docstring of `cellpose_server.py`:

    POST /detect    one numpy array, any shape (np.save bytes) -> label map
                    (query param eval_kwargs: JSON object with any
                    model.eval() knob, e.g. ?eval_kwargs={"diameter": 30};
                    accepted names listed by /health)
    GET  /health    {"status": "ok", "model": "<name>", "eval_params": [...]}
