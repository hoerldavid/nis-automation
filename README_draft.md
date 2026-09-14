# autoFRAP – NIS-Elements automation

autoFRAP automates multi-FOV, multi-cycle FRAP experiments on Nikon microscopes controlled by NIS Elements. It acquires a survey image, detects objects, selects one object per cycle, creates a stimulation mask for that object and acquires the FRAP time series. See DESIGN_GOALS_AUTOFRAP.md for the full workflow – the framework supports more than half-nucleus stimulation, e.g. organelle detection, expression filtering, random circles, clusters, etc.

## Install

```bash
# clone
git clone <repo-url>
cd autofrap

# editable install with dependencies
pip install -e .
```

### Microscope workstation

This is where the pipeline runs.

```bash
export CELLPOSE_SERVER_URL=http://<server>:8000  # only if using Cellpose remote detector
python -m autofrap.pipeline --help
```

### Cellpose server

Run on a separate machine with GPU / Apple Silicon.

```bash
pip install cellpose
python /path/to/repo/cellpose_server.py --device auto --host 0.0.0.0 --port 8000
```

`cellpose_server.py` lives in the repository root. The built-in Cellpose detectors read `CELLPOSE_SERVER_URL` at import time; set it on the workstation.
```

Dependencies are declared in `pyproject.toml`. The package provides shims for legacy bitsnpieces imports.

## Quick start – dry run with FakeNIS

No microscope required:

```bash
python autofrap/autofrap_bitsnpieces/dry_run_pipeline.py \
  --preset simple_seg \
  --out test_acquisitions/dry_run
```

Presets:
* `simple_seg` → `autofrap/detectors/simple_seg_detector.py` (Otsu + watershed, local, no server)
* `cellpose` → `autofrap/detectors/cellpose_remote_detector.py` with `diameter=70`

The dry run is designed to work with arbitrary existing ND2 files. The test script uses a hardcoded survey glob, but the pipeline itself accepts any source list. It copies source ND2s as surveys/FRAPs and writes QC PNGs.

## Run on the microscope

Prerequisites on the workstation:
* NIS Elements running, NIS-Elements Automation macro interface enabled
* `nis_ar` on PATH
* Python environment with the package installed

Example 2×2 grid, 3 cycles per FOV:

```bash
python -m autofrap.pipeline \
  --out /path/to/output \
  --nx 2 --ny 2 \
  --spacing 1.0 \
  --max-cycles 3 \
  --name grid_run \
  --detector autofrap/detectors/cellpose_remote_detector.py \
  --detector-arg diameter=70 \
  --detector-arg min_size=50
```

`--name` sets the experiment name; the run directory is created under `--out` with a timestamp prefix, e.g. `20260901_123456_grid_run/`. See `--help` for all options.

Key arguments:
* `--out` output root, per-FOV subdirs created automatically
* `--nx/--ny` grid size
* `--spacing` FOV spacing in FOV units
* `--max-cycles` cycles per FOV
* `--detector` path to detector module
* `--detector-arg key=value` forwarded to the detector via `build_detector`

### Spiral visit order

```bash
python -m autofrap.pipeline \
  --out /path/to/output \
  --spiral \
  --nx 5 --ny 5 \
  --spacing 1.0 \
  --max-cycles 3 \
  --name spiral_run \
  --detector autofrap/detectors/simple_seg_detector.py \
  --detector-arg cell_sigma=16.0 \
  --detector-arg otsu_frac=0.3 \
  --detector-arg min_eroded_extent=0.90
```

`--spiral N` generates a centre-out spiral via `grid_utils.spiral_positions`.

## Detectors

You provide a detector `.py` file that defines a `detection_fun`. The pipeline imports the file and calls `detection_fun(survey_file, **kwargs)` for each survey image.

The function must return labels and optionally a stimulation mask and a visualization image:

```
survey_file -> (labels,) or (labels, stim_mask) or (labels, stim_mask, viz)
```

`labels` is a 2D int array, `stim_mask` is a 2D bool array with at most one connected region per cell, `viz` is a 2D grayscale or RGB(A) image for the QC overlay.

The built-in detectors are just examples of this contract. You can use the Cellpose remote detector or the local simple Otsu+watershed detector, or write your own.

`build_detector` composes load → detect → mask → viz and routes CLI parameters to sub-functions via `parameter_map`. With `parameter_map='auto'` each sub-function receives the arguments it can accept; explicit mappings can be provided to avoid name collisions, see `autofrap/core/detection.py`.

Built-in:
* `autofrap/detectors/dummy_detector.py` – labels only, for testing
* `autofrap/detectors/simple_seg_detector.py` – Otsu high-pass + watershed, local
* `autofrap/detectors/cellpose_remote_detector.py` – remote Cellpose server, half-nucleus stim mask
* `autofrap/detectors/cellpose_remote_halfnucleus_modular.py` – same as above built with `build_detector` and QC viz

Authoring a detector:
```python
from autofrap.detection import build_detector
from functools import partial

def load_fun(path): ...
def detect_fun(image, **kw): ...

detection_fun = build_detector(
    load_fun=load_fun,
    detector_fun=detect_fun,
    stim_mask_fun=half_object_stim_mask,
    visualization_fun=lambda img: img,
    parameter_map='auto'
)
```

## Output

Per FOV:
* `fovNN_cycleNN_survey.nd2`
* `fovNN_cycleNN_frap.nd2` – with `StandardROI` + `StimulationROI`
* `fovNN_cycleNN_survey_qc.png` – overlay with labels, FRAP mask, selected cell

The pipeline returns to start position on completion or abort.

## Notes

* Survey files can be multi-channel. `load_fun` returns `(c, y, x)` and detection functions can use any channel(s); the visualization is the only `(y, x, 3/4)` array.
* The microscope workstation must have both an ND-Acquisition template for the survey image and an ND-Stimulation template for the FRAP time series defined in NIS Elements GUI.
* For real runs, ensure the ND acquisition template is a single image with one or more channels, no Time/XY/Large Image tabs.

See `STATUS.md` for the full development log and open TODOs.
