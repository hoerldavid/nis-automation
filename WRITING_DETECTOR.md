# Writing your own detector

This guide shows how to write a custom **detector** for the autoFRAP
pipeline: a single Python file that tells the pipeline
*which cells to pick* in each survey image and *which part of each
cell* to photobleach.

For most experiments, pieces of logic already exists in `autofrap/core/` - writing a detector for your experiment is mostly **choosing and combining parts**, not writing code from scratch.

## 1. What the pipeline expects

A detector is one Python file that defines one function called `detection_fun`:

```python
def detection_fun(survey_file_path):
    # survey_file_path: path to one FOV's survey image (nd2 file)
    ...
    return labels # or (labels, mask) or (labels, mask, viz)
```

The return values (positional, in this order):

* `labels` — required. A 2D integer array the size of the image, with
  `0` = background and `1..N` = the N cells/nuclei you picked.
* `mask` — optional. A 2D boolean array of the same size, marking the
  pixels to photobleach. Should be one connected component per cell/nucleus. Omit it to bleach the whole cell/nucleus.
* `viz` — optional. A 2D grayscale or RGB(A) image, used as the
  background of the per-FOV QC overlay.

The pipeline loads your file and calls `detection_fun` on the survey image(s) taken at every FOV. It will then run FRAP on the stimulation region of one cell/nucleus. If you run more than one cycle per FOV, the detector is run again and the pipeline takes care not to FRAP the same cell twice (by keeping track of the positions of already-FRAPed objects).

To use a detector file, pass the file via the `--detector` flag:

```
python -m autofrap.pipeline --detector path/to/your_detector.py ...
```

## 2. `build_detector`: assemble instead of writing

Writing `detection_fun` from scratch can be tedious. Therefore, we offer `build_detector()` to assemble a `detection_fun` from parts: you pass in the functions for
loading, detecting, and (optionally) filtering, masking and
visualizing, and it glues them together in the right order and applies
some housekeeping and checks.

### What happens at each step

When the pipeline calls the `detection_fun` you assembled, the parts
run in this order:

1. **load** — `load_fun(survey_file)` reads the channel(s) you chose
   from the survey file and returns the image (2D `(y, x)` or
   multi-channel `(c, y, x)`).
2. **detect** — `detector_fun(image)` finds the cells/nuclei and
   returns the labels (2D integer, `0` = background, `1..N` =
   cells/nuclei).
3. **filter** (optional) — `filter_function(labels, image)` returns
   the label IDs to keep; all other cells are removed. (Pass
   `filter_function=...` to `build_detector`.)
4. **housekeeping** (automatic, not a part) — cells touching the image
   border are discarded, and the remaining cells are renumbered. By
   default the numbering follows the distance to the image center, so
   the closest cell is bleached first (`relabel='distance'`; also
   `'shuffle'` or `None`).
5. **FRAP region** (optional) — `stim_mask_fun(labels, image)` marks
   the pixels to photobleach and should return a 2D binary mask of FRAP regions. It receives the labels *after*
   housekeeping. Omit it to bleach the whole cell/nucleus.
6. **QC picture** (optional) — `visualization_fun(image)` produces the
   background picture of the per-FOV QC overlay, so you can check
   later where the cells were picked and what was bleached. Omit it
   for an automatic default (grayscale for 2D, RGB composite for
   multi-channel), or pass `False` to draw on a blank canvas.

The result is returned as `labels` (, `mask` (, `viz`)) and the
pipeline takes over from there (bleaching, QC overlay, keeping track
of already-bleached cells across cycles). Note that this whole
sequence is re-run for *every cycle* of every FOV — e.g. a random
bleach position is redrawn each cycle.

The concrete functions for each part are listed in the next section;
all of them are optional except `load_fun` and `detector_fun`.

### Minimal working example

A complete detector file (also usable as a template):

```python
from functools import partial

from autofrap.io.nd2 import read_channel
from autofrap.core.detection import build_detector
from autofrap.core.image.segmentation import remote_detect_objects
from autofrap.core.image.mask import half_object_stim_mask

detection_fun = build_detector(
    load_fun=partial(read_channel, channel=0),          # step 1: which channel(s) to read
    detector_fun=partial(remote_detect_objects,         # step 2: how to find the cells
                         server_url='http://10.163.69.12:8000'),
    # step 2 alternative without a server (local Otsu + watershed
    # nucleus segmentation, no dependencies):
    #   from autofrap.core.image.segmentation import segment_nuclei_otsu_watershed
    #   detector_fun=segment_nuclei_otsu_watershed
    stim_mask_fun=lambda labels, image:                 # step 5: which part to bleach
        half_object_stim_mask(labels),
)
```

That's it — 12 lines. Run it with
`python -m autofrap.pipeline --detector this_file.py --nx 1 --ny 1`.

Reading it back against the steps above:

* **step 1 (`load_fun`)** — reads channel 0 of each survey image.
* **step 2 (`detector_fun`)** — runs Cellpose (on the lab GPU server) to find cells/nuclei.
* **step 5 (`stim_mask_fun`)** — bleaches the left half of each cell.
* step 3 (filter) is not used, so all cells/nuclei detected by Cellpose are candidates for FRAP and no `visualization_fun` is given → the automatic default picture is used for the QC overlay.

The built-in files in `autofrap/detectors/` (e.g.
`cellpose_remote_cluster_modular.py`) are real-world variants of this
template — read one when you want a copy-paste starting point.

## 3. The building blocks

Everything lives under `autofrap/core/` (and `autofrap/io/`). Full
options are documented in the docstrings of each function
(`help(function_name)` in a Python console, or just open the file).

### Loading the survey image

| function | what it does |
|---|---|
| `autofrap.io.nd2.read_channel(file, channel=...)` | reads one (`channel=0`), several (`channel=[0, 2]`), or all (`channel='all'`) channels → 2D `(y, x)` or `(c, y, x)` array. `z_projection='max'` max-projects z-stacks |

**Rule of thumb:** load a single channel if all other steps use that
channel; otherwise load all (`channel='all'`) and let the other steps
pick whichever channel they need (via their `channel` parameter or
`--detector-arg`, see below).

### Detecting cells/nuclei

| function | what it does | typical options |
|---|---|---|
| `autofrap.core.image.segmentation.dummy_detect_objects` | fake detector (one circle + one rectangle) — for testing your file without a real model | none |
| `autofrap.core.image.segmentation.remote_detect_objects` | Cellpose on the lab GPU server | `server_url=...`, `diameter=<cell width in px>`, `min_size=...` |
| `autofrap.core.image.segmentation.segment_nuclei_otsu_watershed` | local, dependency-free Otsu + watershed nucleus segmentation | `cell_sigma` (approximate nucleus radius in px), `otsu_frac` (threshold strictness) |

(The Otsu + watershed detector has more knobs in `detect_objects` / `SimpleSegParams`; the wrapper above exposes the three you will actually tune.)

### Choosing the FRAP region (the "mask")

| choice | what gets bleached |
|---|---|
| omit `stim_mask_fun` | the **whole cell/nucleus** |
| `autofrap.core.image.mask.half_object_stim_mask` | the **left half** of each cell/nucleus (the standard choice) |
| `autofrap.core.image.mask.random_circle_stim_mask` | one **random circle** inside each cell/nucleus (`area_fraction=0.25` → 25% of its area) |
| `autofrap.core.image.mask.cluster_stim_mask` | **bright subcellular clusters** within each cell/nucleus (e.g. a punctate channel; `channel=`, `min_cluster_area=`, `contrast=` — those without a cluster are skipped) |

### Keeping only "good" cells (optional filter)

Pass a `filter_function` that maps `(labels, image)` to the list of
label IDs to keep; labels not in the list are dropped. Ready-made:

| function | keeps cells that… |
|---|---|
| `autofrap.core.image.mask.filter_intensity_inside` | have enough **marker intensity inside** the cell/nucleus (`channel=`, `threshold=`, `metric='mean'` or `'median'`) |
| `autofrap.core.image.mask.filter_intensity_surround` | have a bright **ring around** the cell/nucleus (edge/membrane markers) |

You can also write your own filter freely — it receives the label map
and the loaded image, so anything computable per cell/nucleus works
(size, shape, intensity profile, …). Example from the `build_detector`
docstring:

```python
from skimage.measure import regionprops

def filter_big_enough(labels, image):
    return [rp.label for rp in regionprops(labels)
            if rp.area > 1000]
```

### QC picture (optional visualization)

The QC overlay (`save_qc_overlay` per FOV) shows where each cell was
picked and which part was bleached. The background picture comes from:

| choice | result |
|---|---|
| omit `visualization_fun` | automatic: grayscale for 2D, RGB composite for multi-channel |
| `lambda image: image` / `lambda image: image[0]` | show the image as-is / one specific channel |
| `False` | no background picture (cells drawn on blank canvas) |

This part is cosmetic — if it fails, the pipeline warns and continues.

## 4. Tuning without editing the file

Values can also be set at run time with `--detector-arg key=value`
instead of editing the file. How the keys reach your functions is
controlled by the `parameter_map` setting of `build_detector`. Use the
explicit form: a dict listing, per step, which `--detector-arg` keys
are passed to it (and under which internal name):

```python
detection_fun = build_detector(
    load_fun=read_channel,                        # reads channel 0 by default
    detector_fun=remote_detect_objects,
    stim_mask_fun=cluster_stim_mask,
    parameter_map={
        'load_fun':        {'load_channel': 'channel'},   # -> read_channel(channel=...)
        'detector_fun':    {'det_channel': 'channel',     # -> remote_detect_objects(channel=...)
                            'server_url': 'server_url',
                            'diameter': 'diameter'},
        'stim_mask_fun':   {'mask_channel': 'channel'},   # -> cluster_stim_mask(channel=...)
    })
```

```
python -m autofrap.pipeline --detector your_detector.py \
    --detector-arg load_channel=all \
    --detector-arg det_channel=0 --detector-arg mask_channel=1 \
    --detector-arg diameter=70
```

`load_channel=all` is essential here: the loaded image is passed on to
all later steps, so it must contain every channel those steps use (0
for detection, 1 for the mask) — without it, only channel 0 would be
loaded and selecting channel 1 would fail. This is also how you tune,
e.g., the Cellpose `diameter` for a new sample without touching the
file — and the renaming is what makes detect-in-channel-0,
mask-in-channel-1 possible: all three functions have a `channel`
parameter, but each `--detector-arg` key is routed to exactly one step. (Without any `parameter_map`, `--detector-arg`
keys are not routed anywhere.)

## 5. Testing your detector before a live run

Every detector file can test itself on a single survey image:

```
python your_detector.py path/to/survey.nd2
```

(add the `if __name__ == '__main__':` block from any file in
`autofrap/detectors/` — they all have one). For a visual check without
the pipeline, use the dummy detector to validate your file's plumbing,
or run the pipeline with `--nx 1 --ny 1` on one FOV and look at the QC
overlay.

## 6. What the pipeline checks for you

So you don't have to validate everything yourself, `build_detector`
enforces the contract:

* **labels** must be a 2D integer array, same size as the image
  (`0` = background, `1..N` = cells/nuclei) — anything else raises an
  error with a clear message.
* **mask** must be a 2D boolean array of the same size; a
  cell/nucleus with more than one disconnected bleach region triggers a
  warning (the largest region is used).
* a cell/nucleus with *no* bleach region is allowed — it is simply
  skipped.
* border-touching cells/nuclei are removed automatically; the
  remaining cells are renumbered by distance to the image center
  (nearest bleached first).

## 7. When the modular approach isn't enough

`build_detector` covers everything that fits the
`image → labels (+ mask / picture)` shape. If you need something
different (e.g. a picture that depends on the detected labels, or an
input that isn't a survey nd2 file), write one plain function in your
detector file:

```python
def detection_fun(survey_file):
    ...
    return labels               # or (labels, mask) or (labels, mask, viz)
```

and the pipeline will use it directly. Look at
`autofrap/detectors/example_detector.py` and the
`autofrap.core.detection` module docstring for the exact contract.
