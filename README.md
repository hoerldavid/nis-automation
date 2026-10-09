# autoFRAP – NIS-Elements automation

autoFRAP automates multi-FOV, multi-cycle FRAP experiments on Nikon
microscopes controlled by NIS Elements. It acquires a survey image,
detects objects, selects one object per cycle, creates a stimulation
mask for that object and acquires the FRAP time series. See
DESIGN_GOALS_AUTOFRAP.md for the full workflow – the framework
supports more than half-nucleus stimulation, e.g. organelle
detection, expression filtering, random circles, clusters, etc.

## Install

```bash
# clone
git clone <repo-url>
cd autofrap

# editable install with dependencies (Python >= 3.9)
pip install -e .
```

Dependencies are declared in `pyproject.toml`.

### Microscope workstation

This is where the pipeline runs. Prerequisites: NIS Elements running
with the NIS-Elements Automation macro interface enabled, and a Python
environment with the package installed. The pipeline calls `nis_ar.exe`
(default: the standard NIS-Elements install path; override with
`--nis`).

```bash
python -m autofrap.pipeline --help
```

### Cellpose server (optional)

Only needed for the `cellpose_remote_*` detector files; all other
detectors run fully local. Run on a separate machine with GPU / Apple
Silicon — the server lives in `cellpose_server/` (self-contained:
script + `requirements.txt` + README; see `cellpose_server/README.md`
for setup, including the torch/CUDA and dinov3 notes):

```bash
python /path/to/repo/cellpose_server/cellpose_server.py --device auto --host 0.0.0.0 --port 8000
```

The detector files default to `DEFAULT_CELLPOSE_SERVER_URL` (defined
at the top of each file — edit it there if the server moves), or pass
it per run with `--detector-arg server_url=http://<server>:8000`.

## Quick start – dry run with FakeNIS

No microscope required:

```bash
python -m autofrap.pipeline.dry_run --preset simple_seg \
  --sources "/path/to/data/*survey.nd2"
```

Presets:
* `dummy` → `autofrap/detectors/example_detector.py` (dummy objects, for testing)
* `simple_seg` → `autofrap/detectors/simple_seg_detector.py` (Otsu + watershed, local, no server)
* `cellpose` → `autofrap/detectors/cellpose_remote_halfnucleus_modular.py` with `diameter=70` (needs the server)

The dry run is designed to work with arbitrary existing ND2 files: pass
them as a glob via `--sources` (required — there is no default; the
files live wherever you keep them). It copies source ND2s as
surveys/FRAPs and writes QC PNGs. By default it visits each source file
once (at most 25 positions); pass `--max-positions` / `--grid` / `--nx`
/ `--ny` to override.

## Run on the microscope

Minimal example — the default centre-out spiral (25 positions around
the current stage position), 1 cycle per FOV, local Otsu+watershed
detector:

```bash
python -m autofrap.pipeline \
  --out /path/to/output \
  --name spiral_run \
  --detector autofrap/detectors/simple_seg_detector.py
```

The Cellpose detector files work the same way (server running, see
above), with the segmentation parameters passed via `--detector-arg`:

```bash
  --detector autofrap/detectors/cellpose_remote_halfnucleus_modular.py \
  --detector-arg diameter=70 \
  --detector-arg min_size=50
```

`--name` sets the experiment name; the run directory is created under
`--out` with a timestamp prefix, e.g. `20260901_123456_spiral_run/`. See
`--help` for all options.

Key arguments:
* `--out` output root, per-FOV subdirs created automatically
* `--nis` path to `nis_ar.exe` (default: the standard NIS-Elements install path)
* `--frap-oc` optical configuration used for FRAP stimulation (default: `FRAPPA`)
* `--max-positions` / `--num-positions` number of positions to visit; default 25 in the default spiral order (start plus two loops around it), caps the grid in `--grid` mode
* `--grid` visit positions in a plain NxM grid instead of the default centre-out spiral
* `--nx/--ny` grid size (`--grid` mode)
* `--spacing` FOV spacing in FOV units
* `--max-cycles` cycles per FOV (default: 1)
* `--until-done` run until all cells of a FOV are stimulated (ignores `--max-cycles`)
* `--max-consecutive-failures` abort the run after this many consecutive FOV failures (default: 3)
* `--detector` path to the detector module (required – see below)
* `--detector-arg key=value` tuning parameters forwarded to the detector (repeatable)
* `--verbose` DEBUG logging: per-cycle detail plus the NIS macro traffic (macro bodies, ini results, `nis_ar` output); failed macros are preserved in `<run_dir>/macro_debug/`

### Visit order: spiral (default) or plain grid

By default positions are visited in a centre-out square spiral (via
`autofrap.core.utils.grid.spiral_positions`), starting at the current
stage position. `--max-positions` sets the number of positions to
visit; if omitted it defaults to 25 — the start position plus two
loops around it. For a plain NxM grid instead:

```bash
python -m autofrap.pipeline \
  --out /path/to/output \
  --grid --nx 2 --ny 2 \
  --name grid_run \
  --detector autofrap/detectors/simple_seg_detector.py
```

`--max-positions` then caps the grid: e.g. `--grid --nx 5 --ny 5
--max-positions 20` visits the first 20 positions of the 5×5 grid in
row-major order.

## Detectors

Every run needs `--detector`: a `.py` file that defines
`detection_fun(survey_file) -> labels (, stim_mask (, viz))` — the
object labels for the survey image, optionally the per-cell FRAP
regions and a QC picture. The pipeline imports the file and calls it
once per survey image.

`autofrap/detectors/` contains ready-made detectors for the common
cases (local Otsu+watershed, remote-Cellpose variants with different
stimulation masks, a dummy for testing) — the "Built-in detectors"
table in `WRITING_DETECTOR.md` says what each one does and when to
use it.

To write your own, start with `WRITING_DETECTOR.md`: it walks through
the contract, the building blocks in `autofrap/core/` (load → detect
→ mask → filter → viz), and shows how to assemble a detector with
`build_detector` instead of writing one from scratch.

## Output

Per FOV:
* `fovNN_cycleNN_survey.nd2`
* `fovNN_cycleNN_frap.nd2` – with `StandardROI` + `StimulationROI`
* `fovNN_cycleNN_survey_qc.png` – overlay with labels, FRAP mask, selected cell

The pipeline returns to start position on completion or abort.

## Development: running the tests

The test suite (`tests/`, stdlib `unittest`, no extra dependencies)
mirrors the package layout:

```bash
python -m unittest discover -s tests -t .    # from the repo root
```

`tests/live/` holds the probes that need the microscope (NIS-Elements
open) — they skip automatically on any other machine, and tests that
need data files not present on your machine skip as well, so the
suite is green everywhere. See `STATUS.md` and `docs/ARCHITECTURE.md`
for the layout.

## Notes

* Survey files can be multi-channel. `load_fun` returns `(c, y, x)`
  and detection functions can use any channel(s); the visualization is
  the only `(y, x, 3/4)` array.
* The microscope workstation must have both an ND-Acquisition template
  for the survey image and an ND-Stimulation template for the FRAP
  time series defined in NIS Elements GUI.
* For real runs, ensure the ND acquisition template is a single image
  with one or more channels, no Time/XY/Large Image tabs.
* Ctrl-C is a clean stop: the current macro call runs to completion and
  the run stops at the next safe boundary (end of cycle / between
  FOVs), with per-cycle cleanup. Exit codes: 0 run finished, 1 aborted
  (configuration/resource problem), 130 stopped by user.

See `STATUS.md` for the current state and open TODOs,
`docs/ARCHITECTURE.md` for the code layout, and
`DESIGN_GOALS_AUTOFRAP.md` for the experiment workflow.
