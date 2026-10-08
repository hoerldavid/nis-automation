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

# editable install with dependencies
pip install -e .
```

Dependencies are declared in `pyproject.toml`.

### Microscope workstation

This is where the pipeline runs. Prerequisites: NIS Elements running
with the NIS-Elements Automation macro interface enabled, `nis_ar` on
PATH, and a Python environment with the package installed.

```bash
python -m autofrap.pipeline --help
```

### Cellpose server (optional)

Only needed for the `cellpose_remote_*` detector files; all other
detectors run fully local. Run on a separate machine with GPU / Apple
Silicon:

```bash
pip install cellpose
python /path/to/repo/cellpose_server.py --device auto --host 0.0.0.0 --port 8000
```

`cellpose_server.py` lives in the repository root. The detector files default to `DEFAULT_CELLPOSE_SERVER_URL` (defined
at the top of each file — edit it there if the server moves), or pass
it per run with `--detector-arg server_url=http://<server>:8000`.

## Quick start – dry run with FakeNIS

No microscope required:

```bash
python autofrap/autofrap_bitsnpieces/dry_run_pipeline.py --preset simple_seg
```

Presets:
* `dummy` → `autofrap/detectors/example_detector.py` (dummy objects, for testing)
* `simple_seg` → `autofrap/detectors/simple_seg_detector.py` (Otsu + watershed, local, no server)
* `cellpose` → `autofrap/detectors/cellpose_remote_halfnucleus_modular.py` with `diameter=70` (needs the server)

The dry run is designed to work with arbitrary existing ND2 files. The
test script uses a hardcoded survey glob, but the pipeline itself
accepts any source list. It copies source ND2s as surveys/FRAPs and
writes QC PNGs.

## Run on the microscope

Example 2×2 grid, 3 cycles per FOV, local Otsu+watershed detector:

```bash
python -m autofrap.pipeline \
  --out /path/to/output \
  --grid \
  --nx 2 --ny 2 \
  --spacing 1.0 \
  --max-cycles 3 \
  --name grid_run \
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
`--out` with a timestamp prefix, e.g. `20260901_123456_grid_run/`. See
`--help` for all options.

Key arguments:
* `--out` output root, per-FOV subdirs created automatically
* `--max-positions` / `--num-positions` number of positions to visit; default 25 in the default spiral order (start plus two loops around it), caps the grid in `--grid` mode
* `--grid` visit positions in a plain NxM grid instead of the default centre-out spiral
* `--nx/--ny` grid size (`--grid` mode)
* `--spacing` FOV spacing in FOV units
* `--max-cycles` cycles per FOV
* `--detector` path to the detector module (required – see below)
* `--detector-arg key=value` tuning parameters forwarded to the detector (repeatable)
* `--verbose` DEBUG logging: per-cycle detail plus the NIS macro traffic (macro bodies, ini results, `nis_ar` output); failed macros are preserved in `<run_dir>/macro_debug/`

### Spiral visit order (default)

```bash
python -m autofrap.pipeline \
  --out /path/to/output \
  --spacing 1.0 \
  --max-cycles 3 \
  --name spiral_run \
  --detector autofrap/detectors/simple_seg_detector.py \
  --detector-arg cell_sigma=16.0 \
  --detector-arg otsu_frac=0.3 \
  --detector-arg min_eroded_extent=0.90
```

By default positions are visited in a centre-out square spiral (via
`autofrap.core.utils.grid.spiral_positions`), starting at the current
stage position. `--max-positions` / `--num-positions` sets the number
of positions to visit; if omitted it defaults to 25 — the start
position plus two loops around it. Use `--grid` for a plain NxM grid
instead; `--max-positions` then caps the grid: e.g. `--grid --nx 5
--ny 5 --max-positions 20` visits the first 20 positions of the 5×5
grid in row-major order.

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

## Notes

* Survey files can be multi-channel. `load_fun` returns `(c, y, x)`
  and detection functions can use any channel(s); the visualization is
  the only `(y, x, 3/4)` array.
* The microscope workstation must have both an ND-Acquisition template
  for the survey image and an ND-Stimulation template for the FRAP
  time series defined in NIS Elements GUI.
* For real runs, ensure the ND acquisition template is a single image
  with one or more channels, no Time/XY/Large Image tabs.

See `STATUS.md` for the current state and open TODOs,
`docs/ARCHITECTURE.md` for the code layout, and
`DESIGN_GOALS_AUTOFRAP.md` for the experiment workflow.
