"""
one-off offline dry run of the full autofrap_grid pipeline: FakeNIS
stands in for the scope (survey/FRAP files are copies of real survey
nd2s from the 20260901 grid run), detection is real cellpose via the
remote server (CELLPOSE_SERVER_URL; on the Mac: --device mps).

Run from the repo root:
    CELLPOSE_SERVER_URL=http://localhost:8000 \
        python autofrap/autofrap_bitsnpieces/dry_run_pipeline.py

Expect: 4 FOV x 3 cycles, ~24 cells per FOV at diameter=70; per cycle a
survey .nd2, a FRAP .nd2 (copy of the survey source) and a survey QC png.
"""
import glob
import os
import sys

# Ensure the repo root is on sys.path
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

import autofrap
from autofrap.microscope import fake_nis
from autofrap.core.detection import load_detector_file
from autofrap.core.utils.grid import spiral_positions

SURVEY_GLOB = 'test_acquisitions/autofrap_out/*survey.nd2'
OUT_DIR = 'test_acquisitions/dry_run'

# Spiral traversal settings
SPIRAL = True  # set False to use default grid
SPIRAL_MAX_POS = 5  # number of positions to generate
SPIRAL_FOV = 133.1  # µm, approximate FOV size for this sample
SPIRAL_SPACING = 1.0  # FOV units between spiral layers
START_XY = (0.0, 0.0)  # current stage position (µm) – FakeNIS centre

PRESETS = {
    'cellpose': {
        'detector': 'autofrap/detectors/cellpose_remote_halfnucleus_modular.py',
        'kwargs': {'diameter': 70, 'server_url': 'http://localhost:9000'}
    },
    'simple_seg': {
        'detector': 'autofrap/detectors/simple_seg_detector.py',
        'kwargs': {'cell_sigma': 16.0, 'otsu_frac': 0.3, 'min_eroded_extent': 0.90}
    },
    'dummy': {
        'detector': 'autofrap/detectors/dummy_detector.py',
        'kwargs': {}
    },
}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=list(PRESETS.keys()), default='simple_seg',
                   help='detector preset to use')
    p.add_argument('--detector',
                   help='override detector file, overrides --preset')
    args = p.parse_args()

    sources = sorted(glob.glob(SURVEY_GLOB))
    assert sources, f'no survey files found for {SURVEY_GLOB}'
    print(f'{len(sources)} sources: {[os.path.basename(s) for s in sources]}')
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # Generate the position list (grid or spiral) before the FakeNIS context.
    # -----------------------------------------------------------------
    if SPIRAL:
        positions = spiral_positions(START_XY, fov=SPIRAL_FOV, spacing=SPIRAL_SPACING,
                                    max_positions=SPIRAL_MAX_POS)
    else:
        positions = None

    # Resolve detector and kwargs from preset or explicit override
    if args.detector:
        detector_path = args.detector
        detector_kwargs = {}
    else:
        preset = PRESETS[args.preset]
        detector_path = preset['detector']
        detector_kwargs = preset['kwargs']

    with fake_nis.FakeNIS(sources) as fake:
        det = load_detector_file(detector_path)
        results = autofrap.autofrap_multiposition(
            'fake', OUT_DIR,
            positions=positions, max_cycles=3,
            detection_fun=det,
            name='dryrun',
            **detector_kwargs)

    # autofrap_multiposition results: (i, x, y, fov_dir, fov_results | None)
    n_fov = sum(1 for r in results if r[4] is not None)
    n_cycles = sum(len(r[4]) for r in results if r[4] is not None)
    print(f'{n_cycles} cycles over {n_fov} FOVs')
    for i, x, y, fov_dir, fov_results in results:

        if fov_results is None:
            print(f'fov {i}: no results')
            continue
        for cycle, cell, survey, frap in fov_results:
            print(f'fov{i:02d} c{cycle:02d} cell {cell}: '
                  f'{os.path.basename(survey)} -> {os.path.basename(frap)}')


if __name__ == '__main__':
    main()
