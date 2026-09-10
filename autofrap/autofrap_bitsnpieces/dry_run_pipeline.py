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
from autofrap import fake_nis
from autofrap.detection import load_detector_file

SURVEY_GLOB = 'test_acquisitions/autofrap_grid/20260901_160216/fov*/*survey.nd2'
OUT_DIR = 'test_acquisitions/dry_run'


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--detector',
                   default='autofrap/detectors/cellpose_remote_detector.py',
                   help='detector file to use')
    args = p.parse_args()

    sources = sorted(glob.glob(SURVEY_GLOB))
    assert sources, f'no survey files found for {SURVEY_GLOB}'
    print(f'{len(sources)} sources: {[os.path.basename(s) for s in sources]}')
    os.makedirs(OUT_DIR, exist_ok=True)

    with fake_nis.FakeNIS(sources) as fake:
        det = load_detector_file(args.detector)
        results = autofrap.autofrap_grid(
            'fake', OUT_DIR,
            nx=2, ny=2, max_cycles=3,
            detection_fun=det,
            name='mps_dryrun',
            diameter=70)  # forwarded to the server (20260901: 26 cells here)

    # autofrap_grid results: (i, x, y, fov_dir, fov_results | None)
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
