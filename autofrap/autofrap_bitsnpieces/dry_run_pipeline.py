"""
Offline dry run of the real autoFRAP CLI with FakeNIS standing in for the
scope: survey/FRAP "acquisitions" are copies of real survey nd2 files.

Everything the pipeline CLI offers works as-is (--grid, --max-positions,
--max-cycles, --detector-arg, --max-consecutive-failures, Ctrl-C clean stop, ...) -
this wrapper only adds:

  * FakeNIS patching, sourcing "acquired" surveys from --sources
  * detector presets for convenience (--preset)
  * dry-run defaults: --out test_acquisitions/dry_run, --name dryrun
    (both overridable by passing the same flags)

Run from the repo root:

    python autofrap/autofrap_bitsnpieces/dry_run_pipeline.py --preset dummy
        offline: dummy detector, default centre-out spiral (25 positions),
        1 cycle per FOV
    python autofrap/autofrap_bitsnpieces/dry_run_pipeline.py \
        --preset simple_seg --max-positions 5 --max-cycles 3
    python autofrap/autofrap_bitsnpieces/dry_run_pipeline.py \
        --preset cellpose --detector-arg server_url=http://localhost:9000
        (needs the cellpose server; the preset already sets a default)
"""
import argparse
import glob
import os
import sys

# Ensure the repo root is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
# repo root is 2 levels up
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from autofrap.microscope.fake_nis import FakeNIS
from autofrap.pipeline.autofrap import main as pipeline_main

PRESETS = {
    'cellpose': [
        '--detector', os.path.join(_ROOT, 'autofrap', 'detectors',
                                   'cellpose_remote_halfnucleus_modular.py'),
        '--detector-arg', 'diameter=70',
        '--detector-arg', 'server_url=http://localhost:9000',
    ],
    'simple_seg': [
        '--detector', os.path.join(_ROOT, 'autofrap', 'detectors',
                                   'simple_seg_detector.py'),
        '--detector-arg', 'cell_sigma=16.0',
        '--detector-arg', 'otsu_frac=0.3',
        '--detector-arg', 'min_eroded_extent=0.90',
    ],
    'dummy': [
        '--detector', os.path.join(_ROOT, 'autofrap', 'detectors',
                                   'example_detector.py'),
    ],
}


def main():
    p = argparse.ArgumentParser(
        description='offline dry run of the autoFRAP CLI (FakeNIS)',
        epilog='all arguments not listed here are forwarded verbatim to '
               'the pipeline CLI (python autofrap/pipeline/autofrap.py --help)')
    p.add_argument('--preset', choices=sorted(PRESETS), default='dummy',
                   help='detector preset: expands to --detector / '
                        '--detector-arg (default: %(default)s)')
    p.add_argument('--sources',
                   default='test_acquisitions/autofrap_out/*survey.nd2',
                   help='glob of nd2 files FakeNIS copies as the '
                        '"acquired" surveys (default: %(default)s)')
    p.add_argument('--out', '-o', default='test_acquisitions/dry_run',
                   help='output directory (default: %(default)s)')
    p.add_argument('--name', default='dryrun',
                   help='experiment name (default: %(default)s)')
    args, forwarded = p.parse_known_args()

    sources = sorted(glob.glob(args.sources))
    assert sources, f'no survey sources found for {args.sources!r}'
    print(f'{len(sources)} sources: '
          f'{", ".join(os.path.basename(s) for s in sources)}')

    argv = PRESETS[args.preset] + ['--out', args.out, '--name', args.name] \
        + forwarded

    with FakeNIS(sources):
        # main() raises SystemExit(0/1/130), which propagates through
        # FakeNIS.__exit__ (patch restored) and becomes this process's
        # exit code
        pipeline_main(argv)


if __name__ == '__main__':
    main()
