"""
Offline dry run of the real autoFRAP CLI with FakeNIS standing in for the
scope: survey/FRAP "acquisitions" are copies of real survey nd2 files.

Run from the repo root:

    python -m autofrap.pipeline.dry_run --preset dummy

Everything the pipeline CLI offers works as-is (--grid, --max-positions,
--max-cycles, --detector-arg, --max-consecutive-failures, Ctrl-C clean stop, ...)
- this tool only adds:

  * FakeNIS patching, sourcing "acquired" surveys from --sources
    (required - no default: pass a glob of your own survey files)
  * detector presets for convenience (--preset)
  * dry-run defaults: --out test_acquisitions/dry_run, --name dryrun
    (both overridable by passing the same flags)
  * a position-count default: unless --max-positions / --grid / --nx / --ny
    is passed, the dry run visits each source file once (at most
    SPIRAL_DEFAULT_POSITIONS) instead of the CLI default

Examples:

    python -m autofrap.pipeline.dry_run --preset dummy \
        --sources "test_acquisitions/autofrap_out/*survey.nd2"
        offline: dummy detector, one FOV per source file, 1 cycle per FOV
    python -m autofrap.pipeline.dry_run --preset simple_seg \
        --sources "/path/to/data/*survey.nd2" --max-positions 5 --max-cycles 3
    python -m autofrap.pipeline.dry_run --preset cellpose \
        --sources "/path/to/data/*survey.nd2" \
        --detector-arg server_url=http://localhost:9000
        (needs the cellpose server; the preset already sets a default)
"""
import argparse
import glob
import os

import autofrap
from autofrap.microscope.fake_nis import FakeNIS
from autofrap.pipeline.autofrap import SPIRAL_DEFAULT_POSITIONS
from autofrap.pipeline.autofrap import main as pipeline_main

DETECTORS_DIR = os.path.join(os.path.dirname(autofrap.__file__), 'detectors')

PRESETS = {
    'cellpose': [
        '--detector', os.path.join(DETECTORS_DIR,
                                   'cellpose_remote_halfnucleus_modular.py'),
        '--detector-arg', 'diameter=70',
        '--detector-arg', 'server_url=http://localhost:9000',
    ],
    'simple_seg': [
        '--detector', os.path.join(DETECTORS_DIR, 'simple_seg_detector.py'),
        '--detector-arg', 'cell_sigma=16.0',
        '--detector-arg', 'otsu_frac=0.3',
        '--detector-arg', 'min_eroded_extent=0.90',
    ],
    'dummy': [
        '--detector', os.path.join(DETECTORS_DIR, 'example_detector.py'),
    ],
}

# flags that choose the position count / visit order explicitly - when any
# of them is passed, the dry run does not inject its --max-positions default
POSITION_FLAGS = ('--max-positions', '--grid', '--nx', '--ny')


def _has_position_flag(forwarded):
    return any(a in POSITION_FLAGS
               or any(a.startswith(f + '=') for f in POSITION_FLAGS)
               for a in forwarded)


def cap_positions(forwarded, n_sources):
    """the dry-run --max-positions default: visit each source file once,
    capped at the CLI default - None when a position flag was passed"""
    if _has_position_flag(forwarded):
        return None
    return min(n_sources, SPIRAL_DEFAULT_POSITIONS)


def build_argv(preset, sources, out, name, forwarded):
    """assemble the pipeline CLI argv for a dry run (pure - no FakeNIS,
    no pipeline run)"""
    argv = list(PRESETS[preset]) + ['--out', out, '--name', name] \
        + list(forwarded)
    cap = cap_positions(forwarded, len(sources))
    if cap is not None:
        argv += ['--max-positions', str(cap)]
    return argv


def main(argv=None):
    p = argparse.ArgumentParser(
        description='offline dry run of the autoFRAP CLI (FakeNIS)',
        epilog='all arguments not listed here are forwarded verbatim to '
               'the pipeline CLI (python -m autofrap.pipeline --help)')
    p.add_argument('--preset', choices=sorted(PRESETS), default='dummy',
                   help='detector preset: expands to --detector / '
                        '--detector-arg (default: %(default)s)')
    p.add_argument('--sources', required=True,
                   help='glob of nd2 files FakeNIS copies as the '
                        '"acquired" surveys (required: pass your own '
                        'survey files, e.g. "/path/to/data/*survey.nd2")')
    p.add_argument('--out', '-o', default='test_acquisitions/dry_run',
                   help='output directory (default: %(default)s)')
    p.add_argument('--name', default='dryrun',
                   help='experiment name (default: %(default)s)')
    args, forwarded = p.parse_known_args(argv)

    sources = sorted(glob.glob(args.sources))
    assert sources, f'no survey sources found for {args.sources!r}'
    print(f'{len(sources)} sources: '
          f'{", ".join(os.path.basename(s) for s in sources)}')
    cap = cap_positions(forwarded, len(sources))
    if cap is not None:
        print(f'positions: {cap} (one per source file, at most '
              f'{SPIRAL_DEFAULT_POSITIONS}; pass --max-positions / --grid / '
              f'--nx / --ny to override)')

    with FakeNIS(sources):
        # main() returns on exit 0; SystemExit(1/130) propagates through
        # FakeNIS.__exit__ (patch restored) and becomes this process's
        # exit code
        pipeline_main(build_argv(args.preset, sources, args.out,
                                 args.name, forwarded))


if __name__ == '__main__':
    main()
