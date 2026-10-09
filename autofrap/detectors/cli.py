"""
Offline detector runner: ``python -m autofrap.detectors``

Run a detector file on a single image and save a QC overlay, without
the microscope: the same detector loading (load_detector_file), output
unpacking and cell selection as a live pipeline run. Use it to tune
detector parameters on a survey image and to check which cells would
be found, which one would be picked first and what would be sent to
NIS - before spending scope time.

Usage::

    python -m autofrap.detectors --detector autofrap/detectors/simple_seg_detector.py \\
        path/to/survey.nd2 --detector-arg cell_sigma=16

Writes ``<image stem>_qc.png`` next to the input image (override with
``--out``): detection, FRAP mask, the first cell that would be picked
and its ROI polygons, on the detector's visualization (blank canvas if
it provides none). ``--detector-arg`` works exactly as in a live run.

Exit codes:
  0   detection ran and the QC overlay was saved (even when no cell is
      selectable - the overlay then shows labels + FRAP mask for
      diagnosis)
  1   detector loading / --detector-arg / detection / overlay error
"""
import argparse
import logging
import os
import sys

import numpy as np
from skimage.measure import regionprops

from autofrap.core.detection import (
    load_detector_file,
    parse_detector_args,
    unpack_detection_result,
)
from autofrap.core.image.mask import select_next_cell
from autofrap.core.image.qc import save_qc_overlay

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def parse_cli_args(argv=None):
    """argument parser for the detector runner (see module docstring)"""
    p = argparse.ArgumentParser(
        description="run a detector file on one image and save a QC "
        "overlay (offline - no microscope involved; same detector "
        "contract and cell selection as a live run)"
    )
    p.add_argument(
        "file",
        help="image to run the detector on (e.g. a survey nd2)",
    )
    p.add_argument(
        "--detector",
        required=True,
        help="path to a .py file defining detection_fun (built-ins in "
        "autofrap/detectors/, see WRITING_DETECTOR.md)",
    )
    p.add_argument(
        "--detector-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra parameter to pass to the detector, "
        "e.g. --detector-arg diameter=30 (repeatable; same semantics "
        "as the pipeline's --detector-arg)",
    )
    p.add_argument(
        "--out",
        "-o",
        default=None,
        help="QC overlay output PNG [default: <image stem>_qc.png "
        "next to the input image]",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="DEBUG logging (per-cell detail at INFO is the default)",
    )
    return p.parse_args(argv)


def main(argv=None):
    """CLI entry point (see module docstring for usage and exit codes)"""
    args = parse_cli_args(argv)
    # the CLI owns the logging configuration (library style, as the
    # pipeline modules)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if args.verbose:
        # third-party DEBUG is chatty (matplotlib font matching, ...)
        logging.getLogger("matplotlib").setLevel(logging.INFO)

    detector_name = os.path.splitext(os.path.basename(args.detector))[0]
    out_path = (
        args.out or os.path.splitext(args.file)[0] + "_qc.png"
    )

    # 1. load the detector (same loader + contract as the pipeline)
    logger.info(f"loading detector from: {args.detector}")
    try:
        detection_fun = load_detector_file(args.detector)
    except (ValueError, OSError) as e:
        logger.error(f"could not load detector: {e}")
        sys.exit(1)

    try:
        detector_kwargs = parse_detector_args(args.detector_arg)
    except ValueError as e:
        logger.error(f"--detector-arg {e}")
        sys.exit(1)
    if detector_kwargs:
        logger.info(f"detector args: {detector_kwargs}")

    # 2. run detection on the image and unpack the output
    try:
        result = detection_fun(args.file, **detector_kwargs)
        labels, stimulation_mask, viz = unpack_detection_result(result)
    except Exception as e:
        logger.error(f"detection failed on {args.file}: {e!r}")
        sys.exit(1)

    n_obj = len(np.unique(labels)) - 1
    logger.info(f"{n_obj} object(s) detected")

    # 3. the cell a live run would pick first at a fresh FOV (empty
    # already-imaged map), including the polygon viability check
    cell, cell_poly, stim_poly, skipped = select_next_cell(
        labels, set(), stimulation_mask
    )
    if skipped:
        logger.info(f"cells {skipped}: no viable polygon, skipped")

    # 4. per-cell summary (area + FRAP-region area)
    for rp in regionprops(labels):
        if stimulation_mask is None:
            stim_area = int(rp.area)
        else:
            stim_area = int(np.sum((labels == rp.label) & stimulation_mask))
        logger.info(f"  cell {rp.label}: area {int(rp.area)} px, "
                    f"FRAP region {stim_area} px")

    if cell is None:
        logger.info("no stimulatable cell: nothing would be selected")
    else:
        logger.info(f"cell {cell} would be selected for the first "
                    f"FRAP cycle ({len(cell_poly)} cell / "
                    f"{len(stim_poly)} stim polygon vertices)")

    # 5. QC overlay: image + labels + FRAP mask + first pick + polygons
    caption = (
        f"{detector_name} cell {cell}" if cell is not None
        else f"{detector_name} no cell"
    )
    try:
        save_qc_overlay(
            viz,
            labels,
            out_path,
            stimulation_mask=stimulation_mask,
            cell_id=cell,
            cell_poly=cell_poly,
            stim_poly=stim_poly,
            caption=caption,
        )
    except Exception as e:
        logger.error(f"QC overlay failed: {e!r}")
        sys.exit(1)
    logger.info(f"QC overlay saved: {out_path}")


if __name__ == "__main__":
    main()
