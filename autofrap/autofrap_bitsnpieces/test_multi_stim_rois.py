"""
One-off exploration: multiple stimulation ROIs.

Adds two polygon ROIs to the *current* image (live view), sets both as
stimulation ROIs (ChangeROIType 3), optionally switches optical
configuration, then runs the currently configured sequential stimulation
ND experiment (ND_RunSequentialStimulationExp) and blocks until it
finishes.

Note: ChangeROIType(3) puts both ROIs into stimulation group S1 - part of
this test is to see whether the experiment stimulates both.

Usage (from the repo root, at the microscope, with live view focused
on the desired FOV):
    python -m autofrap.autofrap_bitsnpieces.test_multi_stim_rois \
        --center1 300 300 --center2 500 300 --oc "405 FRAP"
"""

import argparse
import logging
import sys

from autofrap.microscope import nis as nis_util

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def square_points(cx, cy, half):
    """axis-aligned square polygon (clockwise) around (cx, cy)."""
    return [(cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nis", default=r"C:\Program Files\NIS-Elements\nis_ar.exe",
                   help="path to nis_ar.exe")
    p.add_argument("--center1", type=int, nargs=2, required=True,
                   metavar=("X", "Y"), help="center of ROI 1 (pixels)")
    p.add_argument("--center2", type=int, nargs=2, required=True,
                   metavar=("X", "Y"), help="center of ROI 2 (pixels)")
    p.add_argument("--half", type=int, default=10,
                   help="half side length of the square ROIs (pixels)")
    p.add_argument("--oc", default=None,
                   help="optical configuration to switch to before "
                        "stimulation (e.g. the FRAP OC); skip if omitted")
    p.add_argument("--timeout", type=int, default=300,
                   help="macro timeout for the stimulation experiment (s)")
    args = p.parse_args()

    pts1 = square_points(*args.center1, args.half)
    pts2 = square_points(*args.center2, args.half)

    # 1. clean slate, then create both stim ROIs in one batched macro
    roi_calls = [
        (nis_util.OP_DELETE_ALL_ROIS_IN_CURRENT_DOCUMENT, {}),
        (nis_util.OP_CREATE_AND_SET_STIM_ROI, {"points": pts1}),
        (nis_util.OP_CREATE_AND_SET_STIM_ROI, {"points": pts2}),
    ]
    results = nis_util.batch_run_macro(args.nis, roi_calls)
    print("ROI results:", results)
    id1 = results["create_and_set_stim_roi_1"]
    id2 = results["create_and_set_stim_roi_2"]
    print(f"stim ROI 1 id={id1}, stim ROI 2 id={id2}, "
          f"ROI count on image: {nis_util.get_roi_count(args.nis)}")
    if id1 <= 0 or id2 <= 0:
        sys.exit("ERROR: stim ROI creation failed")

    # 2. switch OC (optional) and run the current stimulation experiment
    # (no interactive pause: must run without stdin in the agent TUI)
    stim_calls = []
    if args.oc:
        stim_calls.append((nis_util.OP_SET_OPTICAL_CONFIGURATION,
                           {"name": args.oc}))
    stim_calls.append((nis_util.OP_RUN_STIMULATION_EXPERIMENT, {}))
    nis_util.batch_run_macro(args.nis, stim_calls, timeout=args.timeout)
    print("stimulation experiment finished")


if __name__ == "__main__":
    main()
