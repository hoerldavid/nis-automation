# Shipped detector files for the pipeline's --detector option.
#
# Each file defines ``detection_fun(survey_file)`` with the signature
# ``survey_file -> (labels[, stim_mask[, viz]])``.
#
# Users can also write their own detector file and pass it via
# ``--detector path/to/my_detector.py``.
#
# Test any detector file offline on a single image (no microscope):
# ``python -m autofrap.detectors --detector <file> image.nd2`` (see
# ``autofrap/detectors/cli.py`` and WRITING_DETECTOR.md §5).
