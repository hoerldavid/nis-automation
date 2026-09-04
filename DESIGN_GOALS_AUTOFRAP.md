# autoFRAP pipeline

The goal of the autoFRAP pipeline is to automate FRAP (fluorescence recovery after photobleaching) experiments.

The envisioned workflow is as follows:

1) The user has to set up the microscope via the NIS Elements software. This includes defining an ND-Acquisition template for "survey" images and a ND-Stimulation template for FRAP timeseries. 
    - 1a) When started, the autoFRAP code should make a reasonable effort to check whether the survey ND-Acquisition is set up correctly, e.g. it should only be a single image, with one or more channels. Z-stacks may be allowed in the future but timeseries and multi-position acquisitions don't make sense here.
2) autoFRAP generates a list of XY stage positions around the current position (based on current microscope settings, e.g. objective, field of view, etc.). Uses should set how many positions to visit (either an NxM grid or a square spiral around the starting position).
3) Iterate over the positions in the list and move the stage to the position (outer loop).
4) At each position, we do multiple FRAP-cycles (inner loop) consisting of a survey image folowed by a FRAP time series. Users can set a maximum on how many cycles to perform at each position, usually it will only be one or a few.
5) The autoFRAP script taskes a survey image with the current ND-Acquisition template and saves it to a file (nd2 format).
6) The file path is passed to a detector function that will return:
    - an integer label map of the detected objects (called cells here, as it will likely be the individual cells)
    - (optionally) a binary mask of subregions of the cells in which to perform FRAP. A typical example is half-nucleus FRAP: Here the cell labels will correspond to the nuclei, and the FRAP mask will be halfes of the nuclei. There should not be more than one connected FRAP region per cell.
    - if no FRAP mask is given, the whole objects / cells are used for FRAP.
  The detector function should be easily interchangable, as this is what will be adjusted for different experiments. E.g. instead of detecting cells and halving them, it might instead detect organelles within the cells as FRAP targets or filter cells from the label map if they do not show expression of a certain marker.
7) The autoFRAP script pickes a label from the label map and a FRAP/stimulation region within from the results.
    - to avoid performing FRAP on the same cell twice, the script should make a reasonable effort to track cells between cycles. This is done by overlap-based tracking.
    - cells that do not contain a FRAP region will be skipped.
8) The autoFRAP script add the selected FRAP region as a stimulation ROI (and the whole cell as a regular ROI) and runs the current ND-Stimulation template + saves the resulting FRAP time series to a file (nd2 format).
9) Continue with the next cycle at the current position (inner loop) unless there no more cells or the user-defined maximum number of cycles has been reached.
10) Continue with the next position (outer loop) unless there are no more positions.

    