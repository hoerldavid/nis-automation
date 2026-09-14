"""
QC overlay rendering for the auto-FRAP pipeline.

Renders a single PNG combining the survey image, the detector's label
map, the stimulation mask, and (optionally) the cell selected for FRAP
plus the exact polygons sent to NIS. Saved next to the survey file so
detection + selection can be spot-checked without opening NIS
(STATUS.md TODO #8).

Rendering is headless (Agg backend): the microscope workstation may not
have a display, and this is a pipeline artifact, not interactive work.
"""
import warnings

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from skimage.measure import find_contours, regionprops

# layer palette (bottom -> top)
COLOR_LABEL = (1.0, 1.0, 1.0)      # contours of all labels
COLOR_TEXT = (1.0, 1.0, 1.0)       # label IDs
COLOR_STIM = (1.0, 0.65, 0.0)      # stimulation mask (orange)
COLOR_CELL_POLY = (0.0, 1.0, 1.0)  # whole-cell polygon sent to NIS (cyan)
COLOR_STIM_POLY = (1.0, 0.0, 1.0)  # stimulation polygon sent to NIS (magenta)


def _image_clipping(image):
    """percentile clipping so a few hot pixels don't wash out the image"""
    lo, hi = np.percentile(image, [1, 99.5])
    if hi <= lo:  # constant image: fall back to full range
        lo, hi = float(image.min()), float(image.max())
    return float(lo), float(hi)


def _draw_poly(ax, poly, color, lw, zorder):
    """draw a closed polygon (list of (x, y) pixel coords)"""
    if not poly:
        return
    pts = np.asarray(poly, dtype=float)
    pts = np.vstack([pts, pts[:1]])  # close the contour
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=zorder,
            solid_capstyle='round', solid_joinstyle='round')


def save_qc_overlay(image, labels, path, stimulation_mask=None,
                    cell_id=None, cell_poly=None, stim_poly=None,
                    caption=None, dpi=100):
    """
    render and save a QC overlay PNG

    Layers, bottom to top (explicit zorder; note matplotlib would
    otherwise draw Text above every imshow image regardless of call
    order):
      1. image (2D: grayscale, 1-99.5 % percentile-clipped; RGB(A):
         as-is - a detector-provided visualization is already scaled;
         None: blank black canvas)
      2. stimulation mask (orange fill)
      3. contours of all labels + stimulation mask contour
      4. polygons sent to NIS (solid: whole cell cyan, stimulation magenta)
      5. label IDs (on top, so they stay readable over the orange fill;
         the selected cell's ID is bold cyan, matching its ROI)
      6. legend + caption

    The selected cell gets no extra outline: the cyan/magenta ROI
    polygons already mark it well enough (an extra contour underneath
    them is practically invisible).

    The solid polygons show the *simplified* outlines actually sent to
    NIS, so any deviation from the raw mask contours is visible.

    Parameters
    ----------
    image: 2D np.ndarray (y, x) numeric, or 3D (y, x, 3/4), or None
        2D: the survey channel the detector ran on, shown grayscale
        with 1-99.5 % percentile clipping; RGB(A): shown as-is, no
        scaling or colormap - e.g. a multi-channel visualization
        assembled by the detector (its responsibility to provide);
        None: a blank (black) canvas - the palette (white contours and
        IDs, black-stroked text) is designed for a dark background
    labels: 2D np.ndarray (y, x), int
        label map (0 = background, 1..N = objects)
    path: str
        output PNG path
    stimulation_mask: 2D np.ndarray (y, x), bool, optional
        stimulation-eligible areas (all cells); filled in orange
    cell_id: int, optional
        the label selected for FRAP; highlighted in green
    cell_poly: list of (x, y), optional
        whole-cell polygon as sent to NIS
    stim_poly: list of (x, y), optional
        stimulation polygon as sent to NIS
    caption: str, optional
        text shown top-left (e.g. 'c02  cell 5 of 23')
    dpi: int
        output resolution; the figure is image-sized, so dpi sets the
        pixel size of the PNG (100 -> a 1024 x 1024 image gives a
        1024 x 1024 PNG)

    Returns
    -------
    None (the PNG is written to path)
    """
    if labels.ndim != 2:
        raise ValueError(f'labels must be 2D, got {labels.shape}')
    if image is not None:
        if image.ndim == 2:
            if image.shape != labels.shape:
                raise ValueError(f'image/labels shape mismatch: '
                                 f'{image.shape} vs {labels.shape}')
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            if image.shape[:2] != labels.shape:
                raise ValueError(f'image/labels shape mismatch: '
                                 f'{image.shape} vs {labels.shape}')
        else:
            raise ValueError(f'image must be 2D or (y, x, 3/4), '
                             f'got {image.shape}')

    h, w = labels.shape
    if image is None:
        # no visualization from the detector: blank black canvas, drawn
        # through the regular 3D (as-is) path below
        image = np.zeros((h, w, 3))
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.set_axis_off()
    # pixel-center coordinates: pixel (i, j) is centered at (x=j, y=i),
    # the same convention find_contours / mask_to_polygon / NIS use
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    # 1. image: 2D -> grayscale + percentile clipping (hot pixels);
    # RGB(A) -> as-is
    if image.ndim == 2:
        lo, hi = _image_clipping(image)
        ax.imshow(image, cmap='gray', vmin=lo, vmax=hi, zorder=0)
    else:
        ax.imshow(image, zorder=0)

    # 2. stimulation mask fill (all cells)
    if stimulation_mask is not None:
        overlay = np.zeros((h, w, 4))
        overlay[stimulation_mask] = (*COLOR_STIM, 0.3)
        ax.imshow(overlay, zorder=1)

    # 3. contours: all labels + stimulation mask
    for prop in regionprops(labels):
        mask = labels == prop.label
        for contour in find_contours(mask, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color=COLOR_LABEL,
                    lw=0.8, alpha=0.8, zorder=2)
    if stimulation_mask is not None:
        for contour in find_contours(stimulation_mask, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color=COLOR_STIM,
                    lw=1.0, alpha=0.9, zorder=2.5)

    # 4. selected cell: no outline (the ROI polygons mark it); only its
    # ID text is highlighted below. Still warn if the id is bogus.
    if cell_id is not None and not (labels == cell_id).any():
        warnings.warn(f'cell_id {cell_id} not present in labels',
                      stacklevel=2)

    # 5. polygons sent to NIS
    _draw_poly(ax, cell_poly, COLOR_CELL_POLY, 1.5, zorder=3)
    _draw_poly(ax, stim_poly, COLOR_STIM_POLY, 2.0, zorder=3)

    # 6. label IDs (on top of everything, incl. the orange fill)
    text_outline = [path_effects.withStroke(linewidth=2.0,
                                            foreground='black')]
    for prop in regionprops(labels):
        selected = prop.label == cell_id
        ax.text(prop.centroid[1], prop.centroid[0], str(prop.label),
                color=COLOR_CELL_POLY if selected else COLOR_TEXT,
                fontsize=16 if selected else 10,
                weight='bold' if selected else 'normal',
                ha='center', va='center', path_effects=text_outline,
                zorder=5)

    # 7. legend (only the layers that are present)
    handles = [Line2D([], [], color=COLOR_LABEL, lw=0.8, alpha=0.8,
                      label='cells')]
    if stimulation_mask is not None:
        handles.append(Patch(facecolor=(*COLOR_STIM, 0.3),
                             edgecolor=COLOR_STIM, label='FRAP mask'))
    if cell_poly:
        handles.append(Line2D([], [], color=COLOR_CELL_POLY, lw=1.5,
                              label='selected cell'))
    if stim_poly:
        handles.append(Line2D([], [], color=COLOR_STIM_POLY, lw=2.0,
                              label='stim ROI'))
    legend = ax.legend(handles=handles, loc='upper left', fontsize=12,
                       facecolor='black', framealpha=0.55, edgecolor='none',
                       labelcolor='white')
    legend.set_zorder(6)

    if caption:
        ax.text(0.01, 0.02, caption, transform=ax.transAxes,
                color='white', fontsize=11, ha='left', va='bottom',
                path_effects=text_outline, zorder=6,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                          alpha=0.55))

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
