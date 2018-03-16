from skimage.morphology import remove_small_holes, binary_erosion
from skimage.measure import regionprops, label
from skimage.filters import threshold_local
from skimage.morphology import disk, binary_opening
from skimage.exposure import rescale_intensity
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import pyramid_gaussian
from skimage.color import label2rgb

import javabridge
import bioformats

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import logging


def bbox_pix2unit(bbox, start, pixsize, direction):
    """
    old pixel->unit conversion for bounding boxes
    NB: may no be corect
    TODO: remove if it is no longer necessary
    """
    logger = logging.getLogger(__name__)
    res = (np.array(bbox, dtype=float).reshape((2,2)) * np.array(pixsize, dtype=float) *
            np.array(direction, dtype=float) + np.array(start, dtype=float))
    logger.debug('bbox: {}, toUnit: {}'.format(bbox, res.reshape((4,))))
    return res.reshape((4,))


def aspect(bbox):
    """
    get inverse aspect ratio a bounding box (smaller axis/larger axis)
    Parameters
    ----------
    bbox: 4-tuple
        ymin, xmin, ymax, xmax

    Returns
    -------
    aspect: scalar
        inverse aspect ratio (in 0-1)
    """

    (ymin, xmin, ymax, xmax) = bbox
    exy = ymax - ymin
    exx = xmax - xmin
    return (exy / exx) if (exx > exy) else (exx / exy)


def detect_wings_simple(img, pixel_size=1,
                        ds=2, layers=2, thresh_window=1400,
                        minarea=3000, maxarea=12500, minsolidity=.6,
                        minaspect=.3, plot=False, threshold_fun=None):
    """
    simple wing detection via adaptive thresholding and some filtering by shape

    Parameters
    ----------
    img: np-array (2-dim)
        the input image
    pixel_size: scalar
        pixel size in input image
    ds: scalar
        downsampling factor at each layer
    layers: scalat
        how may downsampling layers to calculate
    thresh_window: integer
        window for adaptive threshold, in original image pixels
    minarea: scalar
        minimum size of objects to detect, in units^2
    maxarea: scalar
        maximum size of objects to detect, in units^2
    minsolidity: scalar
        minimal solidity of detected objects \in (0,1)
    minaspect: scalar
        minimal inverse aspect ratio of detected objects \in (0,1)
    plot: boolean
        whether to plot detections or not
    threshold_fun: function pointer, optional
        thresholding function to use in windows

    Returns
    -------
    bboxes: list of 4-tuples
        bounding boxes (in original image pixel units)
    """

    # scale min and max area to be in pixels^2
    minarea = minarea / pixel_size**2 * ds**(layers*2)
    maxarea = maxarea / pixel_size**2 * ds**(layers*2)

    # scale thresh window size, make sure it is odd
    thresh_window = int(thresh_window/ds**layers)
    thresh_window += 0 if thresh_window%2 == 1 else 1

    logger = logging.getLogger(__name__)

    # some debug output:
    logger.info('wing detection started')
    
    logger.debug('input shape: {}'.format(img.shape))
    logger.debug('ds: {}, layer:{}'.format(ds, layers))
    logger.debug('minarea: {}, maxarea:{}'.format(minarea, maxarea))
    logger.debug('threshold window: {}'.format(thresh_window))

    # downsample
    pyr = [p for p in pyramid_gaussian(img, max_layer= layers, downscale = ds)]
    img_ds = pyr[layers]

    logger.debug('img size after ds: {}'.format(img_ds.shape))

    # rescale to (0-1)
    img_ds = img_ds.astype(float)
    img_ds = rescale_intensity(img_ds, out_range=(0.0, 1.0))
    
    # smooth
    img_ds = gaussian_filter(img_ds, 2.0)

    # adaptive threshold
    if threshold_fun is None:
        thrd = img_ds > threshold_local(img_ds, thresh_window)
    else:
        thrd = img_ds > threshold_local(img_ds, thresh_window, method='generic', param=threshold_fun)

    # clean a bit
    thrd = np.bitwise_not(thrd)
    thrd = binary_opening(thrd, selem=disk(4))

    labelled = label(thrd)

    # filter objs
    ls = [r.label for r in regionprops(labelled) if r.area>minarea and
          r.area<maxarea and r.solidity>minsolidity and aspect(r.bbox) > minaspect]
    
    # filtered binary
    res = np.zeros(thrd.shape)
    l = label(thrd)
    for li in ls:
        res += (l == li)
    
    # more cleaning, plus some erosion to separate touching wings
    r2 = remove_small_holes(res.astype(np.bool), 25000)
    r2 = binary_erosion(r2, selem=disk(3))

    # show detections
    if plot:
        image_label_overlay = label2rgb(label(r2), image=img_ds)
        plt.imshow(image_label_overlay)
        ax = plt.gca()

    # get bboxes
    bboxes = []
    for r in regionprops(label(r2)):

        # TODO: is this really necessary?
        if r.area < (minarea * .8 ):
            continue

        bbox_scaled = np.array(r.bbox) * (ds**layers)
        logger.debug('bbox: {}, upsampled: {}'.format(r.bbox, bbox_scaled))
        bboxes.append(bbox_scaled)
        if plot:
            minr, minc, maxr, maxc = r.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    logger.info('found {} object(s)'.format(len(bboxes)) )
    
    return bboxes


def scale_bbox(bbox, expand_factor = .15):
    """
    expand a bounding box by a fixed factor

    Parameters
    ----------
    bbox: 4-tuple
        ymin, xmin, ymax, xmax
    expand_factor: scalar
        factor by which to scale ( resulting size will be 1+expand_factor)

    Returns
    -------
    bbox_scaled: 4-tuple
        ymin, xmin, ymax, xmax, scaled by factor
    """
    (ymin, xmin, ymax, xmax) = tuple(bbox)
    yrange = ymax - ymin
    xrange = xmax - xmin
    bbox_scaled = (ymin - yrange * expand_factor / 2., xmin - xrange * expand_factor / 2.,
                   ymax + yrange * expand_factor / 2., xmax + xrange * expand_factor / 2.)
    return bbox_scaled


def read_bf(path):
    """

    read an image into a np-array using BioFormats

    Parameters
    ----------
    path: str
        file path to read

    Returns
    -------
    img: np.array
        image as np-array
    """
    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
    img = bioformats.load_image(path, rescale=False)
    return img
