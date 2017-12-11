from skimage.morphology import remove_small_holes, watershed, binary_erosion, remove_small_objects
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, binary_opening, ball, disk
from skimage.measure import regionprops, label
from skimage.filters import threshold_adaptive, threshold_otsu, threshold_local
from skimage.morphology import binary_closing, ball, disk, binary_opening

import javabridge
import bioformats

from matplotlib.widgets import RectangleSelector, Button
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.transform import rescale, pyramid_gaussian, resize
from skimage.color import label2rgb

def bbox_pix2unit(bbox, start, pixsize, direction):
    res = (np.array(bbox, dtype=float).reshape((2,2)) * np.array(pixsize, dtype=float) *
            np.array(direction, dtype=float) + np.array(start, dtype=float))
    return res.reshape((4,))

def aspect(bbox):
    (ymin, xmin, ymax, xmax) = bbox
    exy = ymax - ymin
    exx = xmax - xmin
    return (exy / exx) if (exx > exy) else (exx / exy)

def detect_wings_simple(img, start, pixsize, direction,
                        ds=2, layers=2, thresh_window=351,
                        minarea=20000, maxarea=80000, minsolidity=.6,
                        minaspect=.3, plot=False, threshold_fun=None):
    # downsample
    pyr = [p for p in pyramid_gaussian(img, max_layer= layers, downscale = ds)]
    img_ds = pyr[layers]
    # adaptive threshold
    if threshold_fun is None:
        thrd = img_ds > threshold_local(img_ds, thresh_window)
    else:
        thrd = img_ds > threshold_local(img_ds, thresh_window, method='generic', param=threshold_fun)
    # clean a bit
    thrd = np.bitwise_not(thrd)
    thrd = binary_opening(thrd, selem=disk(4))

    # filter objs
    ls = [r.label for r in regionprops(label(thrd)) if r.area>minarea and
          r.area<maxarea and r.solidity>minsolidity and aspect(r.bbox) > minaspect]
    
    # filtered binary
    res = np.zeros(thrd.shape)
    l = label(thrd)
    for li in ls:
        res += (l == li)
    
    # more cleaning, plus some erosion to separate touching wings
    r2 = remove_small_holes(res.astype(np.bool), 25000)
    r2 = binary_erosion(r2, selem=disk(3))
    
    if plot:
        image_label_overlay = label2rgb(label(r2), image=img_ds)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image_label_overlay)     
        
    
    # get bboxes
    bboxes = []
    for r in regionprops(label(r2)):
        if r.area < (minarea * .8 ):
            continue
            
        bboxes.append(np.array(r.bbox) * (ds**layers))
        if plot:
            minr, minc, maxr, maxc = r.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    
    # pixels to units
    return [bbox_pix2unit(b, start, pixsize, direction) for b in bboxes]

def scale_bbox(bbox, expand_factor = .15):
    (ymin, xmin, ymax, xmax) = tuple(bbox)
    yrange = ymax - ymin
    xrange = xmax - xmin
    return (ymin - yrange * expand_factor / 2., xmin - xrange * expand_factor / 2.,
            ymax + yrange * expand_factor / 2., xmax + xrange * expand_factor / 2.) 
    
def read_bf(path):
    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
    img = bioformats.load_image(path, rescale=False)
    return img