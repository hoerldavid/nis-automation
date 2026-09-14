"""
Simple threshold + watershed nucleus segmentation, cellpose-free.

Pure image -> label map building block. No I/O, no plotting.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential, watershed
from skimage.morphology import dilation


@dataclass
class SimpleSegParams:
    cell_sigma: float = 16.0
    bg_sigma_factor: float = 2.0
    smooth_sigma_factor: float = 1.0
    otsu_frac: float = 0.3
    close_iters: int = 2
    marker_rel: float = 0.6
    merge_ridge_frac: float = 0.45
    min_eroded_extent: float = 0.90
    marker_smooth_sigma: float = 0.0

    def derived(self):
        cs = self.cell_sigma
        bg = cs * self.bg_sigma_factor
        smooth = cs * self.smooth_sigma_factor
        marker_min_dist = int(cs)
        min_object_area = int(round(np.pi * (cs / 3) ** 2))
        erode_iters = max(1, int(round(cs / 8)))
        return {
            "bg_sigma": bg,
            "smooth_sigma": smooth,
            "marker_min_dist": marker_min_dist,
            "min_object_area": min_object_area,
            "erode_iters": erode_iters,
        }


def _merge_over_splits(lab, dist, merge_ridge_frac):
    merged = 0
    for _ in range(50):
        props = regionprops(lab)
        best = None
        for i, a in enumerate(props):
            ma = lab == a.label
            da = dist[ma].max()
            for b in props[i + 1:]:
                mb = lab == b.label
                db = dist[mb].max()
                rmax = max(da, db)
                if rmax <= 0:
                    continue
                adj = dilation(ma, footprint=np.ones((3, 3))) & mb
                if not adj.any():
                    continue
                frac = np.median(dist[adj]) / rmax
                if frac >= merge_ridge_frac and (best is None or frac > best[0]):
                    best = (frac, a.label, b.label)
        if best is None:
            break
        lab[lab == best[2]] = best[1]
        merged += 1
    return lab, merged


def _extent_filter(lab, erode_iters, min_eroded_extent):
    keep, dropped = [], []
    for rp in regionprops(lab):
        er = ndimage.binary_erosion(lab == rp.label, iterations=erode_iters)
        comps = label(er)
        if comps.max() == 0 or comps.max() > 1:
            dropped.append(rp.label)
            continue
        # extent of eroded object vs its convex hull
        if comps.max() == 0:
            extent = 0.0
        else:
            er_prop = regionprops(comps, intensity_image=comps.astype(int))
            # regionprops of comps returns one region
            er_area = er.sum()
            # convex area from first component
            try:
                convex_area = regionprops(comps)[0].area_convex
            except Exception:
                convex_area = er_area
            extent = er_area / convex_area if convex_area > 0 else 0.0
        if extent >= min_eroded_extent:
            keep.append(rp.label)
        else:
            dropped.append(rp.label)
    lab = np.isin(lab, keep) * lab
    return lab, dropped


def detect_objects(image, params: SimpleSegParams, clip_pct=None, return_intermediates=False):
    """
    image -> label map

    Parameters
    ----------
    image : ndarray
        2-D float image. Must be float to avoid underflow in high-pass.
    params : SimpleSegParams
    clip_pct : float or None
        If set, Otsu is computed on the histogram clipped to this percentile.
    return_intermediates : bool
        If True, returns (labels, intermediates) dict.

    Returns
    -------
    labels or (labels, intermediates)
    """
    image = np.asarray(image, dtype=float)
    d = params.derived()
    bg_sigma = d["bg_sigma"]
    smooth_sigma = d["smooth_sigma"]
    marker_min_dist = d["marker_min_dist"]
    min_object_area = d["min_object_area"]
    erode_iters = d["erode_iters"]

    bg = ndimage.gaussian_filter(image, sigma=bg_sigma, mode='reflect')
    hp = image - bg
    sm = ndimage.gaussian_filter(hp, sigma=smooth_sigma, mode='reflect')

    if clip_pct is None:
        thr_hist = np.clip(sm, 0, None)
    else:
        thr_hist = np.clip(sm, 0, np.percentile(sm, clip_pct))
    thr = filters.threshold_otsu(thr_hist) * params.otsu_frac
    mask = sm > thr
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=params.close_iters)
    # remove_small_objects now uses max_size semantics: remove objects with size <= max_size
    mask = morphology.remove_small_objects(mask, max_size=min_object_area)

    dist = ndimage.distance_transform_edt(mask)

    dist_for_markers = dist
    if params.marker_smooth_sigma > 0:
        dist_for_markers = ndimage.gaussian_filter(dist, sigma=params.marker_smooth_sigma, mode='nearest')

    markers = peak_local_max(
        dist_for_markers,
        min_distance=marker_min_dist,
        labels=mask,
        threshold_rel=params.marker_rel,
        exclude_border=True
    )

    mlab = np.zeros(image.shape, dtype=int)
    if markers.size > 0:
        mlab[markers[:, 0], markers[:, 1]] = np.arange(1, len(markers) + 1)

    lab = watershed(-dist, mask=mask, markers=mlab, connectivity=1)

    keep = set(r.label for r in regionprops(lab) if r.area > min_object_area)
    lab = np.isin(lab, list(keep)) * lab
    lab, _, _ = relabel_sequential(lab)
    n_watershed = int(lab.max())

    lab, n_merged = _merge_over_splits(lab, dist, params.merge_ridge_frac)
    dropped_map = lab.copy()
    lab, dropped = _extent_filter(lab, erode_iters, params.min_eroded_extent)
    lab, _, _ = relabel_sequential(lab)

    if return_intermediates:
        intermediates = dict(
            highpass=hp,
            smooth=sm,
            otsu_thr=float(thr),
            mask=mask,
            distance=dist,
            markers=markers,
            n_markers=int(len(markers)),
            n_merged=int(n_merged),
            n_watershed=int(n_watershed),
            dropped=dropped,
            dropped_map=dropped_map,
        )
        return lab, intermediates
    return lab


def segment_nuclei_otsu_watershed(
    image,
    *,
    cell_sigma: float = 16.0,
    otsu_frac: float = 0.3,
    min_eroded_extent: float = 0.90,
    clip_pct: float | None = None,
):
    """
    Friendly wrapper exposing the three main user knobs.
    """
    params = SimpleSegParams(
        cell_sigma=cell_sigma,
        otsu_frac=otsu_frac,
        min_eroded_extent=min_eroded_extent,
    )
    return detect_objects(image, params, clip_pct=clip_pct)
