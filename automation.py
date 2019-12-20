import logging
import json
import os
import shutil
import threading
import traceback
from typing import Iterable

from xmlrpc.client import ServerProxy

import numpy as np
from skimage.external.tifffile import imread
from skimage.transform import AffineTransform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from calmutils.imageio import read_bf
import nis_util
from annotation import manually_correct_rois
from simple_detection import scale_bbox

ND2_SUFFIX = '.nd2'
TIFF_SUFFIX = '.tif'


def copy_lock(src, dst, copyfun=shutil.copy2, lock_ending='lock'):
    lock_file = '.'.join([dst if not os.path.isdir(dst) else os.path.join(dst, src.rsplit(os.sep, 1)[-1]), lock_ending])
    fd = open(lock_file, 'w')
    fd.close()

    copyfun(src, dst)
    os.remove(lock_file)


def copy_lock_to_dir(src, dst, copyfun=shutil.copy2, lock_ending='lock'):
    if not isinstance(src, list):
        src = [src]

    if not os.path.exists(dst):
        os.makedirs(dst)

    if os.path.isfile(dst):
        raise ValueError('destination has to be a dirctory')

    for s in src:
        copy_lock(s, dst, copyfun, lock_ending)


def _pix2unit(x, transform):
    """
    transform a point from pixel coordinates to NIS stage coordinates,
    taking into account offsets, fov, camera rotation or image flipping

    Parameters
    ----------
    x: 2-tuple
        point to transform, in pixels
    transform: AffineTransform
        affine transform pixel -> stage

    Returns
    -------
    x_tr: array-like
        transformed point, in units
    """
    logger = logging.getLogger(__name__)

    res = np.squeeze(transform(x))
    logger.debug('transformed point {} (pixels) to {} (units)'.format(x, res))
    return res


def bbox_pix2unit(bbox, transform):
    """
    Parameters
    ----------
    x: 4-tuple
        point to transform, in pixels
    transform: AffineTransform
        affine transform pixel -> stage

    Returns
    -------
    bbox_tr: 4-tuple
        transformed bounding box (ymin, xmin, ymax, xmax - in units)
    """

    logger = logging.getLogger(__name__)

    # transform bbox
    (ymin, xmin, ymax, xmax) = bbox
    bbox_tr = np.apply_along_axis(lambda x: _pix2unit(x, transform),
                                  1,
                                  np.array([[xmin, ymin],
                                            [xmin, ymax],
                                            [xmax, ymin],
                                            [xmax, ymax]], dtype=float)
                                  )

    # get new min max
    min_ = np.apply_along_axis(np.min, 0, bbox_tr)
    max_ = np.apply_along_axis(np.max, 0, bbox_tr)

    logger.debug('new min: {}, new max: {}'.format(min_, max_))

    # NB: we reverse here to preserve original ymin, xmin, ymax, xmax - order
    bbox_tr_arr = np.array([list(reversed(list(min_))), list(reversed(list(max_)))], dtype=float)
    res = bbox_tr_arr.ravel()

    logger.debug('bbox: {}, toUnit: {}'.format(bbox, res))
    return tuple(list(res))


class WidgetProgressIndicator:
    """
    thin wrapper around an ipywidgets widget to display progress, e.g. progress bar
    and an optional (text) status widget

    Parameters
    ----------
    progress_widget: ipywidgets widget
        widget to display progress, e.g. FloatProgress
    status_widget: ipywidgets widget, optional
        widget to display a status message, e.g. Label
    min: numeric, optional
        value of progress_widget corresponding to 0
    max: numeric, optional
        value of progress_widget corresponding to 1, default 100
    """

    def __init__(self, progress_widget, status_widget=None, min=0, max=100):
        self.progress_widget = progress_widget
        self.status_widget = status_widget
        self.min = min
        self.max = max

    def set_progress(self, p):
        """
        update progress
        Parameters
        ----------
        p: float \in 0,1
            percent complete value to set
        """
        self.progress_widget.value = self.min + p * (self.max - self.min)

    def set_status(self, status):
        """
        update status
        status: string
            status message
        """
        if self.status_widget is not None:
            self.status_widget.value = status


class CommonParameters(object):
    def __init__(self):
        self.prefix = 'myExperiment'
        self.path_to_nis = '/dev/null'
        self.server_path_local = '/dev/null'
        self.save_base_path = '/dev/null'
        self.server_path_remote = '/dev/null'

        self.progress_indicator : WidgetProgressIndicator = None

class OverviewParameters(object):

    def __init__(self):
        self.export_as_tiff = False
        self.field_def_file = '/dev/null'
        self.manual_z = None
        self.oc_overview = 'oc1'
        self.return_overview_img = True

        self._re_use_ov = False


class DetectionParameters(object):
    def __init__(self):
        self.do_manual_annotation = False
        self.detector_adress = 'http://eco-gpu:8000/'
        self.plot_detection = True

        # TODO: remove
        self.object_filter = {
            'area': (15000, 80000)
        }


class DetailParameters(object):
    def __init__(self):
        self.ocs_detail = []
        self.stitcher_adress = 'http://eco-gpu:8001/'
        self.auto_focus_detail = True
        self.channel_for_autofocus = 0
        self.channel_for_stitch = 0
        # TODO: make True always
        self.tiff_export_details = True

        # TODO: do actual parameters, not just project yes/no
        self.projection_params = True

        self.dry_run_details = False

        self.z_range = 1
        self.z_step = 1
        self.z_drive = ""

def _do_overview(config: CommonParameters, ov_parameters: OverviewParameters):
    # keep track of all copy threads so we can join on exit
    logger = logging.getLogger(__name__)

    # skip if result folder already exists on server
    if os.path.exists(os.path.join(config.server_path_local, config.prefix)):
        if not ov_parameters._re_use_ov:  # skip overwrite check if we re-use overview (we are probably debuging)
            raise ValueError(
                'Slide {} was already imaged. Either rename the acquisition or clean old acquisition from server'.format(
                    config.prefix))

    # reset progress indicator
    progress_indicator = config.progress_indicator
    if progress_indicator is not None:
        progress_indicator.set_progress(0.0)
        progress_indicator.set_status('doing overview scan')

    with open(ov_parameters.field_def_file, 'r') as fd:
        field_calib = json.load(fd)

    # user specified manual focus position
    if not ov_parameters.manual_z is None:
        field_calib['zpos'] = ov_parameters.manual_z

    # go to defined z position
    nis_util.set_position(config.path_to_nis, pos_z=field_calib['zpos'])

    # get field and directions
    # NB: this is not the actual field being scanned, but rather [min+1/2 fov - max-1/2fov]
    (left, right, top, bottom) = tuple(field_calib['bbox'])


    # direction of stage movement (y,x)
    # TODO: remove if not necessary
    #direction = [1 if top < bottom else -1, 1 if left < right else -1]

    # set overview optical configuration
    nis_util.set_optical_configuration(config.path_to_nis, ov_parameters.oc_overview)

    # get resolution, binning and fov
    (xres, yres, siz, mag) = nis_util.get_resolution(config.path_to_nis)
    live_fmt, capture_fmt = nis_util.get_camera_format(config.path_to_nis)
    color = nis_util.is_color_camera(config.path_to_nis)

    # we have to parse capture_fmt differently for color and gray camera
    # TODO: extract to separate function
    if color:
        binning_factor = 1.0 if not '1/3' in capture_fmt else 3.0
    else:
        binning_factor = float(capture_fmt.split()[1].split('x')[0])

    fov_x = xres * siz / mag * binning_factor
    fov_y = yres * siz / mag * binning_factor

    logger.debug('overview resolution: {}, {}, {}, {}'.format(xres, yres, siz, mag))
    logger.debug('overview fov: {}, {}'.format(fov_x, fov_y))

    # do overview scan
    # TODO: save directly to server, not c:
    ov_path = os.path.join(config.save_base_path, config.prefix + '_overview' + ND2_SUFFIX)
    if not ov_parameters._re_use_ov:
        nis_util.do_large_image_scan(config.path_to_nis, ov_path, left, right, top, bottom)

    if ov_parameters.export_as_tiff:
        nis_util.export_nd2_to_tiff(config.path_to_nis, ov_path, combine_c=color)
        tiff_ov_path = ov_path[:-len(ND2_SUFFIX)] + TIFF_SUFFIX

    # make root folder for slide on server, skip if it already exists
    if not os.path.exists(os.path.join(config.server_path_local, config.prefix)):
        os.makedirs(os.path.join(config.server_path_local, config.prefix))

    # export optical configurations to server (we save it for every slide)
    nis_util.backup_optical_configurations(config.path_to_nis, os.path.join(config.server_path_local, config.prefix,
                                                                            'optical_configurations.xml'))

    # copy field definition to server
    shutil.copy2(ov_parameters.field_def_file, os.path.join(config.server_path_local, config.prefix))


    # keep image in memory if we need it
    # TODO: maybe read directly from server?
    if ov_parameters.return_overview_img:
        if ov_parameters.export_as_tiff:
            img = imread(tiff_ov_path)
        else:
            img = read_bf(ov_path)

    # async copy to server
    def copy_ov_call():
        # copy to server mount
        _ov_path = ov_path
        _tiff_ov_path = tiff_ov_path
        _prefix = config.prefix

        if not os.path.exists(os.path.join(config.server_path_local, _prefix, 'overviews')):
            os.makedirs(os.path.join(config.server_path_local, _prefix, 'overviews'))

        if not ov_parameters.export_as_tiff:
            copy_lock(_ov_path, os.path.join(config.server_path_local, _prefix, 'overviews'))
        else:
            copy_lock(_tiff_ov_path, os.path.join(config.server_path_local, _prefix, 'overviews'))

        # remove local copies of overviews
        os.remove(_ov_path)
        if ov_parameters.export_as_tiff:
            os.remove(_tiff_ov_path)

    # copy in separate thread
    # TODO: probably not really necessary, remove
    copy_ov_thread = threading.Thread(target=copy_ov_call)
    copy_ov_thread.start()
    # NB: we have to wait for copy to complete before we initialize the detection on server
    copy_ov_thread.join()

    if ov_parameters.return_overview_img:
        return img



def _do_detection(config: CommonParameters, ov_parameters: OverviewParameters, det_params: DetectionParameters, img=None):
    logger = logging.getLogger(__name__)
    logger.info('finished overview, detecting wings...')

    with open(ov_parameters.field_def_file, 'r') as fd:
        field_calib = json.load(fd)

    # pixel to world coordinates transformation from 3-point calibration stored in field_calib file
    coords_px = np.array(field_calib['coords_px'], dtype=np.float)
    coords_st = np.array(field_calib['coords_st'], dtype=np.float)
    at = AffineTransform()
    at.estimate(coords_px, coords_st)

    _suffix = TIFF_SUFFIX if ov_parameters.export_as_tiff else ND2_SUFFIX
    remote_path = '/'.join([config.server_path_remote, config.prefix, 'overviews', config.prefix + '_overview' + _suffix])

    progress_indicator = config.progress_indicator
    if progress_indicator is not None:
        progress_indicator.set_status('detecting wings')

    # where to save the segmentation to
    label_export_path = '/'.join([config.server_path_remote, config.prefix, 'overviews', config.prefix + '_segmentation' + TIFF_SUFFIX])

    # do the detection
    with ServerProxy(det_params.detector_adress) as proxy:
        bboxes = proxy.detect_bbox(remote_path, det_params.object_filter, label_export_path)
        bboxes = [] if bboxes is None else bboxes
        print(bboxes)

    if det_params.do_manual_annotation:
        annots = manually_correct_rois(img, [[x0, y0, x1 - x0, y1 - y0] for (y0, x0, y1, x1) in bboxes],
                                       [1] * len(bboxes))
        annotation_out = os.path.join(config.server_path_local, config.prefix, 'overviews', config.prefix + '_manualrois.json')
        with open(annotation_out, 'w') as fd:
            json.dump([a.to_json() for a in annots], fd)
        bboxes = [a.roi for a in annots]
        logger.debug(bboxes)
        # change to other format
        bboxes = [[y0, x0, y0 + h, x0 + w] for (x0, y0, w, h) in bboxes]
        logger.debug(bboxes)

    # save rois, regardless of wheter we did manual annotation or not
    annotation_out = os.path.join(config.server_path_local, config.prefix, 'overviews', config.prefix + '_autorois.json')
    bboxes_json = [{'y0': int(y0), 'y1': int(y1), 'x0': int(x0), 'x1': int(x1)} for (y0, x0, y1, x1) in bboxes]
    with open(annotation_out, 'w') as fd:
        json.dump(bboxes_json, fd)

    if det_params.plot_detection and img is not None:
        plt.figure()
        plt.imshow(img)

    # extract binning factor again
    # set overview optical configuration
    nis_util.set_optical_configuration(config.path_to_nis, ov_parameters.oc_overview)
    # get resolution, binning and fov
    (xres, yres, siz, mag) = nis_util.get_resolution(config.path_to_nis)
    live_fmt, capture_fmt = nis_util.get_camera_format(config.path_to_nis)
    color = nis_util.is_color_camera(config.path_to_nis)
    # we have to parse capture_fmt differently for color and gray camera
    # TODO: extract to separate function
    if color:
        binning_factor = 1.0 if not '1/3' in capture_fmt else 3.0
    else:
        binning_factor = float(capture_fmt.split()[1].split('x')[0])

    bboxes_scaled = []
    for bbox in bboxes:
        # upsample bounding boxes if necessary
        bbox_scaled = np.array(tuple(bbox)) * binning_factor
        logger.debug('bbox: {}, upsampled: {}'.format(bbox, bbox_scaled))
        bboxes_scaled.append(bbox_scaled)

        # plot bbox
        if det_params.plot_detection and img is not None:
            minr, minc, maxr, maxc = tuple(list(bbox))
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
    if det_params.plot_detection and img is not None:
        plt.show()

    # use scaled bboxes from here on
    bboxes = bboxes_scaled

    # pixels to units
    bboxes = [bbox_pix2unit(b, at) for b in bboxes]

    # expand bounding boxes
    bboxes = [scale_bbox(bbox, expand_factor=.2) for bbox in bboxes]

    logger.info('detected {} wings:'.format(len(bboxes)))

    return bboxes


def _do_detail(bboxes, config: CommonParameters, detail_params: DetailParameters):
    logger = logging.getLogger(__name__)

    threads = []

    try:
        # scan the individual wings
        for idx, bbox in enumerate(bboxes):

            logger.info('scanning wing {}: {}'.format(idx, bbox))

            (ymin, xmin, ymax, xmax) = bbox

            # TODO: do we need direction?
            '''
            (ymin, xmin, ymax, xmax) = (ymin if direction[0] > 0 else ymax,
                                        xmin if direction[1] > 0 else xmax,
                                        ymin if direction[0] < 0 else ymax,
                                        xmin if direction[1] < 0 else xmax)
            '''

            # set overview optical configuration
            nis_util.set_optical_configuration(config.path_to_nis, detail_params.ocs_detail[detail_params.channel_for_stitch])
            color = nis_util.is_color_camera(config.path_to_nis)


            # set oc so we have correct magnification
            nis_util.set_optical_configuration(config.path_to_nis, detail_params.ocs_detail[detail_params.channel_for_autofocus])

            # do autofocus -> move to wing center and focus
            if detail_params.auto_focus_detail:
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                nis_util.set_position(config.path_to_nis, [x_center, y_center])
                nis_util.do_autofocus(config.path_to_nis)

            wing_path = os.path.join(config.save_base_path, config.prefix + '_wing' + str(idx) + ND2_SUFFIX)


            # set to first oc so we have correct magnification
            nis_util.set_optical_configuration(config.path_to_nis, detail_params.ocs_detail[detail_params.channel_for_autofocus])

            # get resolution
            (xres, yres, siz, mag) = nis_util.get_resolution(config.path_to_nis)
            fov = nis_util.get_fov_from_res(nis_util.get_resolution(config.path_to_nis))
            logger.debug('detail resolution: {}, {}, {}, {}'.format(xres, yres, siz, mag))
            logger.debug('fov: {}'.format(fov))

            # get fov
            fov_x = xres * siz / mag
            fov_y = yres * siz / mag

            # generate the coordinates of the tiles
            grid, tilesX, tilesY, overlap = nis_util.gen_grid(fov, [xmin, ymin], [xmax, ymax], 0.15, True, True, True)

            for g in grid:
                logger.debug('wing {}: will scan tile at {}'.format(idx - 1, g))

            # do not actually do the detail acquisition
            if detail_params.dry_run_details:
                continue

            # NB: we have multiple channels, so we have to do
            # manual grid acquisition via multipoint nD acquisition -> has to be stitched afterwards

            # we scan around current z -> get that
            pos = nis_util.get_position(config.path_to_nis)

            # setup nD acquisition
            nda = nis_util.NDAcquisition(wing_path)
            nda.set_z(int(detail_params.z_range / 2), int(detail_params.z_range / 2), int(detail_params.z_step), detail_params.z_drive)
            nda.add_points(map(lambda x: (x[0], x[1], pos[2] - pos[3]), grid))

            for oc in detail_params.ocs_detail:
                nda.add_c(oc)

            nda.prepare(config.path_to_nis)
            nda.run(config.path_to_nis)

            if detail_params.tiff_export_details:
                wing_out_dir = wing_path[:-len(ND2_SUFFIX)]
                if not os.path.exists(wing_out_dir):
                    os.makedirs(wing_out_dir)
                nis_util.export_nd2_to_tiff(config.path_to_nis, wing_path, wing_out_dir)

            def copy_details():
                # copy to server mount
                _wing_path = wing_path
                _wing_out_dir = wing_out_dir
                _tilesX, _tilesY, _overlap = tilesX, tilesY, overlap
                _prefix = config.prefix

                # make raw data directory on server
                if not os.path.exists(os.path.join(config.server_path_local, _prefix, 'raw')):
                    os.makedirs(os.path.join(config.server_path_local, _prefix, 'raw'))

                # copy raw data to server
                copy_lock(_wing_path, os.path.join(config.server_path_local, _prefix, 'raw'))
                if detail_params.tiff_export_details:
                    files = [os.path.join(_wing_out_dir, f) for f in os.listdir(_wing_out_dir) if
                             os.path.isfile(os.path.join(_wing_out_dir, f))]
                    copy_lock_to_dir(files, os.path.join(os.path.join(config.server_path_local, _prefix, 'raw'),
                                                         _wing_out_dir.rsplit(os.sep)[-1]))

                remote_path = '/'.join([config.server_path_remote, _prefix, 'raw',
                                        _wing_out_dir.rsplit(os.sep)[-1] if detail_params.tiff_export_details else
                                        _wing_path.rsplit(os.sep)[-1]])

                # make directories for final stitched files if necessary
                for oc in detail_params.ocs_detail:
                    if not os.path.exists(os.path.join(config.server_path_local, _prefix, oc)):
                        os.makedirs(os.path.join(config.server_path_local, _prefix, oc))

                # parameters for cleanup
                # move stitching to oc directories, delete raw tiffs and temporary stitching folder
                cleanup_params = {'stitching_path': remote_path + '_stitched',
                                  'outpaths': ['/'.join([config.server_path_remote, _prefix, oc]) for oc in detail_params.ocs_detail],
                                  'outnames': [_wing_out_dir.rsplit(os.sep)[-1] + TIFF_SUFFIX] * len(detail_params.ocs_detail),
                                  'raw_paths': [remote_path],
                                  'delete_raw': True,
                                  'delete_stitching': True
                                  }

                with ServerProxy(detail_params.stitcher_adress) as proxy:
                    proxy.stitch([remote_path, _tilesX, _tilesY, _overlap, detail_params.channel_for_stitch + 1 if not color else 'RGB'], detail_params.tiff_export_details,
                                 cleanup_params, detail_params.projection_params)

                # cleanup local
                os.remove(_wing_path)
                if detail_params.tiff_export_details:
                    shutil.rmtree(_wing_out_dir)


            copy_det_thread = threading.Thread(target=copy_details)
            threads.append(copy_det_thread)
            copy_det_thread.start()

            # update progress
            progress_indicator = config.progress_indicator
            if progress_indicator is not None:
                progress_indicator.set_progress((idx + 1) / len(bboxes))
                progress_indicator.set_status('scanning wing {}'.format(idx + 1))

    except KeyboardInterrupt:
        logger.info('Interrupted by user, stopping...')

    except Exception:
        traceback.print_exc()

    finally:
        progress_indicator = config.progress_indicator
        if progress_indicator is not None:
            progress_indicator.set_progress(1.0)
            progress_indicator.set_status('finishing copy to server')

        logger.info('Waiting for all copy threads to finish...')
        for t in threads:
            t.join()
        logger.info('Done.')




def do_scan(configs: Iterable[CommonParameters], ov_params: Iterable[OverviewParameters],
                det_params: Iterable[DetectionParameters], detail_params: Iterable[DetailParameters],
                callback_aftereach=None, callback_beforeeach=None):

    for (config, ov_param, det_param, detail_param) in zip (configs, ov_params, det_params, detail_params):
        if callback_beforeeach is not None:
            callback_beforeeach()
        img = _do_overview(config, ov_param)
        bboxes = _do_detection(config, ov_param, det_param, img)
        _do_detail(bboxes, config, detail_param)
        if callback_aftereach is not None:
            callback_aftereach()


def do_scan_detection_first(configs: Iterable[CommonParameters], ov_params: Iterable[OverviewParameters],
                det_params: Iterable[DetectionParameters], detail_params: Iterable[DetailParameters]):

    bboxes_acc = []
    for (config, ov_param, det_param) in zip(configs, ov_params, det_params):
        img = _do_overview(config, ov_param)
        bboxes = _do_detection(config, ov_param, det_param, img)
        bboxes_acc.append(bboxes)

    for (config, detail_param, bboxes) in zip(configs, detail_params, bboxes_acc):
        _do_detail(bboxes, config, detail_param)

