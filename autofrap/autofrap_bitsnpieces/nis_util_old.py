# FROZEN SNAPSHOT of the pre-refactor nis_util.py (before the shared
# _run_macro helper was extracted, TODO #5). Used by
# test_nis_util_refactor.py to compare the generated .mac macro bodies
# byte for byte. Do not edit.
import subprocess
import configparser
from tempfile import NamedTemporaryFile
import os
from math import ceil
from shutil import move
import logging

from resources import dummy_nd2

# autofocus constants
DEFAULT_FOCUS_CRITERION = 0
DEFAULT_FOCUS_STEP_COARSE = 10.0
DEFAULT_FOCUS_STEP_FINE = 1.0

# NIS flags for exporting as multipage TIFF
EXPORT_TIFF_MASK_T = 1
EXPORT_TIFF_MASK_XY = 16
EXPORT_TIFF_MASK_Z = 256

# prefix for the named tiles in a multipoint ND-acq.
TILE_NAME_PREFIX = 'tile'

# export to TIFF needs a prefix -> dummy that is removed later
EXPORT_DUMMY_PREFIX = '$tiffexport$'


#TODO: color camera centercrop
'''
InputHWUnit("DS-Ri2 Direct 1.0x", "Ri2_Camera_Color_SN_703130")
InputHWUnit("DS-Ri2 Direct 1.0x", "Ri2_Camera_Mono_Emulated_SN_703130")
InputHWUnit("DS-Ri2 Zoom 2.5x", "Ri2_Camera_Mono_Emulated_SN_703130")
InputHWUnit("DS-Ri2 Zoom 2.5x", "Ri2_Camera_Color_SN_703130")
CameraFormatSet(1, "3x8_Kaiser_Full_Area_2.5x_1/3");
CameraFormatSet(1, "3x8_Kaiser_Full_Area_2.5x");
InputHWUnit("DS-Ri2 Direct 1.0x", "Ri2_Camera_Color_SN_703130");
CameraFormatSet(1, "3x8_Kaiser_Center_Scan");
CameraFormatSet(1, "3x8_Kaiser_Center_Scan_1/3");
'''

#def set_camera(path_to_nis, camera_type='grey'):
    
    

def is_color_camera(path_to_nis):
    '''
    Hacky check if we have the color camera active
    '''
    live_fmt, capture_fmt = get_camera_format(path_to_nis)
    return '3x8_Kaiser' in capture_fmt


def export_nd2_to_tiff(path_to_nis, nd2_file, out_dir=None, combine_t=False, combine_yx=False, combine_z=True, combine_c=False):
    # NB: suffix order should be t, xy, z, c (?)

    if out_dir is None:
        # same dir
        out_dir = nd2_file.rsplit(os.sep, 1)[0]

    # multipage mask
    mask = 0
    if combine_t:
        mask += EXPORT_TIFF_MASK_T
    if combine_yx:
        mask += EXPORT_TIFF_MASK_XY
    if combine_z:
        mask += EXPORT_TIFF_MASK_Z

    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        cmd = '''
        char emptystring[1];
        ND_ExportToTIFF("{}","{}","{}",{},0,0,{});
        '''.format(nd2_file, out_dir, EXPORT_DUMMY_PREFIX, 3 if combine_c else 0, mask)

        ntf.writelines([bytes(cmd, 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)

    # NIS needs a prefix for export -> manually remove the dummy prefix and rename files
    for f in os.listdir(out_dir):
        if f.startswith(EXPORT_DUMMY_PREFIX):
            move(os.path.join(out_dir, f), os.path.join(out_dir, f.replace(EXPORT_DUMMY_PREFIX, '')))


def gen_grid(fov, min_, max_, overlap, snake, half_fov_offset=True, center=True):
    """
    generate a grid of coordinates at which to do a tiled acquisition

    Parameters
    ----------
    fov: array-like
        field-of-view in units
    min_: array-like
        minimum of bbox to scan
    max_: array-like
        maximum of bbox to scan
    overlap: scalar \\in (0,1)
        percent overlap
    snake: boolean
        whether to alternate in x or not
    half_fov_offset: boolean
        whether to correct for NIS 'centering' on locations (-> half FOV offset)
    center: boolean
        whether to center the grid on the bounding box or not (in this case, the object will be in the upper left corner)

    Returns
    -------
    grid: list of 2-tuples
        (x,y) - coordinates at which to image
    """

    # whether coordinates are increasing or decreasing in a dimension
    direction = [1 if max_[0] > min_[0] else -1, 1 if max_[1] > min_[1] else -1]

    # number of tiles
    tilesX = (abs(max_[0] - min_[0]) - fov[0]) / (fov[0] * (1 - overlap))
    tilesY = (abs(max_[1] - min_[1]) - fov[1]) / (fov[1] * (1 - overlap))
    tilesX = max(0, int(ceil(tilesX))) + 1
    tilesY = max(0, int(ceil(tilesY))) + 1

    # re-center grid on bbox
    if center:
        totalX = fov[0] + (tilesX - 1) * (fov[0] * (1 - overlap))
        totalY = fov[1] + (tilesY - 1) * (fov[1] * (1 - overlap))

        #print('{} {}'.format(totalX, totalY))
        extraX = totalX - abs(max_[0] - min_[0])
        extraY = totalY - abs(max_[1] - min_[1])

        #print('{} {}'.format(extraX, extraY))
        min_ = [min_[0] - 0.5 * extraX * direction[0], min_[1] - 0.5 * extraY * direction[1]]

    # correct for NIS's half FOV offset
    if half_fov_offset:
        min_ = [min_[0] + 0.5 * fov[0] * direction[0], min_[1] + 0.5 * fov[1] * direction[1]]

    # steps: increasing or decreasing
    stepX = fov[0] * (1 - overlap) if direction[0] == 1 else - (fov[0] * (1 - overlap))
    stepY = fov[1] * (1 - overlap) if direction[1] == 1 else - (fov[1] * (1 - overlap))

    res = []
    for y in range(tilesY):
        row = [(min_[0] + x * stepX, min_[1] + y * stepY) for x in range(tilesX)]
        if snake and (y % 2 != 0):
            row.reverse()
        res.extend(row)
        
    return res, tilesX, tilesY, overlap


def quote(s):
    return '"{}"'.format(s)


def backup_optical_configurations(path_to_nis, backup_path):
    """
    export all optical configurations as XML
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        cmd = '''BackupOptConf("{}");'''.format(backup_path)
        ntf.writelines([bytes(cmd, 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def do_autofocus(path_to_nis, step_coarse=None, step_fine=None, focus_criterion=None, focus_with_piezo=False):
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        cmd = '''
        StgZ_SetActiveZ({});
        StgFocusSetCriterion({});
        StgFocusAdaptiveTwoPasses({},{});
        Freeze();
        '''.format(1 if focus_with_piezo else 0,
                   focus_criterion if not focus_criterion is None else DEFAULT_FOCUS_CRITERION,
                   step_coarse if not step_coarse is None else DEFAULT_FOCUS_STEP_COARSE,
                   step_fine if not step_fine is None else DEFAULT_FOCUS_STEP_FINE,)
        ntf.writelines([bytes(cmd, 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def set_optical_configuration(path_to_nis, oc_name):
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        cmd = 'SelectOptConf("{0}");'.format(*[oc_name])
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def do_large_image_scan(path_to_nis, save_path,
                   left, right, top, bottom,
                   overlap = 0,
                   registration = False, z_count=1, z_step=1.5, close=True ):
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        cmd = '''
        Stg_SetLargeImageStageZParams({}, {}, {});
        Stg_LargeImageScanArea({},{},{},{},0,{},0,{},0,"{}");
        {}
        '''.format(0 if (z_count <= 1) else 1, z_step, z_count,
            left, right, top, bottom, overlap, 1 if registration else 0, save_path, 'CloseCurrentDocument(2);' if close else '')
   
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)

        
def get_camera_format(path_to_nis):
    """
    get camera format string (should contain binning & bit depth)
    
    Parameters
    ----------
    path_to_nis: str
        path to the nis_ar.exe executable
        
    Returns
    -------
    live_format, capture_format: str
        format strings for live mode and capture mode
    """
    
    res = None
    
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()
        
        cmd = '''
        char fmt_live[256];
        char fmt_capture[256];
        
        CameraFormatGet(1, &fmt_live);
        CameraFormatGet(2, &fmt_capture);
        
        Int_SetKeyString("{0}","res","fmt_live", &fmt_live);
        Int_SetKeyString("{0}","res","fmt_capture", &fmt_capture);
        '''.format(*[ntf2.name])        
        
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)])) 
        
        config = configparser.ConfigParser()
        config.read(ntf2.name)
        
        res = (config['res']['fmt_live'], config['res']['fmt_capture'])
        
    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)
    
    return res    
    
    
def get_resolution(path_to_nis):
    
    res = None
    
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()
        
        cmd = '''
        int x;
        int y;
        double siz;
        double mag;
        GetCameraResolution(2,&x,&y,&siz);
        mag = GetCurrentObjMagnification();
        
        Int_SetKeyValue("{0}","res","xres",x);
        Int_SetKeyValue("{0}","res","yres",y);
        Int_SetKeyValue("{0}","res","siz",siz);
        Int_SetKeyValue("{0}","res","mag",mag);
        '''.format(*[ntf2.name])        
        
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)])) 
        
        config = configparser.ConfigParser()
        config.read(ntf2.name)
        
        res = (config['res']['xres'], config['res']['yres'], config['res']['siz'], config.get('res', 'mag'))
        
        res = tuple(map(float, res))
        
    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)
    
    return res


def get_rotation_matrix(path_to_nis):
    
    res = None
    
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()
        
        cmd = '''
        double a11;
        double a12;
        double a21;
        double a22;

        Get_CalibrationAngleMatrix(0,&a11,&a12,&a21,&a22);
        
        Int_SetKeyValue("{0}","res","a11",a11);
        Int_SetKeyValue("{0}","res","a12",a12);
        Int_SetKeyValue("{0}","res","a21",a21);
        Int_SetKeyValue("{0}","res","a22",a22);
        '''.format(*[ntf2.name])        
        
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)])) 
        
        config = configparser.ConfigParser()
        config.read(ntf2.name)
        
        res = (config['res']['a11'], config['res']['a12'], config['res']['a21'], config.get('res', 'a22'))
        
        res = tuple(map(float, res))
        
    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)
    
    return res


def get_cam_rotation(path_to_nis):
    """
    get camera rotation angles [deg]

    Note: flip / 180-deg rotation of the optical path is not exposed by a
    separate macro function in this NIS-Elements version; it is part of the
    camera->stage calibration matrix (see get_rotation_matrix), which NIS
    documents as the "rotation and flip" transformation.

    Returns
    -------
    rotation, rotation2: (float, float)
        CameraGet_Rotate (camera property rotation) and Camera_RotateGet
        (current rotation), in degrees
    """
    
    res = None
    
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()
        
        cmd = '''
        double rotation;
        double rotation2;

        CameraGet_Rotate(1,&rotation);
        Camera_RotateGet(&rotation2);
        
        Int_SetKeyValue("{0}","res","rotation",rotation);
        Int_SetKeyValue("{0}","res","rotation2",rotation2);


        '''.format(*[ntf2.name])        
        
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)])) 
        
        config = configparser.ConfigParser()
        config.read(ntf2.name)
        
        res = (config['res']['rotation'], config['res']['rotation2'])       
        res = tuple(map(float, res))
        
    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)
    
    return res


def get_optical_confs(path_to_nis):
    res = None
    
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()
        
        cmd = '''
        int i;
        char buf[256];
        char name[256];
        
        Int_SetKeyValue("{0}","oc","count",GetOptConfCount());
        
        for(i=0; i < GetOptConfCount(); i=i+1)
        {{
            GetOptConfName(i, &name, 256);
            sprintf(&buf, "conf%i", i);
            Int_SetKeyString("{0}","oc",&buf,&name);
        }}        

        '''.format(*[ntf2.name])
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)])) 
        
        config = configparser.ConfigParser()
        config.read(ntf2.name)
        
        res = []
        for i in range(int(config.get('oc', 'count'))):
            res.append(config.get('oc', 'conf' + str(i)))
                
    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)
    
    return res


def set_position(path_to_nis, pos_xy=None, pos_z=None, pos_piezo=None, relative_xy=False, relative_z=False, relative_piezo=False):

    # nothing to do
    if pos_xy is None and pos_z is None and pos_piezo is None:
        return

    cmd = []
    if not pos_xy is None:
        cmd.append('StgMoveXY({},{},{});'.format(pos_xy[0], pos_xy[1], 1 if relative_xy else 0))
    if not pos_z is None:
        cmd.append('StgMoveMainZ({},{});'.format(pos_z, 1 if relative_z else 0))
    if not pos_piezo is None:
        cmd.append('StgMovePiezoZ({},{});'.format(pos_z, 1 if relative_piezo else 0))

    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf.writelines([bytes('\n'.join(cmd), 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def get_position(path_to_nis):
    res = None
    
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()
        
        cmd = '''
        double x;
        double y;
        double z_0;
        double z_1;
        StgGetPosXY(&x, &y);
        StgGetPosZ(&z_0, 0);
        
        if (StgZ_IsPresent(1))
        {{
            StgGetPosZ(&z_1, 1);
            Int_SetKeyValue("{0}","pos","z1",z_1);
        }}        
        
        Int_SetKeyValue("{0}","pos","x",x);
        Int_SetKeyValue("{0}","pos","y",y);
        Int_SetKeyValue("{0}","pos","z0",z_0);
        
        '''.format(*[ntf2.name])
        
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)])) 
        
        config = configparser.ConfigParser()
        config.read(ntf2.name)
        
        res = [config.get('pos', 'x'), config.get('pos', 'y'), 
               config.get('pos', 'z0'), config.get('pos', 'z1', fallback=None)]
        
        res = tuple(map(float, res))
    
    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)
    
    return res


def get_fov_from_res(res):
    return (res[0] * res[2] / res[3], res[1] * res[2] / res[3])


def grid_positions(path_to_nis, nx=2, ny=2, spacing=1.0):
    """
    compute a grid of stage positions centered on the current stage position

    Parameters
    ----------
    path_to_nis: str
        path to the nis_ar.exe executable
    nx, ny: int
        number of grid positions in x and y
    spacing: float
        distance between neighboring positions in units of FOV size:
        1 -> touching (non-overlapping) FOVs,
        <1 -> overlapping FOVs,
        >1 -> non-overlapping FOVs with a gap

    Returns
    -------
    positions: list of 2-tuples
        (x, y) stage positions, row-major order
    """
    fov_x, fov_y = get_fov_from_res(get_resolution(path_to_nis))
    x0, y0, _, _ = get_position(path_to_nis)

    step_x = spacing * fov_x
    step_y = spacing * fov_y

    return [(x0 + (i - (nx - 1) / 2) * step_x,
             y0 + (j - (ny - 1) / 2) * step_y)
            for j in range(ny) for i in range(nx)]


def run_current_nd_experiment(path_to_nis, outfile=None, open_after=True, progress_bar=True):
    """
    run the ND experiment as currently configured in the NIS GUI
    (ND Acquisition window), without rebuilding its definition

    All experiment parameters are passed as -1 ("keep current") to
    ND_DefineExperiment except the Filename, which acts as the save
    on/off switch: a non-empty **full path** (folder + filename) saves
    there (regardless of the GUI checkbox), an empty filename disables
    saving (the result then stays open in the GUI, and NIS may prompt
    "do you want to save?" when the acquisition closes).

    Parameters
    ----------
    path_to_nis: str
        path to the nis_ar.exe executable
    outfile: str, optional
        destination ND2 file, **full path** (folder + filename);
        None or empty -> no saving
    open_after: bool, default True
        True (default): keep the result open as the current document after
        the run. False: NIS closes the result after the run — with no save
        destination this triggers a save/discard/cancel dialog, so only use
        False when saving
    progress_bar: bool
        show the ND Experiment Acquisition Status window
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        cmd = 'ND_DefineExperiment(-1,-1,-1,-1,-1,"{}","",-1,-1,-1,-1);\n'.format(outfile or '')
        run_fn = 'ND_RunExperiment' if progress_bar else 'ND_RunExperimentNoProgressBar'
        cmd += '{}({});'.format(run_fn, 1 if open_after else 0)
        ntf.writelines([bytes(cmd, 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def run_stimulation_experiment(path_to_nis):
    """
    run the currently configured sequential stimulation ND experiment
    (ND_RunSequentialStimulationExp = "starts the current Sequential
    Stimulation ND experiment"); blocks until the experiment finishes

    the result stays open in the GUI as the current document (unsaved);
    save it with save_current_document
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf.writelines([bytes('ND_RunSequentialStimulationExp();', 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


# predefined NIS ROI colors (RGB values, see CreatePolygonROI docs)
ROI_COLORS = {
    'black': 0,
    'red': 255,
    'green': 65280,
    'yellow': 65535,
    'blue': 16711680,
    'magenta': 16711935,
    'cyan': 16776960,
    'white': 16777215,
    'default': 2147483647,
}


def open_image(path_to_nis, image_path):
    """
    open an image file (makes it the current document)
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf.writelines([bytes('ImageOpen("{}");'.format(image_path), 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def get_current_document(path_to_nis):
    """
    query the current GUI document (Get_Filename FILE_IMAGE)

    Returns
    -------
    name: str
        full path of the currently opened image file, or the document
        title for unsaved documents (e.g. 'ND Acquisition');
        raises KeyError if the query macro failed (like the other get_* functions)
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()

        cmd = '''
        char buf[1024];
        Get_Filename(5, buf);
        Int_SetKeyString("{0}","doc","path",buf);
        '''.format(ntf2.name)

        ntf.writelines([bytes(cmd, 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))

        config = configparser.ConfigParser()
        config.read(ntf2.name)
        return config.get('doc', 'path')

    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)


def save_current_document(path_to_nis, outfile):
    """
    save the current GUI document to outfile (ImageSaveAs)

    saves all layers of the document (ImType 15), lossless (ImCompr 0);
    the format is determined by the file extension (e.g. .nd2 = ND2 format)

    Parameters
    ----------
    path_to_nis: str
        path to the nis_ar.exe executable
    outfile: str
        full destination path
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf.writelines([bytes('ImageSaveAs("{}", 15, 0);'.format(outfile), 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def close_current_document(path_to_nis, save='discard'):
    """
    close the current GUI document (CloseCurrentDocument)

    Parameters
    ----------
    path_to_nis: str
        path to the nis_ar.exe executable
    save: {'ask', 'discard', 'yes'}, default 'discard'
        what to do if the document has unsaved changes:
        'ask'     - show the save/discard/cancel dialog (blocks!)
        'discard' - close without saving, no user interaction
        'yes'     - save the changes, then close
    """
    save_flag = {'ask': 0, 'discard': 2, 'yes': 1}[save]
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf.writelines([bytes('CloseCurrentDocument({});'.format(save_flag), 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def add_polygon_roi(path_to_nis, points, color='green'):
    """
    create a polygon ROI on the current image

    Parameters
    ----------
    points: sequence of (x, y)
        polygon vertices in pixel coordinates ((0,0) = top-left)
    color: str or int
        color name from ROI_COLORS or an RGB value

    Returns
    -------
    roi_id: int
        >0 on success, <0 on failure
    """
    color = ROI_COLORS.get(color, color)
    n = len(points)
    if n < 3:
        raise ValueError('a polygon needs at least 3 points')

    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()

        cmd = ['double pts[{}];'.format(2 * n)]
        for i, (x, y) in enumerate(points):
            cmd.append('pts[{}]={:.6f};'.format(2 * i, x))
            cmd.append('pts[{}]={:.6f};'.format(2 * i + 1, y))
        cmd.append('Int_SetKeyValue("{0}","roi","id",CreatePolygonROI(pts,{1},{2}));'.format(
            ntf2.name, n, color))

        ntf.writelines([bytes('\n'.join(cmd), 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))

        config = configparser.ConfigParser()
        config.read(ntf2.name)
        return int(config['roi']['id'])
    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)


def set_roi_type(path_to_nis, roi_id, roi_type):
    """
    set the type of an ROI (ChangeROIType)

    roi_type: 0 standard, 1 background, 2 reference, 3 stimulation
    note: type 1 *hides* the ROI — do not use
    (the macro's return value is unreliable; verify with get_roi_count if needed)
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf.writelines([bytes('ChangeROIType({}, {});'.format(roi_id, roi_type), 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def delete_roi(path_to_nis, roi_id):
    """
    remove an ROI from the current image (DeleteROI)
    (always returns 0 — verify with get_roi_count if needed)
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf.writelines([bytes('DeleteROI({});'.format(roi_id), 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
    finally:
        os.remove(ntf.name)


def get_roi_count(path_to_nis):
    """
    number of visible ROIs on the current image (GetROICount)
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()

        ntf.writelines([bytes('Int_SetKeyValue("{0}","roi","count",GetROICount());'.format(ntf2.name), 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))

        config = configparser.ConfigParser()
        config.read(ntf2.name)
        return int(config['roi']['count'])

    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)


def get_roi_info(path_to_nis, roi_id):
    """
    read back parameters of an ROI (see GetROIInfo in the NIS manual)

    Returns
    -------
    dict with keys: bbox_l, bbox_t, bbox_r, bbox_b, center_x, center_y,
                   min_feret, max_feret, rotation, color
    """
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()

        cmd = '''
        int l, t, r, b, cx, cy, minf, maxf;
        double rot;
        dword col;
        GetROIInfo({1}, &l, &t, &r, &b, &cx, &cy, &minf, &maxf, &rot, &col);
        Int_SetKeyValue("{0}","roi","l",l);
        Int_SetKeyValue("{0}","roi","t",t);
        Int_SetKeyValue("{0}","roi","r",r);
        Int_SetKeyValue("{0}","roi","b",b);
        Int_SetKeyValue("{0}","roi","cx",cx);
        Int_SetKeyValue("{0}","roi","cy",cy);
        Int_SetKeyValue("{0}","roi","minf",minf);
        Int_SetKeyValue("{0}","roi","maxf",maxf);
        Int_SetKeyValue("{0}","roi","rot",rot);
        Int_SetKeyValue("{0}","roi","col",col);
        '''.format(ntf2.name, roi_id)

        ntf.writelines([bytes(cmd, 'utf-8')])
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))

        config = configparser.ConfigParser()
        config.read(ntf2.name)
        res = {k: int(config['roi'][k]) for k in ('l', 't', 'r', 'b', 'cx', 'cy', 'minf', 'maxf')}
        res = {'bbox_l': res['l'], 'bbox_t': res['t'], 'bbox_r': res['r'], 'bbox_b': res['b'],
               'center_x': res['cx'], 'center_y': res['cy'],
               'min_feret': res['minf'], 'max_feret': res['maxf'],
               'rotation': float(config['roi']['rot']), 'color': int(config['roi']['col'])}
        return res
    finally:
        os.remove(ntf.name)
        os.remove(ntf2.name)


class NDAcquisition:
    def __init__(self, outfile):
        self.outfile = outfile
        self.t = None
        self.xy = None
        self.c = None
        self.z = None
        self.z_device = None
        self.logger = logging.getLogger(__name__)
        
    def set_z(self, bottom, top, step, device_name=None):
        self.z = {'top':top, 'bottom':bottom, 'step':step}
        if device_name != None:
            self.z_device = device_name
    
    def add_points(self, points):
        for p in points:
            self.add_point(*p)
        
    def add_point(self, x, y, z):
        if not self.xy:
            self.xy = list()
        self.xy.append([x,y,z])
        
    def compile_xy_cmd(self):
        lines = list()
        lines.append('ND_SetMultipointExp(0,0,"","",0,0.00000,0.00000);')
        lines.append('ND_ResetMultipointExp();')
        lines.append('ND_KeepPFSOnDuringStageMove(1);')
        for i in range(len(self.xy)):
            lines.append('ND_AppendMultipointPoint({},{},{},"{}{}");'.format(*self.xy[i] + [TILE_NAME_PREFIX, i]))
        return '\n'.join(lines)
    
    def add_c(self, oc_name):
        if not self.c:
            self.c = list()
        self.c.append(oc_name)
    
    def compile_c_cmd(self):
        lines = list()
        lines.append('ND_SetLambdaExp(0);')
        lines.append('ND_ResetLambdaExp();')
        for oc in self.c:
            lines.append('ND_AppendLambdaChannel("{}","{}",0,"","",0,0.00000,0.00000);'.format(*(oc, oc)))
        return '\n'.join(lines)
        
    def compile_z_cmd(self):
        dev = str(self.z_device) if self.z_device != None else ''
        #cmd = 'StgZ_SetActiveZ({0});'.format(*[self.z_device]) if self.z_device != None else ''
        cmd = ('ND_ResetZSeriesExp();\nND_SetZSeriesExp(3,{top},0.00000,{bottom},{step},0,0,0,"'+ dev + '","","");').format(**self.z)
        self.logger.debug(cmd)
        return cmd
    
    def prepare(self, path_to_nis):
        try:
            ntf = NamedTemporaryFile(suffix='.mac', delete=False)
            l = [self.t, self.xy, self.z, self.c]
            b = list(map(lambda x: 1 if x else 0, l))
            ntf.writelines([bytes('''
                ND_ReuseExperiment("{}");
                ND_DefineExperiment({},{},{},{},0,"{}","",0,0,0,0);
                '''.format(*[dummy_nd2] + b + [self.outfile]), 'utf-8')])
            
            if self.z:
                ntf.writelines([bytes(self.compile_z_cmd(), 'utf-8')])  
            
            if self.xy:
                ntf.writelines([bytes(self.compile_xy_cmd(), 'utf-8')])
                
            if self.c:
                ntf.writelines([bytes(self.compile_c_cmd(), 'utf-8')])
                
            ntf.close()
            subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
            
        finally:
            os.remove(ntf.name)
            
    def run(self, path_to_nis):
        try:
            ntf = NamedTemporaryFile(suffix='.mac', delete=False)
                       
            # run cmd
            ntf.writelines([bytes('ND_RunExperiment(0);', 'utf-8')])
            
            ntf.close()
            subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)]))
        finally:
            os.remove(ntf.name)


if __name__ == '__main__':
    print(gen_grid([.6,.6], [1,0], [0,1], 0.0, True, True, True))