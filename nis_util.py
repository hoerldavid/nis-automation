import subprocess
import configparser
from tempfile import NamedTemporaryFile
import os
from math import ceil


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
    overlap: scalar \in (0,1)
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
        
    return res


def quote(s):
    return '"{}"'.format(s)


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
    
    res = None
    
    try:
        ntf = NamedTemporaryFile(suffix='.mac', delete=False)
        ntf2 = NamedTemporaryFile(suffix='.ini', delete=False)
        ntf2.close()
        
        cmd = '''
        int flip;
        int rot180;
        double rotation;
        double rotation2;

        CameraGet_Cam0Flip(1,&flip);
        CameraGet_Cam0Rotate180(1,&rot180);
        CameraGet_Rotate(1,&rotation);
        Camera_RotateGet(&rotation2);
        
        Int_SetKeyValue("{0}","res","flip",flip);
        Int_SetKeyValue("{0}","res","rot180",rot180);
        Int_SetKeyValue("{0}","res","rotation",rotation);
        Int_SetKeyValue("{0}","res","rotation2",rotation2);


        '''.format(*[ntf2.name])        
        
        ntf.writelines([bytes(cmd, 'utf-8')])
        
        ntf.close()
        subprocess.call(' '.join([quote(path_to_nis), '-mw', quote(ntf.name)])) 
        
        config = configparser.ConfigParser()
        config.read(ntf2.name)
        
        res = (config['res']['flip'], config['res']['rot180'], config['res']['rotation'], config['res']['rotation2'])       
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
            sprintf(&buf, "conf%i", "i" );
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


class NDAcquisition:
    def __init__(self, outfile):
        self.outfile = outfile
        self.t = None
        self.xy = None
        self.c = None
        self.z = None
        self.z_device = None
        
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
            lines.append('ND_AppendMultipointPoint({},{},{},"{}");'.format(*self.xy[i] + [i]))
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
        print(cmd)
        return cmd
    
    def prepare(self, path_to_nis):
        try:
            ntf = NamedTemporaryFile(suffix='.mac', delete=False)
            l = [self.t, self.xy, self.z, self.c]
            b = list(map(lambda x: 1 if x else 0, l))
            ntf.writelines([bytes('ND_DefineExperiment({},{},{},{},0,"{}","",0,0,0,0);'.format(*b + [self.outfile]), 'utf-8')])
            
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