import os
import time
import subprocess
from tempfile import NamedTemporaryFile

NIS_EXE = r'C:\Program Files\NIS-Elements\nis_ar.exe'
ROOT = r'C:\Users\David\Desktop\nis-automation'
OUT = os.path.join(ROOT, 'test_stim_saved.nd2')
SCAN_DIRS = [ROOT, r'C:\Users\David\Documents', r'C:\Users\David']

if os.path.exists(OUT):
    os.remove(OUT)

before = {}
for d in SCAN_DIRS:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith('.nd2'):
                p = os.path.join(d, f)
                before[p] = os.path.getmtime(p)

cmd = 'ND_DefineExperiment(-1,-1,-1,-1,-1,"{}","",-1,-1,-1,-1);\nND_RunSequentialStimulationExp();'.format(OUT)
ntf = NamedTemporaryFile(suffix='.mac', delete=False)
try:
    ntf.writelines([bytes(cmd, 'utf-8')])
    ntf.close()
    print('Running ND_DefineExperiment + ND_RunSequentialStimulationExp ...')
    t0 = time.time()
    subprocess.call('"{}" -mw "{}"'.format(NIS_EXE, ntf.name))
    print(f'macro returned after {time.time()-t0:.1f}s')
finally:
    try:
        os.remove(ntf.name)
    except PermissionError:
        pass

time.sleep(2)
print('target file saved:', os.path.exists(OUT))
new = []
for d in SCAN_DIRS:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith('.nd2'):
                p = os.path.join(d, f)
                if p not in before or os.path.getmtime(p) > before[p]:
                    new.append(p)
print('new nd2 files anywhere:', new if new else 'NONE')
