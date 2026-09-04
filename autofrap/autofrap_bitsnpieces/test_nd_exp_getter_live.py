"""
Live probe of the ND Acquisition *dialog* settings (read-only),
one small macro per question (each an independent .mac run, so a
failure in one query doesn't affect the others).

Queries the currently configured ND experiment definition — what the
ND Acquisition dialog shows and what ND_RunExperiment would run —
with no document open:

  1. tab active states          ND_IsAcqTabChecked("<tab>")
                               (verified names: Time, XY, Z, Lambda,
                               Large Image)
  2. number of time phases      ND_GetTimeLapsePhaseCount()
  3. phase 0 schedule           ND_GetTimePhaseSchedule(0, ...)
  4. XY multipoint position count  ND_MP_GetCount()
  5. Z series settings          ND_GetZSeriesExp(...)
  6. channel i settings         ND_GetLambdaChannel(i, ...) (i = 0, 1, ...
                               until the name buffer comes back untouched)

Notes from earlier probing:
  - ND_GetExperimentLoopSize queries the *open document* only (-9 with
    no document open) — not usable for the dialog.
  - NIS sprintf(buf, fmt, args) is not C-variadic: the third argument is
    a comma-separated string of variable names to substitute; use
    strcpy() for literal strings.

Usage (at the microscope, NIS open):
    python autofrap/autofrap_bitsnpieces/test_nd_exp_getter_live.py
"""
import os
import sys
import time

# repo root (for nis_util) — this script lives two levels down in autofrap/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nis_util

NIS = r'C:\Program Files\NIS-Elements\nis_ar.exe'
IN = nis_util.INI_PLACEHOLDER

# verified tab names (20260904, all ticked -> all returned 1)
TAB_CANDIDATES = [
    'Time', 'XY', 'Z', 'Lambda', 'Large Image',
]

MAX_CHANNELS = 8


def ask(label, body, keys, section='q'):
    """run one small macro; return dict of requested keys ({} on failure)"""
    try:
        cfg = nis_util._run_macro(NIS, body, ini=True)
        return {k: cfg[section][k] for k in keys}
    except Exception as e:
        print('  %s: FAILED (%s: %s)' % (label, type(e).__name__, e))
        return {}


def main():
    print('=== tab active states ===')
    for tab in TAB_CANDIDATES:
        body = 'Int_SetKeyValue("%s","q","on",ND_IsAcqTabChecked("%s"));' % (IN, tab)
        res = ask('tab %r' % tab, body, ['on'])
        if res:
            print('  %-12s active=%s' % (tab, res['on']))

    print('=== time ===')
    res = ask('phase count',
              'Int_SetKeyValue("%s","q","phases",ND_GetTimeLapsePhaseCount());' % IN,
              ['phases'])
    phases = int(res['phases']) if res else None
    if res:
        print('  time phases: %s' % res['phases'])
    if phases:
        body = '''
double interval, duration;
int loopcnt;
ND_GetTimePhaseSchedule(0, &interval, &duration, &loopcnt);
Int_SetKeyValue("%s","q","interval",interval);
Int_SetKeyValue("%s","q","duration",duration);
Int_SetKeyValue("%s","q","loopcnt",loopcnt);
''' % (IN, IN, IN)
        res = ask('phase 0 schedule', body, ['interval', 'duration', 'loopcnt'])
        if res:
            print('  phase 0: loopcnt=%s interval=%s ms duration=%s ms'
                  % (res['loopcnt'], res['interval'], res['duration']))

    print('=== xy multipoint ===')
    res = ask('position count',
              'Int_SetKeyValue("%s","q","count",ND_MP_GetCount());' % IN,
              ['count'])
    if res:
        print('  positions: %s' % res['count'])

    print('=== z ===')
    body = '''
int ztype, zcount, zhome_def, zclose;
double ztop, zhome, zbottom, zstep;
char zdevice[256];
char before[256];
char after[256];
ND_GetZSeriesExp(&ztype, &ztop, &zhome, &zbottom, &zstep, &zcount, &zhome_def, &zclose, &zdevice, &before, &after);
Int_SetKeyValue("%s","q","ztype",ztype);
Int_SetKeyValue("%s","q","ztop",ztop);
Int_SetKeyValue("%s","q","zbottom",zbottom);
Int_SetKeyValue("%s","q","zstep",zstep);
Int_SetKeyValue("%s","q","zcount",zcount);
Int_SetKeyString("%s","q","zdevice",zdevice);
''' % (IN, IN, IN, IN, IN, IN)
    res = ask('z series', body, ['ztype', 'ztop', 'zbottom', 'zstep', 'zcount', 'zdevice'])
    if res:
        print('  type=%s top=%s bottom=%s step=%s count=%s device=%r'
              % (res['ztype'], res['ztop'], res['zbottom'], res['zstep'],
                 res['zcount'], res['zdevice']))

    print('=== channels ===')
    for i in range(MAX_CHANNELS):
        body = '''
char name[256];
char oc[256];
char before[256];
char after[256];
int color, aftype, afarg1, afarg2;
strcpy(&name, "SENTINEL");
ND_GetLambdaChannel(%d, &name, &oc, &color, &before, &after, &aftype, &afarg1, &afarg2);
Int_SetKeyString("%s","q","name",name);
Int_SetKeyString("%s","q","oc",oc);
''' % (i, IN, IN)
        res = ask('channel %d' % i, body, ['name', 'oc'])
        if not res:
            break
        if res['name'] == 'SENTINEL':
            print('  channel %d: (buffer untouched -> out of range, stopping)' % i)
            break
        print('  channel %d: name=%r oc=%r' % (i, res['name'], res['oc']))
        if res['name'] == '':
            print('  (empty name — ambiguous; stopping to be safe)')
            break


if __name__ == '__main__':
    main()
