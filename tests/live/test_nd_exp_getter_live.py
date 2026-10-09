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

Skipped automatically off the microscope workstation.
"""
import os
import unittest

from autofrap.microscope import nis as nis_util

NIS = r'C:\Program Files\NIS-Elements\nis_ar.exe'
LIVE = os.name == 'nt' and os.path.exists(NIS)
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


@unittest.skipUnless(LIVE, 'requires the microscope workstation (NIS-Elements)')
class TestNdExpGetterLive(unittest.TestCase):

    def test_nd_acquisition_dialog_probe(self):
        """every dialog query macro runs and returns its keys"""
        for tab in TAB_CANDIDATES:
            res = ask('tab %r' % tab,
                      'Int_SetKeyValue("%s","q","on",ND_IsAcqTabChecked("%s"));'
                      % (IN, tab), ['on'])
            self.assertTrue(res, f'tab {tab!r} query failed')
            print(f'  tab {tab!r}: active={res["on"]}')

        res = ask('phase count',
                  'Int_SetKeyValue("%s","q","phases",ND_GetTimeLapsePhaseCount());'
                  % IN, ['phases'])
        self.assertTrue(res, 'phase count query failed')
        print(f'  time phases: {res["phases"]}')
        phases = int(res['phases'])
        if phases:
            body = '''
double interval, duration;
int loopcnt;
ND_GetTimePhaseSchedule(0, &interval, &duration, &loopcnt);
Int_SetKeyValue("%s","q","interval",interval);
Int_SetKeyValue("%s","q","duration",duration);
Int_SetKeyValue("%s","q","loopcnt",loopcnt);
''' % (IN, IN, IN)
            res = ask('phase 0 schedule', body,
                      ['interval', 'duration', 'loopcnt'])
            self.assertTrue(res, 'phase 0 schedule query failed')
            print('  phase 0: loopcnt=%s interval=%s ms duration=%s ms'
                  % (res['loopcnt'], res['interval'], res['duration']))

        res = ask('position count',
                  'Int_SetKeyValue("%s","q","count",ND_MP_GetCount());' % IN,
                  ['count'])
        self.assertTrue(res, 'position count query failed')
        print(f'  positions: {res["count"]}')

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
        res = ask('z series', body,
                  ['ztype', 'ztop', 'zbottom', 'zstep', 'zcount', 'zdevice'])
        self.assertTrue(res, 'z series query failed')
        print('  type=%s top=%s bottom=%s step=%s count=%s device=%r'
              % (res['ztype'], res['ztop'], res['zbottom'], res['zstep'],
                 res['zcount'], res['zdevice']))

        n_channels = 0
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
                print(f'  channel {i}: (buffer untouched -> out of range, '
                      'stopping)')
                break
            print(f'  channel {i}: name={res["name"]!r} oc={res["oc"]!r}')
            if res['name'] == '':
                print('  (empty name — ambiguous; stopping to be safe)')
                break
            n_channels += 1
        self.assertGreater(n_channels, 0, 'no channels found')


if __name__ == '__main__':
    unittest.main()
