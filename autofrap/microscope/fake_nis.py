"""
Offline stand-in for the NIS macro layer for dry-running autofrap().

autofrap() / autofrap_grid() never touch the NIS executable directly —
every interaction goes through attribute lookups on the nis_util module
(nis_util.run_current_nd_experiment(...) and friends).  Installing
FakeNIS replaces exactly those attributes with in-process fakes, so the
whole pipeline (grid loop, cross-cycle matching, error handling, cleanup)
can be driven offline:

    from autofrap import autofrap
    from autofrap.fake_nis import FakeNIS

    with FakeNIS(['a.nd2', 'b.nd2', 'c.nd2']):
        autofrap.autofrap_multiposition('fake', 'test_acquisitions/dry_run',
                               positions=None, max_cycles=3,
                               detection_fun=my_detection_fun)

`nis_exe` is ignored — any placeholder string works.

"Acquisition" is simulated by copying source nd2 files into the output
directory.  autofrap_grid always prefixes its files with 'fov<NN>'
(file_prefix=f'fov{i:02d}'), so the fake derives the current FOV from
the output file name:

  - all cycles of one FOV copy the same source file
  - the next FOV uses the next source (wrapping if there are more FOVs
    than sources)
  - standalone autofrap() (timestamp prefix, no fov tag) uses sources[0]

The FRAP file is created by the save_current_document fake: frap_out=
'copy' copies the last survey source (a plausible nd2), 'touch' writes
an empty file.  The pipeline's trust-but-verify checks are
filesystem-based (os.path.isfile on the outputs, normcase comparisons of
document paths), so the fakes pass them naturally: the copied surveys
really are on disk, and a small document state machine (open list +
current document, 'Frozen' = the always-open live view) mirrors NIS's
open / activate / save / close semantics.

The failure flags (fail_survey, fail_save, fail_move, ...) mirror the
knobs of the inline FakeNIS in test_autofrap_errors.py for offline
error-path tests.
"""
import os
import re
import shutil

from autofrap.microscope import nis as nis_util  # patch target

# the nis_util surface used by autofrap() / autofrap_grid()
# (check: grep -o "nis_util\.[a-z_]*" autofrap/pipeline.py)
PATCHED_FUNCTIONS = (
    'get_nd_acq_tabs', 'get_position', 'get_resolution', 'set_position',
    'run_current_nd_experiment', 'run_stimulation_experiment',
    'save_current_document', 'get_current_document', 'open_image',
    'close_current_document', 'activate_opened_document',
    'add_polygon_roi', 'set_roi_type', 'delete_roi',
    'set_optical_configuration',
    'batch_run_macro',
    'delete_all_rois_in_current_document',
    'close_all_docs',
    'checkpoint',
)

_FOV_RE = re.compile(r'fov(\d+)_')


class FakeNIS:
    """
    fake nis_util surface for offline pipeline runs (context manager)

    Parameters
    ----------
    sources: list of str
        nd2 files used as the "acquired" surveys — one per FOV, cycled
        (see the module docstring for the FOV mapping); each is copied
        to the survey output path of its FOV
    position: (x, y, z0)
        value returned by get_position (the grid centers on the first
        two)
    resolution: (xres, yres, pixel_size, magnification)
        value returned by get_resolution; the grid step is
        pixel_size * xres / magnification per FOV
    frap_out: 'copy' or 'touch'
        how the FRAP file is created on save_current_document
    fail_survey, fail_save, fail_move, roi_id, open_broken,
    abort_add_roi:
        failure knobs for error-path tests (fail_move fails only the
        first stage move, mirroring test_autofrap_errors.py)

    Attributes
    ----------
    calls: list of (name, args)
        every fake call, in order
    open_docs, current:
        the fake document state machine
    """

    def __init__(self, sources, position=(0.0, 0.0, 0.0),
                 resolution=(1024, 1024, 13.0, 100.0), frap_out='copy',
                 fail_survey=False, fail_save=False, fail_move=False,
                 roi_id=1, open_broken=False, abort_add_roi=False):
        sources = [os.path.abspath(s) for s in sources]
        if not sources:
            raise ValueError('sources: give at least one nd2 file')
        for s in sources:
            if not os.path.isfile(s):
                raise FileNotFoundError(f'source file does not exist: {s}')
        if frap_out not in ('copy', 'touch'):
            raise ValueError(f"frap_out must be 'copy' or 'touch', "
                             f'got {frap_out!r}')
        self.sources = sources
        self.position = tuple(position)
        self.resolution = tuple(resolution)
        self.frap_out = frap_out
        self.fail_survey = fail_survey
        self.fail_save = fail_save
        self.fail_move = fail_move
        self.roi_id = roi_id
        self.open_broken = open_broken
        self.abort_add_roi = abort_add_roi
        # state
        self.calls = []
        self.open_docs = []          # open documents (paths or titles)
        self.current = 'Frozen'      # the always-open live view
        self._next_roi = 0
        self._last_source = None     # source of the last survey acquisition
        self._orig = {}

    # ------------------------------------------------------------ #
    # context manager: swap the fakes into and out of nis_util     #
    # ------------------------------------------------------------ #
    def __enter__(self):
        self._orig = {name: getattr(nis_util, name)
                      for name in PATCHED_FUNCTIONS}
        for name in PATCHED_FUNCTIONS:
            setattr(nis_util, name, getattr(self, '_' + name))
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(nis_util, name, orig)
        return False

    # ------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------ #
    def _call(self, name, *args):
        self.calls.append((name, args))

    def calls_of(self, name):
        """args of all recorded calls to `name`"""
        return [args for n, args in self.calls if n == name]

    # ------------------------------ setup ------------------------------ #
    def _get_nd_acq_tabs(self, nis):
        self._call('get_nd_acq_tabs')
        return {'Time': False, 'XY': False, 'Z': False,
                'Lambda': False, 'Large Image': False}

    def _get_position(self, nis):
        self._call('get_position')
        return self.position

    def _get_resolution(self, nis):
        self._call('get_resolution')
        return self.resolution

    def _set_position(self, nis, pos_xy=None, pos_z=None, pos_piezo=None,
                      relative_xy=False, relative_z=False,
                      relative_piezo=False):
        self._call('set_position', pos_xy)
        if self.fail_move:
            self.fail_move = False  # fail only the first move
            raise KeyError('pos')

    # ----------------------------- acquisition ------------------------- #
    def _source_for(self, outfile):
        """source file for the acquisition written to `outfile`"""
        m = _FOV_RE.match(os.path.basename(outfile))
        if m:
            # autofrap_grid names the files fov<NN>_cycle<NN>_...
            return self.sources[(int(m.group(1)) - 1) % len(self.sources)]
        return self.sources[0]

    def _run_current_nd_experiment(self, nis, outfile=None,
                                   open_after=True, progress_bar=True):
        self._call('run_current_nd_experiment', outfile)
        if self.fail_survey or outfile is None:
            return  # NIS did not save: the pipeline's isfile check fails
        self._last_source = self._source_for(outfile)
        shutil.copy(self._last_source, outfile)
        if open_after:
            self._open(outfile)

    def _run_stimulation_experiment(self, nis):
        self._call('run_stimulation_experiment')
        # the result stays open as the current (unsaved) document
        self._open('ND Acquisition')

    # ---------------------------- document state ----------------------- #
    def _open(self, name):
        if name not in self.open_docs:
            self.open_docs.append(name)
        self.current = name

    def _get_current_document(self, nis):
        self._call('get_current_document')
        return self.current

    def _open_image(self, nis, image_path):
        self._call('open_image', image_path)
        if self.open_broken:
            return  # stay on the current document (pipeline check fails)
        self._open(image_path)

    def _close_current_document(self, nis, save='discard'):
        self._call('close_current_document', save)
        if self.current in self.open_docs:
            self.open_docs.remove(self.current)
        self.current = 'Frozen'

    def _activate_opened_document(self, nis, name):
        self._call('activate_opened_document', name)
        # same matching semantics as the real wrapper (nis_util)
        doc = nis_util._match_opened_document(name, self.open_docs)
        if doc is not None:
            self.current = doc
            return doc
        base = os.path.basename(os.path.normcase(name))
        hits = [d for d in self.open_docs
                if os.path.basename(os.path.normcase(d)) == base]
        if hits:
            raise RuntimeError(f'ambiguous match for {name!r}: {hits}')
        raise FileNotFoundError(
            f'{name!r} is not among the open documents: {self.open_docs}')

    def _save_current_document(self, nis, outfile):
        self._call('save_current_document', outfile)
        if self.fail_save:
            return  # ImageSaveAs wrote nothing
        if self.frap_out == 'copy':
            shutil.copy(self._last_source or self.sources[0], outfile)
        else:
            open(outfile, 'wb').close()
        # ImageSaveAs rebinds the current document to the file
        if self.current in self.open_docs:
            self.open_docs[self.open_docs.index(self.current)] = outfile
        self.current = outfile

    # ------------------------------- ROIs ------------------------------ #
    def _add_polygon_roi(self, nis, points, color='green'):
        self._call('add_polygon_roi', len(points))
        if self.abort_add_roi:
            raise KeyError('id')  # simulate an empty ini read-back
        self._next_roi += 1
        return self.roi_id

    def _set_roi_type(self, nis, roi_id, roi_type):
        self._call('set_roi_type', roi_id, roi_type)

    def _delete_roi(self, nis, roi_id):
        self._call('delete_roi', roi_id)

    # ---------------------------- optical conf ------------------------- #
    def _set_optical_configuration(self, nis, oc_name):
        self._call('set_optical_configuration', oc_name)

    def _delete_all_rois_in_current_document(self, nis):
        self._call('delete_all_rois_in_current_document')
        self._next_roi = 0

    def _close_all_docs(self, nis):
        self._call('close_all_docs')
        self.open_docs.clear()
        self.current = 'Frozen'

    def _checkpoint(self, nis, key='ok', value=1):
        self._call('checkpoint', key, value)

    # ---------------------------- batch runner ------------------------- #
    def _batch_run_macro(self, nis, calls, timeout=20):
        self._call('batch_run_macro', len(calls))
        out = {}
        for i, (op, params) in enumerate(calls):
            sec = f"{op.name}_{i}"
            # reads – return stored fake state
            if op.name == 'position':
                out[sec] = self.position
                continue
            if op.name == 'resolution':
                out[sec] = self.resolution
                continue
            if op.name == 'nd_acq_tabs':
                out[sec] = {'Time': False, 'XY': False, 'Z': False,
                            'Lambda': False, 'Large Image': False}
                continue
            # setters – delegate to existing fakes for call logging / failure
            if op.name == 'set_position':
                pos_xy = None
                if 'x' in params and 'y' in params:
                    pos_xy = (params['x'], params['y'])
                self._set_position(
                    nis,
                    pos_xy=pos_xy,
                    pos_z=params.get('z'),
                    pos_piezo=params.get('piezo'),
                    relative_xy=params.get('relative_xy', False),
                    relative_z=params.get('relative_z', False),
                    relative_piezo=params.get('relative_piezo', False),
                )
                out[sec] = None
                continue
            if op.name == 'set_optical_configuration':
                self._set_optical_configuration(nis, params.get('name'))
                out[sec] = None
                continue
            if op.name == 'add_polygon_roi':
                # delegate to existing fake add_polygon_roi
                # params contains points and color
                points = params.get('points', [])
                color = params.get('color', 'green')
                # simulate call
                self._call('add_polygon_roi', len(points))
                if self.abort_add_roi:
                    raise KeyError('id')
                self._next_roi += 1
                # return the ini format expected by MacroOp.parse
                out[sec] = {'id': self.roi_id}
                continue
            if op.name == 'delete_all_rois_in_current_document':
                # NOP for fake – ROIs are session-global, just clear counter
                self._next_roi = 0
                out[sec] = None
                continue
            if op.name == 'close_all_docs':
                self.open_docs.clear()
                self.current = 'Frozen'
                out[sec] = None
                continue
            if op.name == 'checkpoint':
                out[sec] = True
                continue
            # acquisition ops – simulate side effects
            if op.name == 'run_current_nd_experiment':
                outfile = params.get('outfile')
                open_after = params.get('open_after', True)
                self._run_current_nd_experiment(
                    nis, outfile=outfile, open_after=open_after, progress_bar=True
                )
                out[sec] = None
                continue
            if op.name == 'run_stimulation_experiment':
                self._run_stimulation_experiment(nis)
                out[sec] = None
                continue
            # unknown / future ops – NOP
            out[sec] = None
        return out
