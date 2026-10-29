"""
Live probe: large-image (stitched) scan via the Stg_LargeImageScanArea
macro — anchor question (20261027). OUTCOME: the macro route is broken
on this unit (4x4 tiles for a 2x2-FOV request, very slow per tile, no
document/file left behind; see docs/SESSION_HISTORY.md 20261027). The
anchor question was settled via the ND Acquisition Large Image route
instead: stitched ND large images anchor at the image center, same as
single images (docs/NIS_REFERENCE.md § Pixel <-> stage transform).
Kept as the record of the macro-route attempt.

What it does:
  1. reads the current stage position and FOV (from camera resolution);
  2. runs a scan (no overlap, no registration, single z) over a
     stage-coordinate rectangle of 2x2 FOVs centered on the current
     position (NIS chooses the actual tiling — observed 4x4 on the
     first live run, 20261029), leaving the stitched document open;
  3. saves the document explicitly (on this NIS version the scan does
     NOT write the filename param to disk itself) and closes it;
  4. reads back the stage position stored in the nd2 metadata
     (autofrap.io.nd2.stage_position) and prints it next to the
     reference candidates (scan-area center, first-tile center /
     corner).

Run directly at the microscope (NIS open):
    python autofrap/autofrap_bitsnpieces/probe_large_image_anchor.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from autofrap.microscope import nis as nis_util

NIS_EXE = r'C:\Program Files\NIS-Elements\nis_ar.exe'
SAVE_PATH = os.path.join(tempfile.gettempdir(), 'large_image_probe.nd2')


def main():
    x, y = nis_util.get_position(NIS_EXE)[:2]
    fov_x, fov_y = nis_util.get_fov_from_res(nis_util.get_resolution(NIS_EXE))
    print('current stage position : x=%.1f  y=%.1f' % (x, y))
    print('FOV                   : %.1f x %.1f um' % (fov_x, fov_y))

    # 2x2 tiles, no overlap: area of 2x2 FOVs centered on the current
    # position. The rectangle is symmetric (+/- 1 FOV per axis), so the
    # scan area is the same whichever way NIS interprets
    # left/right/top/bottom.
    left, right = x - fov_x, x + fov_x
    top, bottom = y - fov_y, y + fov_y
    print('scan area (stage um)  : left=%.1f right=%.1f top=%.1f bottom=%.1f'
          % (left, right, top, bottom))
    print()

    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
    print('running scan -> %s (blocks a while) ...' % SAVE_PATH)
    # close=False: on this NIS version the scan leaves the stitched image
    # as an unsaved document — the default close=True would silently
    # discard it (CloseCurrentDocument(2), no dialog; live 20261029)
    nis_util.do_large_image_scan(NIS_EXE, SAVE_PATH,
                                 left, right, top, bottom,
                                 overlap=0, registration=False,
                                 z_count=1, close=False, timeout=600)
    print('scan finished; document is now open in NIS (unsaved).')

    docs = nis_util.get_opened_documents(NIS_EXE)
    print('open documents in NIS :', docs)

    if not os.path.exists(SAVE_PATH):
        print('no file yet — activating the stitched document, then saving ...')
        # the scan leaves the stitched image as an in-memory document
        # (titled with the filename param on this NIS version); the
        # current document may still be something else (e.g. the frozen
        # live view) — ImageSaveAs would then silently write nothing
        # (known gotcha), so activate by name first
        matched = nis_util.activate_opened_document(NIS_EXE, SAVE_PATH)
        print('activated:', matched)
        nis_util.save_current_document(NIS_EXE, SAVE_PATH)
        if os.path.exists(SAVE_PATH):
            print('saved to', SAVE_PATH)
            nis_util.close_current_document(NIS_EXE, 'discard')
            print('document closed.')
        else:
            print('!! save produced no file — leaving the document open,')
            print('!! check the NIS GUI (what is the active document?)')

    from autofrap.io.nd2 import stage_position
    import nd2
    with nd2.ND2File(SAVE_PATH) as f:
        dims = f.sizes
    stored = stage_position(SAVE_PATH)
    print()
    print('stitched file sizes   :', dims)
    print('stored nd2 position   : x=%.1f  y=%.1f  z=%.1f' % stored)
    print()
    print('reference candidates (stage um):')
    print('  scan-area center    : x=%.1f  y=%.1f' % (x, y))
    print('  first-tile center   : x=%.1f  y=%.1f (min-x/min-y corner)'
          % (left + fov_x / 2, top + fov_y / 2))
    print('  (first-tile corner rule would store position = first-tile center')
    print('   -/+ FOV/2 offsets; compare the stored value against these.)')


if __name__ == '__main__':
    main()
