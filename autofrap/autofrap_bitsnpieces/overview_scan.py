"""
Part 1 of the grid survey pipeline:
  grid of positions -> move stage -> capture with current NIS settings

(Detection + per-ROI FRAP/photostimulation come later.)
"""
import os
import sys
import time

# repo root (for nis_util) — this script lives two levels down in autofrap/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nis_util

NIS_EXE = r'C:\Program Files\NIS-Elements\nis_ar.exe'
OUT_DIR = r'C:\Users\David\Desktop\nis-automation\overview'


def overview_scan(nis_exe, out_dir, nx=2, ny=2, spacing=0.8,
                  settle_s=1.0, return_to_start=True):
    """
    Scan an nx x ny grid of positions (centered on the current stage
    position) and capture one image per position using the ND
    acquisition currently configured in the NIS GUI.

    Parameters
    ----------
    spacing: float
        distance between neighbors in units of FOV
        (1 = touching, <1 = overlap, >1 = gap)
    settle_s: float
        extra settling time [s] after each stage move
    return_to_start: bool
        move back to the starting position after the scan

    Returns
    -------
    results: list of (index, x, y, outfile, saved) tuples
    """
    os.makedirs(out_dir, exist_ok=True)

    start_xy = nis_util.get_position(nis_exe)[:2]
    positions = nis_util.grid_positions(nis_exe, nx=nx, ny=ny, spacing=spacing)
    stamp = time.strftime('%Y%m%d_%H%M%S')

    print(f'grid: {nx}x{ny}, spacing={spacing} FOV, start=({start_xy[0]:+.2f}, {start_xy[1]:+.2f})')
    results = []
    for i, (x, y) in enumerate(positions, 1):
        outfile = os.path.join(out_dir, f'{stamp}_p{i:02d}_{x:+07.1f}_{y:+07.1f}.nd2')
        print(f'[{i}/{len(positions)}] ({x:+.1f}, {y:+.1f}) um', flush=True)

        nis_util.set_position(nis_exe, pos_xy=(x, y))
        time.sleep(settle_s)

        t0 = time.time()
        nis_util.run_current_nd_experiment(nis_exe, outfile=outfile,
                                           progress_bar=False)
        saved = os.path.exists(outfile)
        print(f'    -> {os.path.basename(outfile)} ({time.time()-t0:.1f}s, saved={saved})', flush=True)
        results.append((i, x, y, outfile, saved))

    if return_to_start:
        nis_util.set_position(nis_exe, pos_xy=start_xy)
        time.sleep(settle_s)
        print(f'moved back to start ({start_xy[0]:+.2f}, {start_xy[1]:+.2f})')

    return results


if __name__ == '__main__':
    results = overview_scan(NIS_EXE, OUT_DIR, nx=2, ny=2, spacing=0.8)

    print('\nSummary:')
    for i, x, y, f, ok in results:
        print(f'  {i}: ({x:+.1f}, {y:+.1f}) -> {os.path.basename(f)} {"OK" if ok else "MISSING!"}')
    print(f'{sum(1 for r in results if r[4])}/{len(results)} images saved to {OUT_DIR}')
