"""
verify nd2_helpers.read_channel: axis-order-independent channel
selection, multi-channel selection, dimension validation (T/P rejected,
Z only with a projection), and the axis logic on synthetic arrays
(no-C files and odd axis orders that no nd2 file here can provide)

run: python autofrap/autofrap_bitsnpieces/test_read_channel.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import nd2
from autofrap.io import nd2 as nd2_helpers
SURVEY = os.path.join(ROOT, 'test_acquisitions', 'autofrap_out',
                      '20260824_125948_c01_survey.nd2')   # (C, Y, X), 3 ch
FRAP = os.path.join(ROOT, 'test_acquisitions', 'autofrap_out',
                    '20260824_125948_c01_frap.nd2')       # (T, Y, X), no C
ZSTACK = os.path.join(ROOT, '01.nd2')                     # (Z, C, Y, X)

failures = 0


def check(name, ok):
    global failures
    if not ok:
        failures += 1
    print(f"{'ok  ' if ok else 'FAIL'} {name}")


def expect_error(name, fun, exc, contains):
    try:
        fun()
        check(name, False)
        print('       (no error raised)')
    except exc as e:
        ok = all(c in str(e) for c in contains)
        check(name, ok)
        if not ok:
            print(f'       (message: {e})')
    except Exception as e:
        check(name, False)
        print(f'       (wrong exception: {e!r})')


def expect_value_error(name, fun, contains):
    expect_error(name, fun, ValueError, contains)


# 1. survey file (C, Y, X): same results as the old f.asarray()[channel]
print(f'-- {os.path.basename(SURVEY)} (C, Y, X)')
with nd2.ND2File(SURVEY) as f:
    full = f.asarray()
for ch in range(full.shape[0]):
    img = nd2_helpers.read_channel(SURVEY, ch)
    check(f'channel {ch}: shape {img.shape}, dtype {img.dtype}',
          img.shape == full.shape[1:] and img.dtype == full.dtype
          and np.array_equal(img, full[ch]))
multi = nd2_helpers.read_channel(SURVEY, (0, 1))
check('channels (0, 1) -> (2, y, x)', multi.shape == (2, *full.shape[1:])
      and np.array_equal(multi, full[[0, 1]]))
reordered = nd2_helpers.read_channel(SURVEY, (1, 0))
check('channels (1, 0) keep the given order',
      np.array_equal(reordered, full[[1, 0]]))
expect_error('channel 5 out of range', lambda: nd2_helpers.read_channel(SURVEY, 5),
             (ValueError, IndexError), [])

# equivalence with the old implementation across all copied survey files
survey_dir = os.path.dirname(SURVEY)
n_eq = 0
for fn in sorted(os.listdir(survey_dir)):
    if not fn.endswith('_survey.nd2'):
        continue
    path = os.path.join(survey_dir, fn)
    with nd2.ND2File(path) as f:
        full = f.asarray()
    for ch in range(full.shape[0]):
        if not np.array_equal(nd2_helpers.read_channel(path, ch), full[ch]):
            check(f'old-behavior equivalence {fn} ch{ch}', False)
        n_eq += 1
print(f'       old-behavior equivalence: {n_eq} channel(s) checked')

# 2. FRAP timeseries (T, Y, X): unsupported dimension -> error
print(f'-- {os.path.basename(FRAP)} (T, Y, X)')
expect_value_error('timeseries (T) rejected',
                   lambda: nd2_helpers.read_channel(FRAP, 0), ['T'])

# 3. z-stack (Z, C, Y, X): error by default, max projection on request
print(f'-- {os.path.basename(ZSTACK)} (Z, C, Y, X)')
with nd2.ND2File(ZSTACK) as f:
    full = f.asarray()
    proj = full.max(axis=0)
expect_value_error('z-stack rejected by default',
                   lambda: nd2_helpers.read_channel(ZSTACK, 0), ['Z', 'z_projection'])
expect_value_error("z_projection='mean' unknown",
                   lambda: nd2_helpers.read_channel(ZSTACK, 0, z_projection='mean'), ['z_projection'])
img = nd2_helpers.read_channel(ZSTACK, 0, z_projection='max')
check('z_projection max, ch 0 -> (y, x)', img.shape == proj.shape[1:]
      and np.array_equal(img, proj[0]))
multi = nd2_helpers.read_channel(ZSTACK, (1, 0), z_projection='max')
check('z_projection max, ch (1, 0) -> (2, y, x) in order',
      multi.shape == (2, *proj.shape[1:]) and np.array_equal(multi, proj[[1, 0]]))

# 4. axis logic on synthetic arrays (no nd2 writer in nd2 0.11.3):
#    no-C files and axis orders that no file here provides
print('-- synthetic axis logic (_extract)')
rng = np.random.default_rng(0)
y, x = 6, 7
# note: np.transpose(a, perm) has result shape j == a.shape[perm[j]]
base_cyx = rng.integers(0, 400, (3, y, x), dtype=np.uint16)  # (C, Y, X)
a = base_cyx[0]
b = np.transpose(base_cyx, (1, 0, 2))                        # (Y, C, X)
z = rng.integers(0, 400, (4, y, x), dtype=np.uint16)         # (Z, Y, X)
base_czyx = rng.integers(0, 400, (3, 4, y, x), dtype=np.uint16)  # (C, Z, Y, X)
zc = np.transpose(base_czyx, (1, 0, 2, 3))                   # (Z, C, Y, X)

check('(Y, X), ch 0 -> unchanged',
      np.array_equal(nd2_helpers._extract(a, ['Y', 'X'], [0], None), a))
check('(Z, Y, X), ch 0, max -> (y, x)',
      np.array_equal(nd2_helpers._extract(z, ['Z', 'Y', 'X'], [0], 'max'), z.max(axis=0)))
check('(C, Y, X), ch 0 -> (y, x)',
      np.array_equal(nd2_helpers._extract(base_cyx, ['C', 'Y', 'X'], [0], None),
                     base_cyx[0]))
check('(C, Y, X), ch (0, 1) -> (2, y, x)',
      np.array_equal(nd2_helpers._extract(base_cyx, ['C', 'Y', 'X'], [0, 1], None),
                     np.stack([base_cyx[0], base_cyx[1]])))
check('(Z, C, Y, X), ch 1, max -> (y, x)',
      np.array_equal(nd2_helpers._extract(zc, ['Z', 'C', 'Y', 'X'], [1], 'max'),
                     zc[:, 1, :, :].max(axis=0)))
check('(C, Z, Y, X), ch (0, 1), max -> (2, y, x)',
      np.array_equal(nd2_helpers._extract(base_czyx, ['C', 'Z', 'Y', 'X'],
                                          [0, 1], 'max'),
                     np.stack([base_czyx[0].max(axis=0),
                               base_czyx[1].max(axis=0)])))
check('(Y, C, X), ch 0 -> (y, x) (odd order)',
      np.array_equal(nd2_helpers._extract(b, ['Y', 'C', 'X'], [0], None),
                     base_cyx[0]))

print()
print(f'{failures} failure(s)')
sys.exit(1 if failures else 0)
