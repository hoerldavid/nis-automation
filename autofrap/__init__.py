"""
Auto-FRAP pipeline — package interface.

Import helpers at the repo root (e.g. nis_util) by inserting the parent
directory into sys.path on first import.  This is safe: the check
prevents duplicates, and the root only holds a handful of top-level
modules that would never shadow anything inside this package.
"""
import os, sys

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _here not in sys.path:
    sys.path.insert(0, _here)

# ------------------------------------------------------------------ #
# Public API re-exports                                                #
# ------------------------------------------------------------------ #

from autofrap.pipeline import (  # noqa: E402,F401
    AutofrapError,
    RecoverableError,
    NonRecoverableError,
    autofrap,
    autofrap_grid,
    grid_positions,
    next_stimulatable_cell,
)
