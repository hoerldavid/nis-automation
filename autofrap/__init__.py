"""
Auto-FRAP pipeline — package interface.
"""

# ------------------------------------------------------------------ #
# Public API re-exports                                                #
# ------------------------------------------------------------------ #

from autofrap.pipeline.autofrap import (  # noqa: E402,F401
    AutofrapError,
    RecoverableError,
    NonRecoverableError,
    autofrap,
    autofrap_multiposition,
    grid_positions,
    next_stimulatable_cell,
)

# Backwards‑compatible alias (the old name still works)
autofrap_grid = autofrap_multiposition
