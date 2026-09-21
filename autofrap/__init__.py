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
    autofrap_loop_outer,
    grid_positions,
    next_stimulatable_cell,
)

# Backwards‑compatible alias
# Old name 'autofrap_multiposition' now points to the loop outer function
# TODO: remove?
autofrap_multiposition = autofrap_loop_outer
autofrap_grid = autofrap_multiposition
