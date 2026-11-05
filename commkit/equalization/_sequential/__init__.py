"""Sequential adaptive equalizers (private).

:func:`lms`, :func:`rls`, :func:`cma` and :func:`rde`, re-exported by
:mod:`commkit.equalization`.  ``_dd.py`` holds the trained and
decision-directed ones (``lms``, ``rls``), ``_blind.py`` the blind ones
(``cma``, ``rde``) and ``_setup.py`` their shared front end and result
assembly; the kernels are in ``commkit/equalization/_numba_loops.py``, each
fetched by a ``_get_numba_*`` getter in the module that calls it.
"""

from __future__ import annotations

from ._blind import cma, rde
from ._dd import lms, rls

__all__ = [
    "cma",
    "lms",
    "rde",
    "rls",
]
