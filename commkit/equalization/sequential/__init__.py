"""Sequential adaptive equalizers.

:func:`lms`, :func:`rls`, :func:`cma` and :func:`rde`.  ``_dd.py`` holds the decision-directed ones (``lms``, ``rls``), ``_blind.py``
the blind ones (``cma``, ``rde``) and ``_setup.py`` their shared front end
and result assembly; the kernels are in
``commkit/equalization/_kernels_numba.py``.
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
