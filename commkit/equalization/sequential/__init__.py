"""
Public sequential adaptive equalizers: lms, rls, cma, rde.

Split into ``_dd.py`` (decision-directed: ``lms``, ``rls``) and ``_blind.py``
(blind: ``cma``, ``rde``), mirroring the package-level ``sequential.py``
(DD) vs. ``blind.py`` (blind) split.  The public import surface is
unchanged: ``from commkit.equalization import lms, rls, cma, rde`` and
``from commkit.equalization.sequential import ...`` continue to work.
"""

from __future__ import annotations

from ._blind import cma, rde
from ._dd import _check_rls_divergence, lms, rls

__all__ = [
    "_check_rls_divergence",
    "cma",
    "lms",
    "rde",
    "rls",
]
