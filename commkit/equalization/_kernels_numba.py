"""Entry points to the Numba kernels of the sequential equalizers (private).

Each getter returns a compiled kernel of ``_numba_loops``, importing that
module (and Numba) on first use, so ``import commkit`` stays light.  Users:
:func:`commkit.equalization.lms` and ``rls`` (with and without inline
carrier-phase recovery), ``cma`` and ``rde`` (blind and pilot-aided), and the
cycle-slip scan of ``block_lms``.
"""

from types import ModuleType
from typing import Any


def _loops() -> ModuleType:
    from . import _numba_loops

    return _numba_loops


def _get_numba_lms() -> Any:
    """Decision-directed LMS."""
    return _loops().lms_loop


def _get_numba_rls() -> Any:
    """Decision-directed RLS."""
    return _loops().rls_loop


def _get_numba_cpr() -> Any:
    """LMS or RLS with inline carrier recovery (PLL or BPS) and cycle-slip
    repair."""
    return _loops().cpr_loop


def _get_numba_cma() -> Any:
    """CMA, blind or pilot-aided."""
    return _loops().cma_loop


def _get_numba_rde() -> Any:
    """RDE, blind or pilot-aided."""
    return _loops().rde_loop


def _get_numba_cs_block() -> Any:
    """The cycle-slip repair of one block of the block equalizers."""
    return _loops().cs_block
