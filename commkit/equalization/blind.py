"""Blind block equalizers.

:func:`block_cma` and :func:`block_rde` are the frequency-domain block
versions of ``cma`` and ``rde``.
"""

from __future__ import annotations

from typing import Any

from ..backend import ArrayType
from ..core._signal_adapter import adapt_signal, require_integer_sps
from ..core.signal import Signal
from ._block import _block_fdaf_blind
from ._common import _godard_radius, _rde_ring_radii
from .pilots import EqualizerPilots
from .result import EqualizerResult, EqualizerState, _attach_equalized_signal
from .sequential._blind import _check_pilots

__all__ = ["block_cma", "block_rde"]

# -----------------------------------------------------------------------------
# BLOCK BLIND EQUALIZERS (Signal-aware)
# -----------------------------------------------------------------------------


def _block_blind(
    kind: str,
    samples: ArrayType | Signal,
    *,
    num_taps: int,
    sps: int | None,
    step_size: float | None,
    block_size: int,
    constellation: Any,
    initial_taps: ArrayType | None,
    pilots: EqualizerPilots | None,
    pad_mode: str,
    state: EqualizerState | None,
    cuda_graph: bool,
) -> EqualizerResult:
    """The body of :func:`block_cma` and :func:`block_rde` (``kind``)."""
    name = f"block_{kind}()"
    signal_adapter = adapt_signal(samples, function_name=name)
    samples = signal_adapter.array
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), name)
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    pilot_ref, pilot_mask, pilot_gain_db = _check_pilots(
        signal_adapter.array, sps, pilots, name, state
    )
    if kind == "cma":
        r2, radii_np = _godard_radius(constellation), None
    else:
        r2, radii_np = 1.0, _rde_ring_radii(constellation)
    label = f"Block-{kind.upper()}"
    result = _block_fdaf_blind(
        kind,
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        r2=r2,
        radii_np=radii_np,
        initial_taps=initial_taps,
        state=state,
        pad_mode=pad_mode,
        pilot_ref=pilot_ref,
        pilot_mask=pilot_mask,
        pilot_gain_db=pilot_gain_db,
        cuda_graph=cuda_graph,
        name=label if pilot_ref is None else f"{label}(PA)",
    )
    return _attach_equalized_signal(result, signal_adapter.signal, state)


def block_cma(
    samples: ArrayType | Signal,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 2e-4,
    block_size: int = 256,
    constellation: Any = None,
    initial_taps: ArrayType | None = None,
    pilots: EqualizerPilots | None = None,
    pad_mode: str = "zeros",
    state: EqualizerState | None = None,
    cuda_graph: bool = True,
) -> EqualizerResult:
    """Equalize blindly with the frequency-domain constant-modulus algorithm.

    The phase-blind sibling of :func:`block_lms`: the same overlap-save
    frequency-domain adaptive filter (FDAF), driven by the Godard
    constant-modulus error instead of training symbols or decisions.  For
    blind acquisition at the high-throughput operating point (slow or static
    channels, GPU input); :func:`cma` adapts every symbol instead, for the
    fastest channel dynamics, and is usually faster on the CPU.

    Like :func:`cma`, it recovers the channel up to a phase ambiguity: run a
    carrier-phase recovery stage afterwards (there is no ``cpr``, see the
    Notes of :func:`cma`).  ``pilots`` switch the pilot positions to the LMS
    residual error, which resolves the ambiguity there while the data
    positions stay blind.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` supplies
        ``sps`` and ``constellation``.
    num_taps : int, default 21
        Taps per FIR filter, in samples.
    sps : int, optional
        Samples per symbol, an integer.  Taken from the Signal; required for
        array input.  A value that disagrees with the Signal raises.
    step_size : float, default 2e-4
        Step size on the scale of :func:`cma`.  The weights are frozen
        across each block, so the stability ceiling is about ``block_size``
        times lower than per symbol; reduce it if the run raises divergence.
    block_size : int, default 256
        Output symbols per weight update.
    constellation : Constellation, optional
        Sets the Godard radius, as for :func:`cma`.  Defaults to the
        Signal's; without one the target is the unit circle.
    initial_taps : array_like, optional
        Initial tap weights, as for :func:`cma`; only for a cold start.
    pilots : EqualizerPilots, optional
        Known pilots, as for :func:`cma`.
    pad_mode : {'zeros', 'edge', 'reflect'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous call with the same
        configuration; see :func:`lms`.
    cuda_graph : bool, default True
        On the GPU, capture the per-block body once into a CUDA graph and
        replay it (one launch per block, a large gain at small
        ``block_size``).  Ignored on the CPU and for the pilot-aided path.

    Returns
    -------
    EqualizerResult
        ``y_hat``, ``weights``, ``error`` and ``state`` on the input's
        device, and for Signal input ``signal``, the 1-SPS output Signal.
    """
    return _block_blind(
        "cma",
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        constellation=constellation,
        initial_taps=initial_taps,
        pilots=pilots,
        pad_mode=pad_mode,
        state=state,
        cuda_graph=cuda_graph,
    )


def block_rde(
    samples: ArrayType | Signal,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float | None = None,
    block_size: int = 256,
    constellation: Any = None,
    initial_taps: ArrayType | None = None,
    pilots: EqualizerPilots | None = None,
    pad_mode: str = "zeros",
    state: EqualizerState | None = None,
    cuda_graph: bool = True,
) -> EqualizerResult:
    """Equalize blindly with the frequency-domain radius-directed algorithm.

    Related to :func:`block_cma` as :func:`rde` is to :func:`cma`: each
    symbol is driven towards its nearest constellation ring instead of one
    Godard circle, which converges blindly on multi-ring QAM (16-, 64-QAM)
    where CMA's single radius stalls.  The same overlap-save engine and
    operating point as :func:`block_cma`.

    Blind up to a phase ambiguity (run carrier-phase recovery afterwards);
    ``pilots`` resolve it at the pilot positions.  From a
    cold start RDE converges several times slower than CMA: pass ``state=``
    from a :func:`block_cma` call over the first part of the record to start
    from CMA-converged taps.

    Parameters
    ----------
    samples : array_like or Signal
        Input samples, ``(N,)`` or ``(C, N)``.  A :class:`Signal` supplies
        ``sps`` and ``constellation``.
    num_taps : int, default 21
        Taps per FIR filter, in samples.
    sps : int, optional
        Samples per symbol, an integer.  Taken from the Signal; required for
        array input.  A value that disagrees with the Signal raises.
    step_size : float, optional
        Step size on the scale of :func:`rde`, with the about ``block_size``
        times lower stability ceiling of frozen-block adaptation.  Defaults
        to ``min(1e-3, 0.5 / block_size)``: the per-symbol step, capped at
        half the ceiling (``step_size * block_size`` below about 1) for large
        blocks.
    block_size : int, default 256
        Output symbols per weight update.
    constellation : Constellation, optional
        Sets the ring radii, as for :func:`rde`.  Defaults to the Signal's;
        without one the single ring is the unit circle.
    initial_taps : array_like, optional
        Initial tap weights, as for :func:`rde`; only for a cold start.
    pilots : EqualizerPilots, optional
        Known pilots, as for :func:`rde`.
    pad_mode : {'zeros', 'edge', 'reflect'}, default 'zeros'
        Left padding of a cold start; see :func:`lms`.
    state : EqualizerState, optional
        Continue from ``result.state`` of a previous ``block_rde`` or
        ``block_cma`` call with the same configuration; see :func:`lms`.
    cuda_graph : bool, default True
        On the GPU, capture the per-block body once into a CUDA graph and
        replay it.  Ignored on the CPU and for the pilot-aided path.

    Returns
    -------
    EqualizerResult
        ``y_hat``, ``weights``, ``error`` and ``state`` on the input's
        device, and for Signal input ``signal``, the 1-SPS output Signal.
    """
    return _block_blind(
        "rde",
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        constellation=constellation,
        initial_taps=initial_taps,
        pilots=pilots,
        pad_mode=pad_mode,
        state=state,
        cuda_graph=cuda_graph,
    )


# -----------------------------------------------------------------------------
# PILOT REFERENCE BUILDER (array-only)
