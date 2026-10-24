"""Blind block equalizers and pilot references.

:func:`block_cma` and :func:`block_rde` are the frequency-domain block
versions of ``cma`` and ``rde``; :func:`build_pilot_ref` packs pilot symbols
into the dense reference of pilot-aided ``cma`` / ``rde``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._array import broadcast_channels
from ..backend import ArrayType
from ..core._signal_adapter import adapt_signal, require_integer_sps
from ..core.signal import Signal
from ._block import _block_fdaf_blind
from ._common import _godard_radius, _rde_ring_radii
from .result import EqualizerResult, EqualizerState, _attach_equalized_signal
from .sequential._blind import _check_pilots

__all__ = ["block_cma", "block_rde", "build_pilot_ref"]

# -----------------------------------------------------------------------------
# BLOCK BLIND EQUALIZERS (Signal-aware)
# -----------------------------------------------------------------------------


def block_cma(
    samples: ArrayType | Signal,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 2e-4,
    block_size: int = 256,
    constellation: Any = None,
    initial_taps: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
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
    Notes of :func:`cma`).  ``pilot_ref`` / ``pilot_mask`` switch the pilot
    positions to the LMS residual error, which resolves the ambiguity there
    while the data positions stay blind.

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
    pilot_ref, pilot_mask : array_like, optional
        Dense pilot reference and mask, from :func:`build_pilot_ref`.
    pilot_gain_db : float, default 0.0
        Pilot boost in dB, as for :func:`cma`.
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
    signal_adapter = adapt_signal(samples, function_name="block_cma()")
    samples = signal_adapter.array
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), "block_cma()")
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    pilot_ref, pilot_mask = _check_pilots(
        signal_adapter.array, sps, pilot_ref, pilot_mask, "block_cma()", state
    )

    r2 = _godard_radius(constellation)
    result = _block_fdaf_blind(
        "cma",
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        r2=r2,
        radii_np=None,
        initial_taps=initial_taps,
        state=state,
        pad_mode=pad_mode,
        pilot_ref=pilot_ref,
        pilot_mask=pilot_mask,
        pilot_gain_db=pilot_gain_db,
        cuda_graph=cuda_graph,
        name="Block-CMA" if pilot_ref is None else "Block-CMA(PA)",
    )
    return _attach_equalized_signal(result, signal_adapter.signal, state)


def block_rde(
    samples: ArrayType | Signal,
    *,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float | None = None,
    block_size: int = 256,
    constellation: Any = None,
    initial_taps: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
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
    ``pilot_ref`` / ``pilot_mask`` resolve it at the pilot positions.  From a
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
    pilot_ref, pilot_mask : array_like, optional
        Dense pilot reference and mask, from :func:`build_pilot_ref`.
    pilot_gain_db : float, default 0.0
        Pilot boost in dB, as for :func:`rde`.
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
    signal_adapter = adapt_signal(samples, function_name="block_rde()")
    samples = signal_adapter.array
    sps = require_integer_sps(signal_adapter.resolve_fact("sps", sps), "block_rde()")
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    pilot_ref, pilot_mask = _check_pilots(
        signal_adapter.array, sps, pilot_ref, pilot_mask, "block_rde()", state
    )

    radii_np = _rde_ring_radii(constellation)
    result = _block_fdaf_blind(
        "rde",
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        r2=1.0,
        radii_np=radii_np,
        initial_taps=initial_taps,
        state=state,
        pad_mode=pad_mode,
        pilot_ref=pilot_ref,
        pilot_mask=pilot_mask,
        pilot_gain_db=pilot_gain_db,
        cuda_graph=cuda_graph,
        name="Block-RDE" if pilot_ref is None else "Block-RDE(PA)",
    )
    return _attach_equalized_signal(result, signal_adapter.signal, state)


# -----------------------------------------------------------------------------
# PILOT REFERENCE BUILDER (array-only)
# -----------------------------------------------------------------------------
# build_pilot_ref packs sparse pilot symbols into a dense reference array for
# cma/rde's pilot_ref/pilot_mask - a construction helper for equalizer input
# parameters, not a transform on a Signal, so it takes plain arrays only.


def build_pilot_ref(
    pilot_symbols: np.ndarray,
    pilot_mask: np.ndarray,
    *,
    num_symbols: int,
    num_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the dense pilot reference and mask of the pilot-aided blind equalizers.

    Packs sparse pilot symbols into a dense ``(C, num_symbols)`` array for the
    ``pilot_ref`` / ``pilot_mask`` arguments of :func:`cma`, :func:`rde`,
    :func:`block_cma` and :func:`block_rde`: zeros at the data positions,
    the known symbols at the pilot positions.

    Parameters
    ----------
    pilot_symbols : array_like
        Known pilot symbols in transmission order, ``(K,)`` (broadcast to
        every channel) or ``(C, K)``: the K pilots only, no data symbols.
    pilot_mask : array_like
        Boolean ``(num_symbols,)``, ``True`` at the K pilot positions of the
        equalized body.
    num_symbols : int
        Number of output symbols (payload and pilots).
    num_channels : int
        Number of receive channels C.

    Returns
    -------
    pilot_ref : numpy.ndarray
        Host complex64 ``(C, num_symbols)`` reference: zeros at the data
        positions, the pilot symbols at the pilot positions.
    pilot_mask_u8 : numpy.ndarray
        Host uint8 ``(num_symbols,)``: 1 at the pilot positions, 0 elsewhere.

    Examples
    --------
    Build the reference from a ``SingleCarrierFrame`` and
    pass it directly to ``rde``:

    ::

        struct = frame.get_structure_map(unit="symbols", sps=1, include_preamble=False)
        pilot_ref, pilot_mask_u8 = build_pilot_ref(
            pilot_symbols=frame.pilot_symbols,
            pilot_mask=np.asarray(struct["pilots"]),
            num_symbols=n_body_symbols,
            num_channels=num_channels,
        )
        result = rde(body_samples, ..., pilot_ref=pilot_ref, pilot_mask=pilot_mask_u8)
    """
    pilot_ref_arr = np.zeros((num_channels, num_symbols), dtype=np.complex64)
    mask_uint8 = np.zeros(num_symbols, dtype=np.uint8)

    pilot_positions = np.where(pilot_mask)[0]
    n_pilots = len(pilot_positions)

    if pilot_symbols.ndim == 1:
        pilot_symbols = broadcast_channels(
            pilot_symbols, num_channels, np, name="pilot_symbols"
        )
    else:
        pilot_symbols = np.asarray(pilot_symbols, dtype=np.complex64)

    for ch in range(num_channels):
        pilot_ref_arr[ch, pilot_positions] = pilot_symbols[ch, :n_pilots]
    mask_uint8[pilot_positions] = 1

    return pilot_ref_arr, mask_uint8
