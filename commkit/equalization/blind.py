"""Block blind equalizers (block_cma, block_rde) and pilot reference builder."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backend import ArrayType
from ..core.signal import Signal
from ..helpers import broadcast_channels, rewrap_signal, unwrap_signal
from ._block import _block_fdaf_blind
from ._common import _godard_radius, _rde_ring_radii
from .result import EqualizerResult

# -----------------------------------------------------------------------------
# BLOCK BLIND EQUALIZERS (Signal-aware)
# -----------------------------------------------------------------------------


def block_cma(
    samples: ArrayType | Signal,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 2e-4,
    block_size: int = 256,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    w_init: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
    pmf: Any | None = None,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
    cuda_graph: bool = True,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
) -> EqualizerResult:
    """Blind frequency-domain CMA equalizer (overlap-save FDAF).

    The blind, phase-directed sibling of :func:`block_lms`: the same
    overlap-save frequency-domain adaptive filter, but driven by the Godard
    constant-modulus error instead of a trained/DD slicer.  Use it for blind
    acquisition at the high-throughput frequency-domain operating point (slow or
    static channels, GPU/CuPy input); for the trained/DD case use
    :func:`block_lms`, and for fastest dynamics on a single stream use
    :func:`cma` with ``backend='numba'``.

    Like :func:`cma`, it is fully blind and recovers the channel up to a phase
    ambiguity - run a carrier-phase recovery stage afterwards.  Supplying
    ``pilot_ref``/``pilot_mask`` (build them with :func:`build_pilot_ref`)
    switches pilot positions to the LMS residual error, resolving the phase
    ambiguity there while data positions stay blind.

    ``step_size`` is on the same scale as :func:`cma`; because the weights are
    frozen across each ``block_size``-symbol block, the stability ceiling is
    ~``block_size``x lower (reduce ``mu`` only if the run raises divergence).
    The primary target is GPU (CuPy); on CPU :func:`cma` is usually faster.

    Parameters mirror :func:`cma` (no ``cpr_type`` - CMA is phase-blind, see
    :func:`cma` Notes).  On GPU, ``cuda_graph=True`` (default) captures the
    per-block FDAF body once and replays it, collapsing the per-block kernel
    launches into a single graph launch (a large win at small ``block_size``);
    it is ignored on CPU and silently disabled for the pilot-aided path.
    Returns an :class:`EqualizerResult` with ``y_hat``, ``weights``, ``error``
    on the input's device.  A :class:`Signal` returns ``y_hat`` as a new
    :class:`Signal` at the symbol rate (``sampling_rate = symbol_rate``);
    ``sps`` defaults to the signal's ``sps`` when not given explicitly.
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        result = block_cma(
            x,
            num_taps=num_taps,
            sps=sps if sps is not None else int(sig.sps),
            step_size=step_size,
            block_size=block_size,
            modulation=modulation,
            order=order,
            unipolar=unipolar,
            w_init=w_init,
            pilot_ref=pilot_ref,
            pilot_mask=pilot_mask,
            pilot_gain_db=pilot_gain_db,
            pmf=pmf,
            input_norm_factor=input_norm_factor,
            samples_prefix=samples_prefix,
            pad_mode=pad_mode,
            cuda_graph=cuda_graph,
            debug_plot=debug_plot,
            plot_smoothing=plot_smoothing,
        )
        result.y_hat = rewrap_signal(sig, result.y_hat, sampling_rate=sig.symbol_rate)
        return result

    if sps is None:
        sps = 2

    r2, c_ps = _godard_radius(modulation, order, unipolar, pmf)
    return _block_fdaf_blind(
        "cma",
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        r2=r2,
        radii_np=None,
        w_init=w_init,
        input_norm_factor=input_norm_factor,
        samples_prefix=samples_prefix,
        pad_mode=pad_mode,
        pilot_ref=pilot_ref,
        pilot_mask=pilot_mask,
        pilot_gain_db=pilot_gain_db,
        c_ps=c_ps,
        cuda_graph=cuda_graph,
        debug_plot=debug_plot,
        plot_smoothing=plot_smoothing,
        name="Block-CMA" if pilot_ref is None else "Block-CMA(PA)",
    )


def block_rde(
    samples: ArrayType | Signal,
    num_taps: int = 21,
    sps: int | None = None,
    step_size: float = 2e-4,
    block_size: int = 256,
    modulation: str | None = None,
    order: int | None = None,
    unipolar: bool = False,
    w_init: ArrayType | None = None,
    pilot_ref: ArrayType | None = None,
    pilot_mask: np.ndarray | None = None,
    pilot_gain_db: float = 0.0,
    pmf: Any | None = None,
    input_norm_factor: float | np.ndarray | None = None,
    samples_prefix: ArrayType | None = None,
    pad_mode: str = "zeros",
    cuda_graph: bool = True,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
) -> EqualizerResult:
    """Blind frequency-domain radius-directed equalizer (overlap-save FDAF).

    The multi-ring blind sibling of :func:`block_lms`, related to
    :func:`block_cma` as :func:`rde` is to :func:`cma`: it drives each symbol
    toward its nearest constellation ring instead of a single Godard circle,
    giving correct blind convergence on higher-order QAM (16/64-QAM) where CMA's
    single-radius target stalls.  Same overlap-save frequency-domain engine and
    throughput positioning as :func:`block_cma`.

    Fully blind up to a phase ambiguity (run CPR afterwards);
    ``pilot_ref``/``pilot_mask`` resolve the ambiguity at pilot positions.
    ``step_size`` is on the same scale as :func:`rde`, with the
    ~``block_size``x-lower stability ceiling of frozen-block adaptation.

    Parameters mirror :func:`rde`.  On GPU, ``cuda_graph=True`` (default)
    captures the per-block FDAF body once and replays it, collapsing the
    per-block kernel launches into a single graph launch (a large win at small
    ``block_size``); it is ignored on CPU and silently disabled for the
    pilot-aided path.  Returns an :class:`EqualizerResult` with ``y_hat``,
    ``weights``, ``error`` on the input's device.  A :class:`Signal` returns
    ``y_hat`` as a new :class:`Signal` at the symbol rate (``sampling_rate =
    symbol_rate``); ``sps`` defaults to the signal's ``sps`` when not given
    explicitly.
    """
    x, sig = unwrap_signal(samples)
    if sig is not None:
        result = block_rde(
            x,
            num_taps=num_taps,
            sps=sps if sps is not None else int(sig.sps),
            step_size=step_size,
            block_size=block_size,
            modulation=modulation,
            order=order,
            unipolar=unipolar,
            w_init=w_init,
            pilot_ref=pilot_ref,
            pilot_mask=pilot_mask,
            pilot_gain_db=pilot_gain_db,
            pmf=pmf,
            input_norm_factor=input_norm_factor,
            samples_prefix=samples_prefix,
            pad_mode=pad_mode,
            cuda_graph=cuda_graph,
            debug_plot=debug_plot,
            plot_smoothing=plot_smoothing,
        )
        result.y_hat = rewrap_signal(sig, result.y_hat, sampling_rate=sig.symbol_rate)
        return result

    if sps is None:
        sps = 2

    radii_np, c_ps = _rde_ring_radii(modulation, order, unipolar, pmf)
    return _block_fdaf_blind(
        "rde",
        samples,
        num_taps=num_taps,
        sps=sps,
        step_size=step_size,
        block_size=block_size,
        r2=1.0,
        radii_np=radii_np,
        w_init=w_init,
        input_norm_factor=input_norm_factor,
        samples_prefix=samples_prefix,
        pad_mode=pad_mode,
        pilot_ref=pilot_ref,
        pilot_mask=pilot_mask,
        pilot_gain_db=pilot_gain_db,
        c_ps=c_ps,
        cuda_graph=cuda_graph,
        debug_plot=debug_plot,
        plot_smoothing=plot_smoothing,
        name="Block-RDE" if pilot_ref is None else "Block-RDE(PA)",
    )


# -----------------------------------------------------------------------------
# PILOT REFERENCE BUILDER (array-only)
# -----------------------------------------------------------------------------
# build_pilot_ref packs sparse pilot symbols into a dense reference array for
# cma/rde's pilot_ref/pilot_mask - a construction helper for equalizer input
# parameters, not a transform on a Signal, so it takes plain arrays only.


def build_pilot_ref(
    pilot_symbols: np.ndarray,
    pilot_mask: np.ndarray,
    n_sym: int,
    num_ch: int,
) -> tuple:
    """Build dense pilot reference array and uint8 mask for the hybrid PA kernel.

    Packs sparse pilot symbols into a dense ``(C, n_sym)`` array suitable for
    passing to ``cma`` or ``rde`` as ``pilot_ref`` / ``pilot_mask``.
    Data positions are filled with zeros; the mask marks which positions carry
    known reference symbols.

    Parameters
    ----------
    pilot_symbols : (K,) or (C, K) complex64 ndarray
        Known pilot symbols in transmission order (only the K pilot positions,
        no data symbols).  A 1-D array is broadcast to all C channels.
    pilot_mask : (n_sym,) bool ndarray
        ``True`` at the K pilot positions within the equalized body region.
    n_sym : int
        Total number of output symbols (payload + pilots).
    num_ch : int
        Number of receive channels C.

    Returns
    -------
    pilot_ref : (C, n_sym) complex64 ndarray
        Dense reference array - zeros at data positions, pilot symbols at
        pilot positions.
    pilot_mask_u8 : (n_sym,) uint8 ndarray
        ``1`` at pilot positions, ``0`` elsewhere.

    Examples
    --------
    Build the reference from a ``SingleCarrierFrame`` and
    pass it directly to ``rde``:

    ::

        struct = frame.get_structure_map(unit="symbols", sps=1, include_preamble=False)
        pilot_ref, pilot_mask_u8 = build_pilot_ref(
            pilot_symbols=frame.pilot_symbols,
            pilot_mask=np.asarray(struct["pilots"]),
            n_sym=n_body_symbols,
            num_ch=num_ch,
        )
        result = rde(body_samples, ..., pilot_ref=pilot_ref, pilot_mask=pilot_mask_u8)
    """
    pilot_ref_arr = np.zeros((num_ch, n_sym), dtype=np.complex64)
    mask_uint8 = np.zeros(n_sym, dtype=np.uint8)

    pilot_positions = np.where(pilot_mask)[0]
    n_pilots = len(pilot_positions)

    if pilot_symbols.ndim == 1:
        pilot_symbols = broadcast_channels(
            pilot_symbols, num_ch, np, name="pilot_symbols"
        )
    else:
        pilot_symbols = np.asarray(pilot_symbols, dtype=np.complex64)

    for ch in range(num_ch):
        pilot_ref_arr[ch, pilot_positions] = pilot_symbols[ch, :n_pilots]
    mask_uint8[pilot_positions] = 1

    return pilot_ref_arr, mask_uint8
