"""Data-aided carrier-phase trajectory extraction.

The foundational extractor of the carrier-phase analysis chain: forms the
data-aided unwrapped phase that every downstream estimator (drift, linewidth,
Allan deviation) consumes.
"""

from ..backend import ArrayType, dispatch
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..helpers import as_2d, broadcast_channels, restore_1d
from ..recovery.corrections import resolve_channel_permutation

__all__ = ["carrier_phase_trajectory"]


def carrier_phase_trajectory(
    y_eq: ArrayType | Signal,
    ref_symbols: ArrayType,
    *,
    channel_pairing: str = "auto",
) -> ArrayType:
    r"""Data-aided unwrapped carrier phase from frozen-tap output + known symbols.

    Forms ``angle(y · conj(d))`` (which cancels the data modulation - for QAM
    ``|d|²`` is real-positive so the carrier angle is preserved) and unwraps it
    in ``float64``.  Because the symbols are *known*, the result carries only
    carrier phase + AWGN angle noise and **never cycle-slips** - unlike a
    blind/feed-forward estimate which would add its own estimator noise and
    slips that corrupt a linewidth estimate.  A constant offset or
    +/- pi/2 ambiguity is irrelevant; only the time variation is used.

    Parameters
    ----------
    y_eq : array_like or Signal
        Equalized symbols at 1 sps (e.g. ``apply_taps``
        output with the CPR **disabled** so the carrier phase is left intact).
        Shape ``(N,)`` (SISO) or ``(C, N)`` (MIMO, time on last axis).  A
        :class:`Signal` is unwrapped to its ``.samples``.
    ref_symbols : array_like
        Known transmitted symbols, same layout as ``y_eq``.  The two are
        truncated to their common length on the last axis.
    channel_pairing : {"auto", "identity", "swap"}, default "auto"
        For MIMO inputs the equalizer may permute the streams (for dual-pol,
        map pol 0<->1).  ``"auto"`` resolves it with
        ``recovery.resolve_channel_permutation(metric="phase_increment")``,
        which picks the assignment of lowest total phase-error increment
        variance - the frequency-offset-immune scoring this stage needs.
        ``"swap"`` forces the dual-pol swap and ``"identity"`` forces none.
        Ignored for SISO.

    Returns
    -------
    array_like
        Unwrapped carrier phase in radians (``float64``), shape matching the
        truncated input, on the same backend as ``y_eq``.

    Notes
    -----
    **Limitations.**

    * ``y_eq`` and ``ref_symbols`` must be *symbol-aligned* (same start, same
      ordering).  A misalignment does not fail loudly - it turns the product
      ``y·conj(d)`` into noise-like phase and inflates every downstream
      linewidth estimate.  ``channel_pairing="auto"`` only resolves a channel
      permutation, not a time shift.
    * The per-symbol phase *step* must stay below π for ``unwrap`` to be
      exact: ``|2πΔf·T_sym + Δφ_pn + Δφ_awgn| < π``.  In practice this bounds
      the residual frequency offset to ``|Δf| < R/2`` per symbol and requires
      moderate SNR (≳ 5 dB); beyond that the trajectory itself slips.
    * Residual equalizer ISI appears as extra white angle noise.  It is
      indistinguishable from AWGN here, which is why the downstream
      ``linewidth_increment(method="slope")`` fits it into the intercept
      instead of requiring an explicit noise estimate.
    """
    y_eq = adapt_signal(y_eq, function_name="carrier_phase_trajectory()").array
    y, xp, _ = dispatch(y_eq)

    y2, was_1d = as_2d(y, name="y_eq")
    c = y2.shape[0]
    d2 = broadcast_channels(xp.asarray(ref_symbols), c, xp, name="ref_symbols")

    n = min(y2.shape[-1], d2.shape[-1])
    y2, d2 = y2[:, :n], d2[:, :n]

    if c > 1 and channel_pairing == "auto":
        # Same assignment machinery as the post-CPR resolver, scored by the
        # phase-increment metric: the carrier phase is intact here by design,
        # so the coherence metric would collapse for every pairing.
        d2 = resolve_channel_permutation(d2, y2, metric="phase_increment")
    elif c == 2 and channel_pairing == "swap":
        d2 = d2[::-1]

    phi = xp.unwrap(xp.angle(y2 * xp.conj(d2)).astype(xp.float64), axis=-1)
    return restore_1d(was_1d, phi)
