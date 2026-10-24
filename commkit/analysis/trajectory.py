"""Data-aided carrier-phase trajectory: :func:`carrier_phase_trajectory`.

The first stage of the carrier-phase analyses: the data-aided, unwrapped
phase that every downstream estimator (drift, linewidth, Allan deviation)
consumes.
"""

import numpy as np

from .._array import as_2d, broadcast_channels, restore_1d
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import adapt_signal
from ..core.signal import Signal
from ..recovery.corrections import resolve_channel_permutation

__all__ = ["carrier_phase_trajectory"]


def carrier_phase_trajectory(
    y_eq: ArrayType | Signal,
    reference: ArrayType | None = None,
    *,
    channel_pairing: str = "auto",
) -> ArrayType:
    r"""Extract the data-aided, unwrapped carrier phase of equalized symbols.

    Forms ``angle(y · conj(d))`` with the known symbols ``d``, which removes
    the data modulation (``|d|²`` is real and positive, so the carrier angle
    is kept), and unwraps it in float64.  Because the symbols are known, the
    result holds only the carrier phase and the angle noise of the AWGN, and
    it never slips a cycle; a blind or feed-forward estimate would add its
    own estimator noise and slips, which bias a linewidth estimate.  A
    constant offset or a ``±π/2`` ambiguity does not matter: only the
    variation in time is used.

    Parameters
    ----------
    y_eq : array_like or Signal
        Equalized symbols at one sample per symbol, ``(N,)`` or ``(C, N)``,
        with the carrier phase left in (e.g. the output of ``apply_taps``,
        or of an equalizer without ``cpr``).  A :class:`Signal` must be at
        one sample per symbol; its ``reference.symbols`` is the default
        reference.
    reference : array_like, optional
        Known transmitted symbols, same layout as ``y_eq``; required for
        arrays.  The two are truncated to their common length on the last
        axis (both start at the first symbol).
    channel_pairing : {"auto", "identity", "swap"}, default "auto"
        For MIMO input the equalizer may have permuted the streams (for
        dual-pol, swapped polarizations 0 and 1).  ``"auto"`` resolves the
        permutation with
        ``recovery.resolve_channel_permutation(metric="phase_increment")``,
        which picks the assignment with the lowest total variance of the
        phase-error increments, a score that a frequency offset does not
        disturb.  ``"swap"`` forces the dual-pol swap, ``"identity"`` no
        permutation.  Ignored for SISO.

    Returns
    -------
    ndarray
        Unwrapped carrier phase in radians (float64), the shape of the
        truncated input, on the device of ``y_eq``.

    Notes
    -----
    **Limitations.**

    * ``y_eq`` and ``reference`` must be symbol-aligned (same start, same
      order).  A misalignment does not raise: it turns ``y·conj(d)`` into a
      noise-like phase and inflates every downstream linewidth estimate.  ``channel_pairing="auto"`` only resolves a channel
      permutation, not a time shift.
    * The per-symbol phase *step* must stay below π for ``unwrap`` to be
      exact: ``|2πΔf·T_sym + Δφ_pn + Δφ_awgn| < π``.  This bounds the
      residual frequency offset to ``|Δf| < R/2`` and needs a moderate SNR
      (≳ 5 dB); beyond that the trajectory itself slips.
    * Residual ISI of the equalizer appears as extra white angle noise.  It
      cannot be told apart from AWGN here, which is why the downstream
      ``IncrementSlope`` fits it into its intercept instead of requiring an
      explicit noise estimate.
    """
    name = "carrier_phase_trajectory()"
    adapter = adapt_signal(y_eq, function_name=name)
    y, xp, _ = dispatch(adapter.symbol_array())
    if reference is None:
        sig = adapter.signal
        if sig is None or sig.reference is None:
            raise ValueError(
                f"{name} needs reference symbols: pass reference= or a Signal "
                "that has a reference."
            )
        reference = sig.reference.symbols

    y2, was_1d = as_2d(y, name="y_eq")
    c = y2.shape[0]
    d2 = broadcast_channels(
        to_device(reference, "cpu" if xp is np else "gpu"), c, xp, name="reference"
    )

    n = min(y2.shape[-1], d2.shape[-1])
    y2, d2 = y2[:, :n], d2[:, :n]

    if c > 1 and channel_pairing == "auto":
        # Same assignment machinery as the post-CPR resolver, scored by the
        # phase-increment metric: the carrier phase is intact here by design,
        # so the coherence metric would collapse for every pairing.
        d2 = resolve_channel_permutation(d2, y2, metric="phase_increment")
    elif c == 2 and channel_pairing == "swap":
        d2 = d2[::-1]

    # The product in the samples' precision, the unwrap in float64.
    d2 = xp.asarray(d2, dtype=np.result_type(y2.dtype, np.complex64))
    phi = xp.unwrap(xp.angle(y2 * xp.conj(d2)).astype(xp.float64), axis=-1)
    return restore_1d(was_1d, phi)
