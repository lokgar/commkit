"""Phase corrections, cycle-slip repair, and ambiguity resolution."""

import logging

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..core.signal import Signal
from ..helpers import (
    as_2d,
    broadcast_channels,
    restore_1d,
    rewrap_signal,
    unwrap_signal,
)
from ..logger import logger


def smooth_phase_wiener(
    phase: ArrayType,
    process_variance: float | None = None,
    measurement_variance: float | None = None,
    linewidth: float | None = None,
    sampling_rate: float | None = None,
    detrend: bool = True,
) -> ArrayType:
    r"""
    Zero-phase Wiener smoother for a random-walk (Wiener) carrier phase.

    Optimal minimum-variance estimate of a phase that random-walks with
    per-sample increment variance ``q`` (the random-walk strength) observed in
    white phase-estimation noise of variance ``r``.  Applies the non-causal
    Wiener filter

        H(w)       = S_phi(w) / (S_phi(w) + r),
        S_phi(w)   = q / (2 - 2*cos(w)),

    in the frequency domain (FFT -> multiply by the real, even H -> IFFT), so it
    is zero-phase (no group delay) and O(N log N).  This is the principled way to
    hit the smallest residual phase std for a given ``q`` and ``r`` - it trades
    tracking lag against additive noise automatically, where a fixed extraction
    bandwidth must be tuned by hand.

    The smoother is agnostic to where the track came from: it needs only the two
    scalars ``q`` and ``r``.  ``q`` may be given directly or derived from a
    linewidth (see ``process_variance`` / ``linewidth`` below); ``r`` is supplied
    as ``measurement_variance``, however the caller measured it.

    Because it only rescales the phase track (a deterministic, sample-independent
    low-pass), it stays a unit-modulus correction downstream and cannot hide
    excess noise when applied to a common reference phase rather than the data
    samples.

    Parameters
    ----------
    phase : (N,) or (C, N) array
        Unwrapped phase track (e.g. from a ``recover_carrier_phase_pilot_tone*``
        function), in radians.
    process_variance : float, optional
        Per-sample phase-increment variance q [rad²] - the random-walk strength.
        Provide this, or derive it from ``linewidth`` + ``sampling_rate`` (see
        below).  Exactly one of the two routes is required.
    measurement_variance : float, optional
        Phase-estimation noise variance r [rad²] - the per-sample variance of the
        additive phase-measurement noise, however it was measured.  Required.
    linewidth, sampling_rate : float, optional
        Convenience route to ``process_variance``: q = 2*pi*linewidth / f_s, where
        ``linewidth`` is the combined oscillator linewidth [Hz] and
        ``sampling_rate`` is f_s [Hz] of the ``phase`` track.  Both must be given
        together, and only when ``process_variance`` is omitted.
    detrend : bool, default True
        Remove the per-channel mean + linear trend before filtering and add it
        back after.  Recommended: the random-walk PSD diverges at DC, so a raw
        ramp (residual frequency offset) would be distorted; detrending keeps it
        exact.

    Returns
    -------
    array_like
        Smoothed phase, same shape and backend as ``phase``.
    """
    if process_variance is None:
        if linewidth is None or sampling_rate is None:
            raise ValueError(
                "Provide process_variance, or both linewidth and sampling_rate."
            )
        process_variance = 2.0 * np.pi * float(linewidth) / float(sampling_rate)
    if measurement_variance is None:
        raise ValueError("Provide measurement_variance.")
    q, r = float(process_variance), float(measurement_variance)
    if q <= 0.0 or r <= 0.0:
        raise ValueError(f"process/measurement variance must be > 0, got q={q}, r={r}.")

    phase, xp, _ = dispatch(phase)
    phase, was_1d = as_2d(phase, name="phase")
    C, N = phase.shape
    phi = phase.astype(xp.float64)

    # Detrend per channel (mean + linear) so the DC-divergent random-walk PSD
    # does not distort the residual-FOE ramp; restore the trend after filtering.
    n = xp.arange(N, dtype=xp.float64)
    if detrend:
        nc = n - xp.mean(n)
        denom = xp.sum(nc * nc)
        mean = xp.mean(phi, axis=-1, keepdims=True)
        slope = xp.sum((phi - mean) * nc[None, :], axis=-1, keepdims=True) / denom
        trend = mean + slope * nc[None, :]
    else:
        trend = xp.zeros((C, 1), dtype=xp.float64)
    phi_c = phi - trend

    # Real, even Wiener gain H(ω); keep DC (H[0]=1) where S_φ -> ∞.  The phase
    # track is real, so the filter runs on the half spectrum (rfft/irfft) -
    # half the transform work and spectrum memory of the full complex FFT.
    omega = 2.0 * xp.pi * xp.fft.rfftfreq(N)
    denom_w = 2.0 - 2.0 * xp.cos(omega)
    denom_w = xp.where(denom_w <= 0.0, xp.full_like(denom_w, 1e-300), denom_w)
    S = q / denom_w
    H = S / (S + r)
    H[0] = 1.0

    phi_s = xp.fft.irfft(xp.fft.rfft(phi_c, axis=-1) * H[None, :], n=N, axis=-1)
    phi_s = phi_s + trend

    if logger.isEnabledFor(logging.INFO):
        # Two std reductions + host syncs, needed only for the line below.
        std_in = float(xp.std(phi_c))
        std_out = float(xp.std(phi_s - trend))
        logger.info(
            "Wiener phase smoother: q=%.3g, r=%.3g rad², residual std %.2f° -> %.2f°.",
            q,
            r,
            np.degrees(std_in),
            np.degrees(std_out),
        )

    return restore_1d(was_1d, phi_s)


_PHASE_ROTATE_KERNEL: dict = {}


def _get_cupy_phase_rotate():
    """Compile and cache the fused CuPy phase-rotation kernel.

    Fuses the whole ``s · exp(-j·wrap(φ))`` chain - float64 wrap, float32
    sin/cos, complex multiply - into a single elementwise kernel: one read of
    the symbols, one read of the phase, one write, instead of the ~7 separate
    full-record kernel passes and temporaries of the ufunc chain.
    """
    if "k" not in _PHASE_ROTATE_KERNEL:
        import cupy as cp

        _PHASE_ROTATE_KERNEL["k"] = cp.ElementwiseKernel(
            "complex64 s, float64 phi",
            "complex64 out",
            """
            double w = phi - rint(phi * 0.15915494309189535) * 6.283185307179586;
            float sw, cw;
            sincosf((float)w, &sw, &cw);
            out = s * complex<float>(cw, -sw);
            """,
            "commkit_phase_rotate",
        )
    return _PHASE_ROTATE_KERNEL["k"]


def correct_carrier_phase(
    symbols: ArrayType | Signal,
    phase_vector: ArrayType,
) -> ArrayType | Signal:
    """
    Applies carrier phase correction to a symbol sequence.

    Rotates each symbol by the negative of the estimated phase to cancel
    the carrier phase offset: y[n] = s[n] * exp(-j * phi_hat[n]).

    Parameters
    ----------
    symbols : array_like or Signal
        Complex symbols. Shape: (N,) or (C, N).
    phase_vector : array_like
        Per-symbol phase estimates in radians.  Shape: (N,) for SISO, or
        broadcastable to ``symbols.shape`` for MIMO.

    Returns
    -------
    array_like or Signal
        Phase-corrected symbols, same shape and dtype as ``symbols``.  A
        :class:`Signal` returns a new phase-corrected :class:`Signal`.
    """
    x, sig = unwrap_signal(symbols)
    if sig is not None:
        return rewrap_signal(sig, correct_carrier_phase(x, phase_vector))

    symbols, xp, _ = dispatch(symbols)
    logger.debug("Applying carrier phase correction: shape=%s", symbols.shape)
    # Wrap to [-π, π] in float64 (handles unbounded phase trajectories from
    # standalone CPR), then rotate with a float32 phasor.
    phase_f64 = xp.asarray(phase_vector, dtype=xp.float64)
    if xp is not np and symbols.dtype == xp.complex64:
        # GPU fast path: single fused kernel (broadcasts (N,) phase over (C, N)).
        return _get_cupy_phase_rotate()(symbols, phase_f64)
    two_pi = 2.0 * xp.pi
    phase_wrapped = (phase_f64 - xp.round(phase_f64 / two_pi) * two_pi).astype(
        xp.float32
    )
    phasor = xp.exp(-1j * phase_wrapped)
    if phasor.dtype != symbols.dtype:
        phasor = phasor.astype(symbols.dtype)
    return symbols * phasor


_NUMBA_CYCLE_SLIP: dict = {}


def _get_numba_cycle_slip():
    """JIT-compile and cache the Numba cycle-slip correction kernel.

    Returns
    -------
    callable
        Numba-compiled ``_cycle_slip_loop``.
    """
    if "cs" not in _NUMBA_CYCLE_SLIP:
        import numba

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _cycle_slip_loop(phi_u, symmetry, history_length, threshold):
            """Cycle-slip detection and correction via linear extrapolation.

            Scans the block-phase trajectory ``phi_u`` sequentially.  For each
            block, linearly extrapolates from up to ``history_length`` past
            *corrected* blocks.  When the deviation exceeds ``threshold``, a
            ``π/2`` step correction is applied.

            The linear regression uses relative coordinates [0, W-1] so that
            ``Sx`` and ``Sxx`` are exact compile-time constants.  Only ``Sy``
            and ``Sxy`` are maintained as running state, updated in O(1) per
            step via a closed-form sliding-window identity.

            Parameters
            ----------
            phi_u : (B,) float64
                Block-phase trajectory after M-fold unwrap (modified in place).
            symmetry : int
                Rotational symmetry order; correction quantum = ``2π/symmetry``.
                Pass 4 for QAM (all BPS, VV, Tikhonov use 4-fold symmetry).
            history_length : int
                Maximum number of past corrected blocks used for extrapolation.
                Use ``min(b, history_length)`` at each step.
            threshold : float64
                Deviation from extrapolated value that triggers a correction
                (radians).  Default in the caller: ``π/4``.

            Returns
            -------
            (B,) float64
                Corrected block-phase trajectory (same array, modified in place).
            """
            two_pi = 2.0 * np.pi
            quantum = two_pi / float(symmetry)
            B = len(phi_u)
            W = history_length
            W_f = float(W)

            # Precompute full-window regression constants in relative coords [0, W-1].
            # With relative coords the x-values are always small integers, so Sx and
            # Sxx never grow and there is no catastrophic cancellation regardless of
            # how many total blocks have been processed.
            Sx_full = W_f * (W_f - 1.0) / 2.0
            Sxx_full = W_f * (W_f - 1.0) * (2.0 * W_f - 1.0) / 6.0
            denom_full = W_f * Sxx_full - Sx_full * Sx_full  # W²(W²-1)/12

            # Only Sy and Sxy need to be maintained as running state.
            buf_y = np.empty(W, dtype=np.float64)
            buf_head = 0  # next write slot (circular)
            n_buf = 0  # valid entries currently in buffer

            Sy = 0.0
            Sxy = 0.0

            for b in range(B):
                y_b = phi_u[b]

                if n_buf == 0:
                    # First block: trust it unconditionally (at relative position 0).
                    buf_y[0] = y_b
                    buf_head = 1
                    n_buf = 1
                    Sy = y_b
                    Sxy = 0.0  # 0 * y_b
                    continue

                if n_buf < min(10, W):
                    # Constant extrapolation during warmup to avoid cementing false slips.
                    phi_pred = buf_y[(buf_head - 1) % W]
                else:
                    # Linear extrapolation.  Prediction target is always one step past
                    # the newest buffered entry, i.e. relative coordinate = n_buf.
                    x_pred = float(n_buf)
                    n_f = float(n_buf)
                    if n_buf < W:
                        # Partial window: derive exact Sx/Sxx from closed-form sums.
                        Sx_p = n_f * (n_f - 1.0) / 2.0
                        Sxx_p = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
                        denom = n_f * Sxx_p - Sx_p * Sx_p
                        if abs(denom) > 1e-30:
                            slope = (n_f * Sxy - Sx_p * Sy) / denom
                            intercept = (Sy - slope * Sx_p) / n_f
                        else:
                            slope = 0.0
                            intercept = Sy / n_f
                    else:
                        # Full window: use precomputed constants (numerically exact).
                        if denom_full > 1e-30:
                            slope = (W_f * Sxy - Sx_full * Sy) / denom_full
                            intercept = (Sy - slope * Sx_full) / W_f
                        else:
                            slope = 0.0
                            intercept = Sy / W_f
                    phi_pred = slope * x_pred + intercept

                diff = y_b - phi_pred
                # Round to nearest correction quantum
                k = round(diff / quantum)
                if abs(diff) > threshold and k != 0:
                    phi_u[b] -= float(k) * quantum
                    y_b = phi_u[b]

                # Update circular buffer using relative coordinates.
                if n_buf == W:
                    # Slide window: evict oldest (relative pos 0), shift all down by 1,
                    # add y_b at relative position W-1.
                    # Sxy update uses the identity:
                    #   Sxy_new = Sxy_old - Sy_old + y_old + (W-1)·y_new
                    # (derived by relabelling positions after eviction)
                    old_idx = buf_head % W
                    y_old = buf_y[old_idx]
                    Sxy = Sxy - Sy + y_old + (W_f - 1.0) * y_b  # must precede Sy update
                    Sy = Sy - y_old + y_b
                    buf_y[old_idx] = y_b
                    buf_head += 1
                else:
                    # Append at relative position n_buf.
                    idx = buf_head % W
                    buf_y[idx] = y_b
                    Sxy += float(n_buf) * y_b
                    Sy += y_b
                    buf_head += 1
                    n_buf += 1

            return phi_u

        _NUMBA_CYCLE_SLIP["cs"] = _cycle_slip_loop

    return _NUMBA_CYCLE_SLIP["cs"]


def correct_cycle_slips(
    phi_u: np.ndarray,
    symmetry: int = 4,
    history_length: int = 1000,
    threshold: float = np.pi / 4,
) -> np.ndarray:
    """
    Detects and corrects cycle slips in a block-phase trajectory.

    After ``xp.unwrap`` resolves the M-fold ambiguity, residual cycle slips
    may remain where the unwrapper chose the wrong quadrant.  This function
    scans the trajectory sequentially: for each block it extrapolates the
    expected phase from up to ``history_length`` past corrected blocks using
    a linear fit.  When the deviation exceeds ``threshold``, the block is
    corrected by the nearest integer multiple of ``2π/symmetry``.

    Algorithm: linear extrapolation from the previous
    ``history_length`` corrected phases; correction quantum = ``π/2`` for
    4-fold QAM symmetry; threshold = ``π/4``.

    Parameters
    ----------
    phi_u : (B,) float64
        Block-phase trajectory on CPU after M-fold unwrap (e.g. output of
        ``xp.unwrap(phi_raw * M) / M``).  **Modified in place.**
    symmetry : int, default 4
        Rotational symmetry order of the constellation.  Correction quantum
        is ``2π/symmetry``.  Use 4 for all square QAM constellations and BPS
        (which always searches over ``[0, π/2)``).  For M-PSK use ``symmetry = M``.
    history_length : int, default 1000
        Number of past corrected blocks used for linear extrapolation.
        Reduce for short bursts.
    threshold : float, default π/4
        Deviation from the extrapolated phase that triggers a correction.
        ``π/4`` is the midpoint between adjacent correction quanta for 4-fold
        symmetry.

    Returns
    -------
    (B,) float64
        Corrected block-phase trajectory (same NumPy array).

    Notes
    -----
    Runs on CPU only (sequential scan; Numba-compiled).
    The caller should transfer ``phi_u`` to CPU before calling and move
    the result back to the device if needed.
    """
    phi_u = np.asarray(phi_u, dtype=np.float64)
    kernel = _get_numba_cycle_slip()
    return kernel(phi_u, int(symmetry), int(history_length), float(threshold))


def _pairing_scores(y, s, xp, metric: str):
    """``(C_out, C_ref)`` match scores between output and reference streams.

    Higher is better, so the assignment below maximizes the total score.

    * ``"coherence"`` - rotation-invariant normalized cross-correlation
      magnitude ``|<y_i, s_j>| / (||y_i|| ||s_j||)``, in ``[0, 1]``.  One
      matmul, and blind to any constant phase rotation.
    * ``"phase_increment"`` - minus the variance of the *wrapped phase-error
      increment* ``Var(Δ angle(y_i · conj(s_j)))``.  Immune not only to a
      constant rotation but to a residual frequency offset and to phase noise,
      which drive the coherence sum toward zero for every pairing (the
      rotating phasor averages out) and make ``"coherence"`` unusable on
      records whose carrier phase is deliberately left intact.  Evaluated one
      reference stream at a time so the working set stays ``(C, N)`` rather
      than ``(C, C, N)``.
    """
    if metric == "coherence":
        yn = y / xp.maximum(xp.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
        sn = s / xp.maximum(xp.linalg.norm(s, axis=-1, keepdims=True), 1e-12)
        return to_device(xp.abs(yn @ xp.conj(sn).T), "cpu")
    if metric == "phase_increment":
        cols = [
            xp.var(
                xp.diff(
                    xp.angle(y * xp.conj(s[j][None, :])).astype(xp.float64), axis=-1
                ),
                axis=-1,
            )
            for j in range(s.shape[0])
        ]
        return to_device(-xp.stack(cols, axis=-1), "cpu")
    raise ValueError(
        f"Unknown metric {metric!r} (use 'coherence' or 'phase_increment')."
    )


def resolve_channel_permutation(
    symbols: ArrayType | Signal,
    ref_symbols: ArrayType | None = None,
    *,
    num_skip_symbols: int = 0,
    metric: str = "coherence",
) -> ArrayType | Signal:
    """Resolve a polarization (channel) permutation after MIMO equalization.

    A MIMO (butterfly) equalizer has a **polarization-permutation ambiguity**:
    it may emit the streams in swapped output order (output 0 carries pol 1,
    etc.) - a perfectly valid demux that per-channel metrics would otherwise
    score as random, since they compare ``output[i]`` with ``ref[i]``.  This
    matches each output stream to the reference stream it actually carries (the
    bijective assignment maximizing the **rotation-invariant** cross-correlation
    magnitude ``|Σ yᵢ · conj(sⱼ)|``) and reorders ``symbols`` to ``ref_symbols``
    order.

    Run this **before** ``resolve_phase_ambiguity`` (it is rotation
    invariant, so the two compose) and before SER/BER.  For a converged
    *data-aided* equalizer the outputs are already pinned to the training order,
    so this is a no-op; it is the robust fix for **blind** equalizers, whose
    output order is arbitrary.  Only a *constant* permutation is resolved - a
    mid-stream swap is an equalizer-tracking issue, not a labeling one.

    Parameters
    ----------
    symbols : array_like
        Recovered symbols, ``(N,)`` or ``(C, N)``.  Returned unchanged for SISO.
    ref_symbols : array_like
        Known transmitted symbols, same layout as ``symbols`` (the full
        sequence, a pilot subset, or any known reference).
    num_skip_symbols : int, default 0
        Leading symbols excluded from the correlation scoring (e.g. an
        unconverged transient).  The reorder still covers the full input.
    metric : {"coherence", "phase_increment"}, default "coherence"
        How output streams are scored against reference streams.
        ``"coherence"`` is the rotation-invariant correlation magnitude - the
        right choice after carrier recovery.  ``"phase_increment"`` scores by
        the (negated) variance of the wrapped phase-error increment, which
        survives a residual frequency offset and strong phase noise; use it on
        records whose carrier phase is deliberately intact (e.g. frozen-tap
        output feeding ``analysis.carrier_phase_trajectory``), where the
        coherence sum collapses toward zero for *every* pairing and the
        assignment would be arbitrary.

    Returns
    -------
    array_like
        ``symbols`` with channels reordered to match ``ref_symbols``; same
        shape, dtype, and backend.
    When ``symbols`` is a :class:`Signal`, ``resolved_symbols`` is reordered
    against ``source_symbols`` and a new :class:`Signal` is returned.
    """
    if isinstance(symbols, Signal):
        sig = symbols
        if sig.resolved_symbols is None:
            raise ValueError(
                "resolved_symbols is not set. Call resolve_symbols(sig) or assign "
                "resolved_symbols before calling resolve_channel_permutation()."
            )
        if sig.source_symbols is None:
            raise ValueError(
                "source_symbols is not set. Populate source_symbols (the known TX "
                "symbol sequence) before calling resolve_channel_permutation()."
            )
        # Output field (resolved_symbols) is written by hand: rewrap_signal
        # always targets .samples, and here input and output share a field
        # name that isn't it.
        resolved_symbols, _ = unwrap_signal(sig, field="resolved_symbols")
        new = sig.copy()
        new.resolved_symbols = resolve_channel_permutation(
            resolved_symbols,
            sig.source_symbols,
            num_skip_symbols=num_skip_symbols,
            metric=metric,
        )
        return new

    if ref_symbols is None:
        raise ValueError("resolve_channel_permutation() requires ref_symbols.")

    from scipy.optimize import linear_sum_assignment

    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        return symbols
    C, N = symbols.shape
    if C == 1:
        return symbols

    ref = broadcast_channels(xp.asarray(ref_symbols), C, xp, name="ref_symbols")
    n = min(N, ref.shape[-1])
    y = symbols[:, num_skip_symbols:n]
    s = ref[:, num_skip_symbols:n]

    M = _pairing_scores(y, s, xp, metric)  # (C_out, C_ref), higher = better

    _, perm = linear_sum_assignment(-M)  # perm[i] = ref stream matched by output i
    perm = np.asarray(perm)
    inv = np.argsort(perm)  # reorder: out'[j] is the output carrying ref j

    assigned = M[np.arange(C), perm]
    is_identity = bool(np.array_equal(perm, np.arange(C)))
    matrix_str = np.array2string(M, precision=2, suppress_small=True)
    if metric == "coherence":
        weak = float(assigned.min()) < 0.3
        quality = f"min coherence {float(assigned.min()):.2f}"
    else:
        # A mismatched pairing leaves a uniformly distributed phase error,
        # whose increment variance approaches 2π²/3 ≈ 6.6 rad²; a locked
        # pairing sits orders of magnitude below that.
        worst = float(-assigned.min())
        weak = worst > 1.0
        quality = f"max increment variance {worst:.2f} rad²"
    if weak:
        # An output did not lock to any distinct reference stream - the demux
        # likely collapsed (both outputs on one pol) rather than swapped.
        logger.warning(
            "resolve_channel_permutation: weak match (%s) - streams may not "
            "be cleanly separated (EQ collapse?). Applying best assignment "
            "%s anyway. Score matrix (rows=out, cols=ref):\n%s",
            quality,
            perm.tolist(),
            matrix_str,
        )
    elif is_identity:
        logger.info(
            "resolve_channel_permutation: identity %s (no swap).", perm.tolist()
        )
    else:
        logger.info(
            "resolve_channel_permutation: POLARIZATION SWAP %s - "
            "reordering outputs to reference order. "
            "Score matrix (rows=out, cols=ref):\n%s",
            perm.tolist(),
            matrix_str,
        )

    return symbols[xp.asarray(inv)]


def resolve_phase_ambiguity(
    symbols: ArrayType | Signal,
    ref_symbols: ArrayType | None = None,
    modulation: str | None = None,
    order: int | None = None,
    symmetry_order: int | None = None,
    num_skip_symbols: int = 0,
    pmf: np.ndarray | None = None,
) -> ArrayType | Signal:
    """
    Resolves rotational phase ambiguity after blind carrier phase recovery.

    Blind CPR methods (VV, BPS, Tikhonov) cannot distinguish between
    ``symmetry_order`` rotational copies of the constellation.  This function
    tests all candidate rotations, scores each by Symbol Error Rate (SER)
    against the known transmitted symbols, and returns the symbols rotated by
    the best candidate.

    For MIMO inputs each channel is resolved independently - after MIMO
    equalisation the output streams may land on different ambiguity branches.

    Parameters
    ----------
    symbols : array_like
        Received complex symbols after CPR and ``correct_carrier_phase``.
        Shape: ``(N,)`` or ``(C, N)``.
    ref_symbols : array_like
        Known transmitted symbols (unit-average-power normalised).
        Shape: ``(N,)`` or ``(C, N)``.
    modulation : str
        Modulation scheme (case-insensitive): ``'qam'``, ``'psk'``, etc.
    order : int
        Modulation order.
    symmetry_order : int, optional
        Number of rotationally equivalent constellation copies to test.
        Defaults to 4 for QAM (4-fold ``π/2`` symmetry) and ``order`` for
        PSK.  Override for non-standard constellations.
    num_skip_symbols : int, default 0
        Number of leading symbols to exclude from SER scoring.  The applied
        rotation still covers the full input - only the scoring window is
        trimmed.  Useful when the first ``num_skip_symbols`` symbols have not
        yet converged and would bias the rotation choice.  Must be strictly
        less than the total symbol count.
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  Forwarded to
        ``ser`` so the diagnostic SER reported in
        the log is unbiased for shaped constellations.  The phase-rotation
        choice itself uses a scale-invariant inner product and does not
        depend on ``pmf``.

    Returns
    -------
    array_like
        Phase-ambiguity-resolved symbols, same shape and dtype as ``symbols``.

    When ``symbols`` is a :class:`Signal`, ``resolved_symbols`` is resolved
    against ``source_symbols`` (using the signal's modulation/order/pmf) and a
    new :class:`Signal` is returned.
    """
    if isinstance(symbols, Signal):
        sig = symbols
        if sig.resolved_symbols is None:
            raise ValueError(
                "resolved_symbols is not set. Call resolve_symbols(sig) or assign "
                "resolved_symbols before calling resolve_phase_ambiguity()."
            )
        if sig.source_symbols is None:
            raise ValueError(
                "source_symbols is not set. Populate source_symbols (the known TX "
                "symbol sequence) before calling resolve_phase_ambiguity()."
            )
        if sig.mod_scheme is None or sig.mod_order is None:
            raise ValueError("mod_scheme and mod_order must be set.")
        resolved_symbols, _ = unwrap_signal(sig, field="resolved_symbols")
        new = sig.copy()
        new.resolved_symbols = resolve_phase_ambiguity(
            resolved_symbols,
            sig.source_symbols,
            sig.mod_scheme,
            sig.mod_order,
            symmetry_order=symmetry_order,
            num_skip_symbols=num_skip_symbols,
            pmf=sig.ps_pmf,
        )
        return new

    if ref_symbols is None or modulation is None or order is None:
        raise ValueError(
            "resolve_phase_ambiguity() requires ref_symbols, modulation, and order."
        )

    from ..metrics import ser as _ser

    symbols, xp, _ = dispatch(symbols)
    symbols, was_1d = as_2d(symbols, name="symbols")
    C, N = symbols.shape

    if num_skip_symbols >= N:
        raise ValueError(
            f"num_skip_symbols={num_skip_symbols} must be less than the total "
            f"symbol count N={N}."
        )

    ref = broadcast_channels(xp.asarray(ref_symbols), C, xp, name="ref_symbols")

    if symmetry_order is None:
        symmetry_order = 4 if "qam" in modulation.lower() else order

    step = 2.0 * np.pi / symmetry_order

    # ML phase ambiguity estimator: the optimal rotation maximises
    # Re(e^{jkθ} · Σ y_n s_n*), which equals choosing k closest to
    # -∠(Σ y_n s_n*) / step.  Single inner product replaces symmetry_order
    # full SER passes.  All channels batched: one D2H of the (C,) angles
    # instead of one float() sync per channel.
    seg_y = symbols[:, num_skip_symbols:]
    seg_r = ref[:, num_skip_symbols:]
    corr = xp.sum(seg_y * xp.conj(seg_r), axis=-1)  # (C,)
    theta_np = -to_device(xp.angle(corr), "cpu")  # (C,) float64, one transfer
    best_k_np = np.round(theta_np / step).astype(np.int64) % symmetry_order
    phasors = xp.asarray(
        np.exp(1j * best_k_np * step).astype(symbols.dtype)
    )  # (C,) - built on host from host indices, single H2D
    out = symbols * phasors[:, None]

    # SER is diagnostic-only: skip the per-channel reduction syncs entirely
    # when INFO logging is disabled.
    if logger.isEnabledFor(logging.INFO):
        for ch in range(C):
            best_ser = float(
                xp.mean(
                    xp.asarray(
                        _ser(
                            out[ch, num_skip_symbols:],
                            seg_r[ch],
                            modulation,
                            order,
                            pmf=pmf,
                        )
                    )
                )
            )
            logger.info(
                "Phase ambiguity resolution: ch=%s, best_k=%s, "
                "rotation=%.1f°, SER=%.4f",
                ch,
                int(best_k_np[ch]),
                best_k_np[ch] * step * 180.0 / np.pi,
                best_ser,
            )

    return restore_1d(was_1d, out)


def correct_phase_rotation(
    symbols: ArrayType | Signal,
    ref_symbols: ArrayType | None = None,
    num_skip_symbols: int = 0,
) -> ArrayType | Signal:
    """Correct the static per-channel phase rotation using a reference sequence.

    A rotationally-invariant blind equalizer (CMA, RDE) leaves an arbitrary
    constant phase offset on each output channel - not limited to the discrete
    ``k·π/M`` grid that ``resolve_phase_ambiguity`` tests.  This function
    estimates the continuous rotation per channel via the ML inner-product
    estimator ``θ = -∠(Σ y·s*)`` over a known reference sequence and applies
    the correction to the full symbol block.

    The reference may be shorter than ``symbols`` (e.g. a transmitted preamble
    or the first ``N_ref`` source symbols); estimation uses only the overlapping
    window.

    Parameters
    ----------
    symbols : array_like or Signal
        Equalizer output symbols.  Shape: ``(N,)`` or ``(C, N)``.  When a
        :class:`Signal` is passed, ``resolved_symbols`` is corrected and
        ``ref_symbols`` defaults to ``source_symbols``.
    ref_symbols : array_like, optional
        Known transmitted symbols.  Shape: ``(N_ref,)`` or ``(C, N_ref)``,
        where ``N_ref <= N``.  Each channel is matched independently; a
        single-channel ref is broadcast across all output channels.
        Required for array input.
    num_skip_symbols : int, default 0
        Leading symbols excluded from the rotation estimate (e.g. the
        unconverged equalizer transient).  The correction is still applied
        to the full ``symbols``.

    Returns
    -------
    array_like or Signal
        Phase-corrected symbols, same shape and dtype as ``symbols``.  A
        :class:`Signal` returns a new :class:`Signal` with ``resolved_symbols``
        corrected.
    """
    if isinstance(symbols, Signal):
        sig = symbols
        if sig.resolved_symbols is None:
            raise ValueError(
                "resolved_symbols is not set. Call resolve_symbols(sig) or assign "
                "resolved_symbols before calling correct_phase_rotation()."
            )
        ref = ref_symbols if ref_symbols is not None else sig.source_symbols
        if ref is None:
            raise ValueError(
                "No reference available. Provide ref_symbols or ensure "
                "source_symbols is set on the Signal."
            )
        resolved_symbols, _ = unwrap_signal(sig, field="resolved_symbols")
        new = sig.copy()
        new.resolved_symbols = correct_phase_rotation(
            resolved_symbols, ref, num_skip_symbols=num_skip_symbols
        )
        return new

    if ref_symbols is None:
        raise ValueError(
            "correct_phase_rotation() requires ref_symbols for array input."
        )

    symbols, xp, _ = dispatch(symbols)
    symbols, was_1d = as_2d(symbols, name="symbols")
    C, N = symbols.shape

    ref = broadcast_channels(xp.asarray(ref_symbols), C, xp, name="ref_symbols")
    N_ref = ref.shape[-1]

    if num_skip_symbols >= N_ref:
        raise ValueError(
            f"num_skip_symbols={num_skip_symbols} must be less than the reference "
            f"length N_ref={N_ref}."
        )

    seg_y = symbols[:, num_skip_symbols:N_ref]  # (C, N_est)
    seg_r = ref[:, num_skip_symbols:]  # (C, N_est)
    thetas = -xp.angle(xp.sum(seg_y * xp.conj(seg_r), axis=-1))  # (C,) on device
    phasors = xp.exp(1j * thetas).astype(symbols.dtype)  # (C,) on device
    out = symbols * phasors[:, None]

    if logger.isEnabledFor(logging.INFO):
        # Host transfer of thetas is needed only for this per-channel log;
        # the correction itself (phasors) is computed on-device above.
        thetas_deg = np.degrees(to_device(thetas, "cpu"))
        for ch, deg in enumerate(thetas_deg.tolist()):
            logger.info("correct_phase_rotation: ch=%s, theta=%.2f°", ch, deg)

    return restore_1d(was_1d, out)
