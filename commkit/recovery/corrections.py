"""Phase corrections, cycle-slip repair, and ambiguity resolution."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._array import as_2d, broadcast_channels, restore_1d
from ..backend import ArrayType, dispatch, to_device
from ..core._signal_adapter import S, adapt_signal
from ..core.signal import Signal
from ..helpers import remove_linear_trend
from ..logger import logger
from ._common import _Context, _Phase


def _log_phase_summary(
    phi,
    prefix_fmt: str,
    prefix_args: tuple,
    suffix_fmt: str,
    suffix_args: tuple,
) -> None:
    r"""Log a CPR algorithm's "phase mean/std in degrees" INFO summary.

    Every CPR estimator (``bps``, ``viterbi_viterbi``, ``tikhonov``, ``pll``,
    ``pilots``' pilot-aided/pilot-tone estimators) ends with the same
    diagnostic: one host transfer of the phase trajectory (skipped entirely
    when INFO logging is disabled), then a mean/std reduction converted to
    degrees and logged as

        prefix_fmt + ": phase mean=%.2f°, std=%.2f° " + suffix_fmt

    with ``prefix_args``/``suffix_args`` supplying each caller's own
    parameters (algorithm name, block/pilot counts, channel count, ...).

    Parameters
    ----------
    phi : array_like
        Phase trajectory in radians, any shape/backend.
    prefix_fmt, suffix_fmt : str
        ``%``-style format strings, joined around the fixed
        ``phase mean=%.2f°, std=%.2f°`` core.
    prefix_args, suffix_args : tuple
        Positional args for ``prefix_fmt``/``suffix_fmt`` respectively.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    phi_np = to_device(phi, "cpu")
    mean_deg = float(np.mean(phi_np)) * 180.0 / np.pi
    std_deg = float(np.std(phi_np)) * 180.0 / np.pi
    fmt = prefix_fmt + ": phase mean=%.2f°, std=%.2f° " + suffix_fmt
    logger.info(fmt, *prefix_args, mean_deg, std_deg, *suffix_args)


# -----------------------------------------------------------------------------
# PHASE-TRACK UTILITIES (array-only)
# -----------------------------------------------------------------------------
# smooth_phase_wiener: Zero-phase Wiener smoother on a phase trajectory.
# correct_cycle_slips: Cycle-slip detection/correction on a phase trajectory.
#
# Both operate on a phase trajectory - a derived quantity, not raw IQ samples
# or any field a Signal carries - so neither is Signal-aware (AGENTS.md).


def smooth_phase_wiener(
    phase: ArrayType,
    *,
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
        Unwrapped phase track (e.g. the ``value`` of a ``PilotTone`` or
        ``PilotTones`` estimate), in radians.
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

    # Detrend per channel so the DC-divergent random-walk PSD does not distort
    # the residual-FOE ramp; restore the slope after filtering.  Only the
    # slope is removed (helpers.remove_linear_trend keeps the mean) - the
    # Wiener gain forces H[0] = 1.0 below, so a constant offset passes through
    # filtering unchanged regardless of whether it was present going in.
    xc = xp.arange(N, dtype=xp.float64) - 0.5 * (N - 1)
    if detrend:
        phi_c, slope = remove_linear_trend(phi)
    else:
        phi_c = phi
        slope = xp.zeros(C, dtype=xp.float64)

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
    phi_s = phi_s + slope[:, None] * xc[None, :]

    if logger.isEnabledFor(logging.INFO):
        # Two std reductions + host syncs, needed only for the line below.
        std_in = float(xp.std(phi_c))
        std_out = float(xp.std(phi_s - slope[:, None] * xc[None, :]))
        logger.info(
            "Wiener phase smoother: q=%.3g, r=%.3g rad², residual std %.2f° -> %.2f°.",
            q,
            r,
            np.degrees(std_in),
            np.degrees(std_out),
        )

    return restore_1d(was_1d, phi_s)


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
            *corrected* blocks.  When the deviation exceeds ``threshold``, it
            is corrected by a multiple of ``2π/symmetry``.

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
    phase: np.ndarray,
    *,
    symmetry: int = 4,
    history: int = 1000,
    threshold: float = np.pi / 4,
) -> np.ndarray:
    """
    Detects and corrects cycle slips in a phase trajectory.

    After ``unwrap`` resolves the M-fold ambiguity, residual cycle slips may
    remain where the unwrapper chose the wrong branch.  The trajectory is
    scanned sequentially: each value is predicted by a linear fit through up
    to ``history`` past corrected values (the previous value while fewer than
    ``min(10, history)`` are available).  A deviation beyond ``threshold`` is
    corrected by the nearest integer multiple of ``2π/symmetry``.

    Parameters
    ----------
    phase : (B,) array_like
        Phase trajectory in radians after the M-fold unwrap, e.g. block
        phases.
    symmetry : int, default 4
        Rotational symmetry of the constellation; the correction quantum is
        ``2π/symmetry`` (4 for square QAM and BPS, M for M-PSK, 1 for pilot
        phases).
    history : int, default 1000
        Past corrected values in the linear fit.  Reduce for short bursts.
    threshold : float, default π/4
        Deviation from the prediction that declares a slip; ``π/4`` is half
        the quantum for 4-fold symmetry.

    Returns
    -------
    (B,) float64 array
        Corrected trajectory, a new array on the input's device.

    Notes
    -----
    A sequential scan, Numba-compiled on the CPU; GPU input makes one host
    round trip.
    """
    phase, xp, _ = dispatch(phase)
    if phase.ndim != 1:
        raise ValueError(f"phase must be 1-D, got shape {phase.shape}.")
    host = np.array(to_device(phase, "cpu"), dtype=np.float64)  # a copy
    out = _get_numba_cycle_slip()(host, int(symmetry), int(history), float(threshold))
    return xp.asarray(out)


@dataclass(frozen=True)
class CycleSlip:
    """Cycle-slip repair of a phase trajectory, nested in a CPR method.

    Each value is predicted by a linear fit through up to ``history`` past
    repaired values; a deviation beyond ``threshold`` is snapped back by the
    nearest multiple of the method's ambiguity quantum (see
    :func:`correct_cycle_slips`).

    Parameters
    ----------
    history : int, default 100
        Past values in the linear fit.
    threshold : float, default π/4
        Deviation in radians that declares a slip.

    Examples
    --------
    >>> est = estimate_carrier_phase(y, BPS(cycle_slip=CycleSlip(history=50)))
    """

    history: int = 100
    threshold: float = np.pi / 4

    def __post_init__(self) -> None:
        if self.history < 1:
            raise ValueError(f"history must be >= 1, got {self.history}.")
        if not self.threshold > 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}.")


def _repair_slips(
    phase: ArrayType, xp: Any, cycle_slip: CycleSlip | None, symmetry: int
) -> ArrayType:
    """Row-wise cycle-slip repair of a ``(R, B)`` trajectory (host round trip)."""
    if cycle_slip is None:
        return phase
    out = xp.empty_like(phase)
    for r in range(phase.shape[0]):
        out[r] = correct_cycle_slips(
            phase[r],
            symmetry=symmetry,
            history=cycle_slip.history,
            threshold=cycle_slip.threshold,
        )
    return out


# -----------------------------------------------------------------------------
# DATA-AIDED STATIC ROTATION
# -----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DataAided:
    """Static per-channel rotation against known symbols.

    A rotationally invariant blind equalizer (CMA, RDE) leaves an arbitrary
    constant phase on each output channel, not limited to the discrete grid
    that :func:`resolve_phase_ambiguity` tests.  The ML estimate per channel
    is ``θ = ∠ Σ y·s*`` over the overlap of the samples and the known
    symbols; the trajectory is constant.

    Parameters
    ----------
    symbols : array_like, optional
        Known transmitted symbols, ``(N_ref,)`` (shared) or ``(C, N_ref)``.
        Only the first ``min(N, N_ref)`` symbols are used.  Defaults to the
        Signal's ``reference`` symbols.
    num_skip_symbols : int, default 0
        Leading symbols excluded from the estimate (an unconverged
        equalizer transient).

    Examples
    --------
    >>> y = correct_carrier_phase(sig, DataAided())  # sig.reference
    >>> y = correct_carrier_phase(y_arr, DataAided(symbols=tx, num_skip_symbols=500))
    """

    symbols: np.ndarray | None = None
    num_skip_symbols: int = 0

    def __post_init__(self) -> None:
        if self.num_skip_symbols < 0:
            raise ValueError(
                f"num_skip_symbols must be >= 0, got {self.num_skip_symbols}."
            )
        if self.symbols is not None:
            symbols = np.array(to_device(self.symbols, "cpu"))
            if symbols.ndim not in (1, 2):
                raise ValueError(
                    f"symbols must have shape (N,) or (C, N), got {symbols.shape}."
                )
            symbols.setflags(write=False)
            object.__setattr__(self, "symbols", symbols)


def _data_aided(x: ArrayType, method: DataAided, ctx: _Context) -> _Phase:
    if ctx.reference is None:
        raise ValueError(
            "estimate_carrier_phase(): DataAided needs known symbols (pass "
            "DataAided(symbols=...) or a Signal with a reference)."
        )
    x, xp, _ = dispatch(x)
    C, N = x.shape
    ref = broadcast_channels(xp.asarray(ctx.reference), C, xp, name="symbols")
    n = min(N, ref.shape[-1])
    skip = method.num_skip_symbols
    if skip >= n:
        raise ValueError(
            f"num_skip_symbols={skip} must be less than the number of known "
            f"symbols used ({n})."
        )
    corr = xp.sum(x[:, skip:n] * xp.conj(ref[:, skip:n]), axis=-1)  # (C,)
    theta = xp.angle(corr).astype(xp.float64)
    if logger.isEnabledFor(logging.INFO):
        for ch, deg in enumerate(np.degrees(to_device(theta, "cpu")).tolist()):
            logger.info("CPR (data-aided): ch=%s, theta=%.2f°", ch, deg)
    return _Phase(phase=xp.broadcast_to(theta[:, None], (C, N)).copy())


# -----------------------------------------------------------------------------
# AMBIGUITY RESOLUTION (Signal-aware)
# -----------------------------------------------------------------------------
# resolve_channel_permutation: Fix a MIMO polarization/channel-order swap.
# resolve_phase_ambiguity:     Fix a rotational (k·2π/symmetry_order) ambiguity.
# Both correct a Signal's samples against reference.symbols.


def _reference_symbols(signal_adapter: Any, reference: Any, name: str) -> Any:
    """The explicit reference, else the Signal's ``reference.symbols``."""
    if reference is not None:
        return reference
    sig = signal_adapter.signal
    if sig is None or sig.reference is None:
        raise ValueError(
            f"{name} needs reference symbols: pass reference= or a Signal that "
            "has a reference."
        )
    return sig.reference.symbols


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
    symbols: S,
    reference: ArrayType | None = None,
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
    magnitude ``|Σ yᵢ · conj(sⱼ)|``) and reorders ``symbols`` to the
    reference order.

    Run this **before** ``resolve_phase_ambiguity`` (it is rotation
    invariant, so the two compose) and before SER/BER.  For a converged
    *data-aided* equalizer the outputs are already pinned to the training order,
    so this is a no-op; it is the robust fix for **blind** equalizers, whose
    output order is arbitrary.  Only a *constant* permutation is resolved - a
    mid-stream swap is an equalizer-tracking issue, not a labeling one.

    Parameters
    ----------
    symbols : array_like or Signal
        Recovered symbols, ``(N,)`` or ``(C, N)``.  Returned unchanged for SISO.
        A Signal must be at one sample per symbol; its ``reference.symbols``
        is the default reference.
    reference : array_like, optional
        Known transmitted symbols, same layout as ``symbols`` (the full
        sequence, a pilot subset, or any known reference).  Required for
        arrays.
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
    array_like or Signal
        ``symbols`` with channels reordered to the reference order; same
        shape, dtype, and backend.  A Signal gives a new Signal with the
        samples reordered (its reference is already in that order).
    """
    name = "resolve_channel_permutation()"
    signal_adapter = adapt_signal(symbols, function_name=name)
    x = signal_adapter.symbol_array()
    resolved = _resolve_channel_permutation_array(
        x,
        _reference_symbols(signal_adapter, reference, name),
        num_skip_symbols=num_skip_symbols,
        metric=metric,
    )
    return signal_adapter.wrap_samples(resolved)


def _resolve_channel_permutation_array(
    symbols: ArrayType,
    ref_symbols: ArrayType,
    *,
    num_skip_symbols: int = 0,
    metric: str = "coherence",
) -> ArrayType:
    """Array-only channel assignment implementation."""

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
    symbols: S,
    reference: ArrayType | None = None,
    *,
    constellation: Any = None,
    symmetry: int | None = None,
    num_skip_symbols: int = 0,
) -> ArrayType | Signal:
    """
    Resolves the rotational phase ambiguity left by blind carrier recovery.

    Blind CPR (Viterbi-Viterbi, BPS, Tikhonov, PLL) cannot tell the
    ``symmetry`` rotated copies of the constellation apart.  The ML choice
    of rotation ``k·2π/symmetry`` is the one closest to ``-∠ Σ y·s*`` against
    the known symbols, and the symbols are returned rotated by it.

    For MIMO inputs each channel is resolved independently - after MIMO
    equalisation the output streams may land on different ambiguity branches.

    Parameters
    ----------
    symbols : array_like or Signal
        Received symbols after carrier phase correction, ``(N,)`` or
        ``(C, N)``.  A Signal must be at one sample per symbol; its
        ``reference.symbols`` is the default reference.
    reference : array_like, optional
        Known transmitted symbols, ``(N,)`` or ``(C, N)``.  Required for
        arrays.
    constellation : Constellation, optional
        Its ``rotational_symmetry`` is the default ``symmetry``; with it the
        log reports the symbol error rate of the choice.  Defaults to the
        Signal's ``constellation``.
    symmetry : int, optional
        Number of rotations tested; overrides the constellation's.  One of
        the two is required.
    num_skip_symbols : int, default 0
        Leading symbols excluded from the estimate (an unconverged
        transient); the rotation still covers the full input.  Must be less
        than the symbol count.

    Returns
    -------
    array_like or Signal
        Rotated symbols, same shape and dtype; a Signal gives a new Signal.
    """
    name = "resolve_phase_ambiguity()"
    signal_adapter = adapt_signal(symbols, function_name=name)
    x = signal_adapter.symbol_array()
    constellation = signal_adapter.resolve_choice("constellation", constellation)
    if symmetry is None and constellation is None:
        raise ValueError(f"{name} requires constellation or symmetry.")
    resolved = _resolve_phase_ambiguity_array(
        x,
        _reference_symbols(signal_adapter, reference, name),
        constellation,
        symmetry=symmetry,
        num_skip_symbols=num_skip_symbols,
    )
    return signal_adapter.wrap_samples(resolved)


def _resolve_phase_ambiguity_array(
    symbols: ArrayType,
    ref_symbols: ArrayType,
    constellation: Any,
    *,
    symmetry: int | None,
    num_skip_symbols: int = 0,
) -> ArrayType:
    """Array-only rotational-ambiguity resolution."""
    symbols, xp, _ = dispatch(symbols)
    symbols, was_1d = as_2d(symbols, name="symbols")
    C, N = symbols.shape

    if num_skip_symbols >= N:
        raise ValueError(
            f"num_skip_symbols={num_skip_symbols} must be less than the total "
            f"symbol count N={N}."
        )

    ref = broadcast_channels(xp.asarray(ref_symbols), C, xp, name="ref_symbols")

    if symmetry is None:
        symmetry = int(constellation.rotational_symmetry)

    step = 2.0 * np.pi / symmetry

    # ML phase ambiguity estimator: the optimal rotation maximises
    # Re(e^{jkθ} · Σ y_n s_n*), which equals choosing k closest to
    # -∠(Σ y_n s_n*) / step.  Single inner product replaces symmetry
    # full SER passes.  All channels batched: one D2H of the (C,) angles
    # instead of one float() sync per channel.
    seg_y = symbols[:, num_skip_symbols:]
    seg_r = ref[:, num_skip_symbols:]
    corr = xp.sum(seg_y * xp.conj(seg_r), axis=-1)  # (C,)
    theta_np = -to_device(xp.angle(corr), "cpu")  # (C,) float64, one transfer
    best_k_np = np.round(theta_np / step).astype(np.int64) % symmetry
    phasors = xp.asarray(
        np.exp(1j * best_k_np * step).astype(symbols.dtype)
    )  # (C,) - built on host from host indices, single H2D
    out = symbols * phasors[:, None]

    # SER is diagnostic-only: skip the decisions entirely when INFO logging is
    # disabled (or there is no constellation to decide on).
    if constellation is not None and logger.isEnabledFor(logging.INFO):
        k = constellation.bits_per_symbol
        n = seg_r.shape[-1]
        bits_out = constellation.demap(out[:, num_skip_symbols:]).reshape(C, n, k)
        bits_ref = constellation.demap(seg_r).reshape(C, n, k)
        ser_np = to_device(
            xp.mean(xp.any(bits_out != bits_ref, axis=-1), axis=-1), "cpu"
        )
        for ch in range(C):
            logger.info(
                "Phase ambiguity resolution: ch=%s, best_k=%s, "
                "rotation=%.1f°, SER=%.4f",
                ch,
                int(best_k_np[ch]),
                best_k_np[ch] * step * 180.0 / np.pi,
                float(ser_np[ch]),
            )

    return restore_1d(was_1d, out)
