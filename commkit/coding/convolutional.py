"""Convolutional codes: :class:`ConvolutionalCode`, :class:`Viterbi` and :class:`BCJR`.

A rate ``1/r`` feed-forward convolutional code of constraint length ``K``
given by ``r`` generator polynomials in octal (``(0o133, 0o171)`` for the
``K = 7`` code of IEEE 802.11 and CCSDS): the most significant bit of each
generator taps the current input, and the ``r`` outputs of each step are
transmitted together.  A block of ``k`` information bits is either
terminated with ``K - 1`` zero tail bits (``n = r (k + K - 1)``) or
tail-biting (the encoder starts in the state of its last ``K - 1`` bits,
``n = r k``).  Encoding is ``r`` XORs of shifted copies on the data's
device.

Decoders: :class:`Viterbi` (maximum likelihood; exact for tail-biting
blocks, by one pass per start state) and :class:`BCJR` (a-posteriori LLRs,
max-log or exact).  Both are sequential per block: Numba over blocks on
the CPU, and GPU input makes one round trip.  The trellis tables and the
forward-backward recursion are shared with the recursive systematic
constituents of :class:`~commkit.coding.TurboCode`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from .base import _Code, _Codec, _Decoded, _register

__all__ = ["BCJR", "ConvolutionalCode", "Viterbi"]

_NUMBA_TRELLIS: dict[str, Any] = {}


class _Trellis(NamedTuple):
    """State-transition tables of a binary trellis (private).

    ``next[s, u]`` the state after input ``u`` in state ``s`` (``s`` holds
    the last ``K - 1`` register bits, the newest as its MSB); ``outputs[s,
    u]`` the ``r`` output bits; ``tail[s]`` the input that drives the
    register towards zero (0 for feed-forward codes, the feedback bit for
    recursive ones).
    """

    next: np.ndarray
    outputs: np.ndarray
    tail: np.ndarray


def _parity(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    while np.any(x):
        out ^= x & 1
        x = x >> 1
    return out


def _trellis(
    generators: tuple[int, ...],
    k: int,
    feedback: int | None = None,
    systematic: bool = False,
) -> _Trellis:
    """Tables of a feed-forward or recursive (``feedback``) code of constraint ``k``.

    Generators and feedback are ``k``-bit masks, MSB the current input.  A
    systematic code outputs the input bit first, then the generators.
    """
    states = 1 << (k - 1)
    s = np.arange(states)[:, None]
    u = np.arange(2)[None, :]
    if feedback is None:
        a = np.broadcast_to(u, (states, 2))
    else:
        a = u ^ _parity(s & (feedback & (states - 1)))
    register = (a << (k - 1)) | s
    outs = [_parity(register & g) for g in generators]
    if systematic:
        outs = [np.broadcast_to(u, (states, 2)), *outs]
    nxt = (a << (k - 2)) | (s >> 1)
    tail = (
        np.zeros(states, dtype=np.int64)
        if feedback is None
        else _parity(np.arange(states) & (feedback & (states - 1)))
    )
    return _Trellis(
        nxt.astype(np.int64),
        np.stack(outs, axis=-1).astype(np.int8),
        tail.astype(np.int64),
    )


@dataclass(frozen=True, eq=False)
class ConvolutionalCode(_Code):
    """Feed-forward convolutional code of rate ``1 / len(generators)``.

    Parameters
    ----------
    generators : tuple of int
        Generator polynomials, ``constraint_length`` bits each, usually
        written in octal; the MSB taps the current input.
    constraint_length : int
        ``K``, register length including the current input, 2 to 16.
    block_length : int
        Information bits per block.
    termination : {"zero", "tail-biting"}, default "zero"
        ``"zero"`` appends ``K - 1`` zero bits that return the encoder to
        the zero state; ``"tail-biting"`` starts it in the state of the
        block's last ``K - 1`` bits, so it ends there too.
    """

    generators: tuple[int, ...]
    constraint_length: int
    block_length: int
    termination: str = "zero"

    def __post_init__(self) -> None:
        k = self.constraint_length
        if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= 16:
            raise ValueError(
                f"constraint_length must be an integer in [2, 16], got {k!r}."
            )
        gens = tuple(int(g) for g in self.generators)
        if len(gens) < 2 or any(not 0 < g < (1 << k) for g in gens):
            raise ValueError(
                f"need two or more generators of at most {k} bits, nonzero."
            )
        object.__setattr__(self, "generators", gens)
        b = self.block_length
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValueError(f"block_length must be an integer >= 1, got {b!r}.")
        if self.termination not in ("zero", "tail-biting"):
            raise ValueError(
                f"termination must be 'zero' or 'tail-biting', got {self.termination!r}."
            )
        if self.termination == "tail-biting" and b < k - 1:
            raise ValueError("a tail-biting block needs at least K - 1 bits.")

    @property
    def n(self) -> int:
        """Transmitted bits per block."""
        steps = self.block_length + (self.constraint_length - 1 if self._zero else 0)
        return steps * len(self.generators)

    @property
    def k(self) -> int:
        """Information bits per block."""
        return self.block_length

    @property
    def _zero(self) -> bool:
        return self.termination == "zero"

    def __repr__(self) -> str:
        gens = ", ".join(f"0o{g:o}" for g in self.generators)
        return f"ConvolutionalCode(({gens}), K={self.constraint_length}, k={self.k})"

    @classmethod
    def k7(cls, block_length: int, *, termination: str = "zero") -> ConvolutionalCode:
        """The rate-1/2, ``K = 7`` code with generators 133 and 171 (octal).

        The industry-standard code of IEEE 802.11 (clause 17.3.5.6), CCSDS
        131.0-B and many others.

        Parameters
        ----------
        block_length : int
            Information bits per block.
        termination : {"zero", "tail-biting"}, default "zero"
            As for :class:`ConvolutionalCode`.

        Returns
        -------
        ConvolutionalCode
        """
        return cls((0o133, 0o171), 7, block_length, termination)

    @cached_property
    def _trellis(self) -> _Trellis:
        return _trellis(self.generators, self.constraint_length)


@dataclass(frozen=True)
class Viterbi:
    """Maximum-likelihood sequence decoding (soft input, hard output)."""


@dataclass(frozen=True)
class BCJR:
    """A-posteriori decoding by the forward-backward (BCJR) algorithm.

    Parameters
    ----------
    exact : bool, default False
        Log-MAP (exact, log-sum-exp) instead of max-log-MAP.
    """

    exact: bool = False


def _encode(code: ConvolutionalCode, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    rows, k = bits.shape
    m = code.constraint_length - 1
    if code._zero:
        head = xp.zeros((rows, m), dtype=np.int8)
        tail = xp.zeros((rows, m), dtype=np.int8)
    else:
        head = bits[:, k - m :] if m else xp.zeros((rows, 0), dtype=np.int8)
        tail = xp.zeros((rows, 0), dtype=np.int8)
    stream = xp.concatenate([head, bits, tail], axis=1)  # u_(t-m) ... for each step
    steps = stream.shape[1] - m
    outs = []
    for g in code.generators:
        out = xp.zeros((rows, steps), dtype=np.int8)
        for d in range(m + 1):
            if (g >> (m - d)) & 1:
                out ^= stream[:, m - d : m - d + steps]
        outs.append(out)
    return xp.stack(outs, axis=2).reshape(rows, -1)


def _get_numba_trellis(name: str) -> Any:
    """Numba Viterbi and BCJR over a trellis's tables, parallel over blocks.

    ``viterbi`` maximizes ``sum (1 - 2 c) L`` over paths from ``start`` to
    ``end`` (``-1`` for any), once per start state when ``tail_biting``
    (start and end equal).  ``bcjr`` is shared with the turbo decoder: per
    step a channel LLR for each output and an a-priori LLR for the input,
    ``forced[t]`` steps (termination tails) taking the tail input only;
    it returns the a-posteriori LLRs of the inputs and of the outputs.
    """
    if not _NUMBA_TRELLIS:
        import numba

        neg = -1.0e300

        @numba.njit(cache=True, nogil=True)
        def branch(llr_t, outputs, s, u):  # type: ignore[no-untyped-def]
            acc = 0.0
            for j in range(outputs.shape[2]):
                v = llr_t[j]
                acc += -v if outputs[s, u, j] else v
            return acc

        @numba.njit(cache=True, nogil=True)
        def combine(a, b, exact):  # type: ignore[no-untyped-def]
            """``log(e^a + e^b)`` (exact) or ``max(a, b)``."""
            if a < b:
                a, b = b, a
            if b <= neg or not exact:
                return a
            return a + np.log1p(np.exp(b - a))

        @numba.njit(cache=True, nogil=True)
        def viterbi_one(llr, nxt, outputs, start, end, decisions, metric, nm):  # type: ignore[no-untyped-def]
            steps = llr.shape[0]
            states = nxt.shape[0]
            for s in range(states):
                metric[s] = 0.0 if (start < 0 or s == start) else neg
            for t in range(steps):
                for s in range(states):
                    nm[s] = neg
                for s in range(states):
                    if metric[s] <= neg:
                        continue
                    for u in range(2):
                        q = nxt[s, u]
                        v = metric[s] + branch(llr[t], outputs, s, u)
                        if v > nm[q]:
                            nm[q] = v
                            decisions[t, q] = s * 2 + u
                for s in range(states):
                    metric[s] = nm[s]
            best = end
            if end < 0:
                best = 0
                for s in range(1, states):
                    if metric[s] > metric[best]:
                        best = s
            return best, metric[best]

        @numba.njit(cache=True, nogil=True, parallel=True)
        def viterbi(llrs, nxt, outputs, tail_biting, info_steps, out_bits):  # type: ignore[no-untyped-def]
            rows, steps = llrs.shape[0], llrs.shape[1]
            states = nxt.shape[0]
            for row in numba.prange(rows):  # type: ignore[attr-defined,no-untyped-call]
                decisions = np.zeros((steps, states), dtype=np.int64)
                best_dec = np.zeros((steps, states), dtype=np.int64)
                metric = np.empty(states)
                nm = np.empty(states)
                best_end = 0
                best_val = neg
                for start in range(states if tail_biting else 1):
                    end = start if tail_biting else 0
                    last, val = viterbi_one(
                        llrs[row], nxt, outputs, start, end, decisions, metric, nm
                    )
                    if val > best_val:
                        best_val = val
                        best_end = last
                        best_dec[:, :] = decisions
                s = best_end
                for t in range(steps - 1, -1, -1):
                    code = best_dec[t, s]
                    if t < info_steps:
                        out_bits[row, t] = code & 1
                    s = code >> 1

        @numba.njit(cache=True, nogil=True)
        def bcjr(
            llr, apriori, nxt, outputs, tail, forced, start, end, exact, app_u, app_c
        ):  # type: ignore[no-untyped-def]
            steps = llr.shape[0]
            states = nxt.shape[0]
            r = outputs.shape[2]
            alpha = np.full((steps + 1, states), neg)
            beta = np.full((steps + 1, states), neg)
            if start < 0:
                alpha[0, :] = 0.0
            else:
                alpha[0, start] = 0.0
            if end < 0:
                beta[steps, :] = 0.0
            else:
                beta[steps, end] = 0.0
            gamma = np.full((steps, states, 2), neg)
            for t in range(steps):
                for s in range(states):
                    for u in range(2):
                        if forced[t] and u != tail[s]:
                            continue
                        g = 0.5 * branch(llr[t], outputs, s, u)
                        g += -0.5 * apriori[t] if u else 0.5 * apriori[t]
                        gamma[t, s, u] = g

            for t in range(steps):
                for s in range(states):
                    if alpha[t, s] <= neg:
                        continue
                    for u in range(2):
                        if gamma[t, s, u] <= neg:
                            continue
                        q = nxt[s, u]
                        alpha[t + 1, q] = combine(
                            alpha[t + 1, q], alpha[t, s] + gamma[t, s, u], exact
                        )
            for t in range(steps - 1, -1, -1):
                for s in range(states):
                    acc = neg
                    for u in range(2):
                        if gamma[t, s, u] <= neg:
                            continue
                        acc = combine(
                            acc, gamma[t, s, u] + beta[t + 1, nxt[s, u]], exact
                        )
                    beta[t, s] = acc
            for t in range(steps):
                zu = neg
                ou = neg
                for j in range(r):
                    app_c[t, j] = 0.0
                c0 = np.full(r, neg)
                c1 = np.full(r, neg)
                for s in range(states):
                    for u in range(2):
                        if gamma[t, s, u] <= neg or alpha[t, s] <= neg:
                            continue
                        v = alpha[t, s] + gamma[t, s, u] + beta[t + 1, nxt[s, u]]
                        if u:
                            ou = combine(ou, v, exact)
                        else:
                            zu = combine(zu, v, exact)
                        for j in range(r):
                            if outputs[s, u, j]:
                                c1[j] = combine(c1[j], v, exact)
                            else:
                                c0[j] = combine(c0[j], v, exact)
                app_u[t] = zu - ou
                for j in range(r):
                    app_c[t, j] = c0[j] - c1[j]

        @numba.njit(cache=True, nogil=True, parallel=True)
        def bcjr_blocks(
            llrs, nxt, outputs, tail, forced, start, end, exact, app_u, app_c
        ):  # type: ignore[no-untyped-def]
            apriori = np.zeros(llrs.shape[1])
            for row in numba.prange(llrs.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
                bcjr(llrs[row], apriori, nxt, outputs, tail, forced, start, end, exact,
                     app_u[row], app_c[row])  # fmt: skip

        _NUMBA_TRELLIS["viterbi"] = viterbi
        _NUMBA_TRELLIS["bcjr"] = bcjr
        _NUMBA_TRELLIS["bcjr_blocks"] = bcjr_blocks
    return _NUMBA_TRELLIS[name]


def _decode(
    code: ConvolutionalCode, llrs: ArrayType, method: Viterbi | BCJR
) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    r = len(code.generators)
    host = np.ascontiguousarray(to_device(llrs, "cpu"), dtype=np.float64)
    rows = host.shape[0]
    steps = host.shape[1] // r
    trellis = code._trellis
    host = host.reshape(rows, steps, r)
    tail_biting = not code._zero
    if isinstance(method, Viterbi):
        bits = np.empty((rows, code.k), dtype=np.int8)
        _get_numba_trellis("viterbi")(
            host, trellis.next, trellis.outputs, tail_biting, code.k, bits
        )
        soft = None
    else:
        if tail_biting:
            raise ValueError(
                "BCJR decodes terminated codes; use Viterbi for tail-biting."
            )
        app_u = np.empty((rows, steps))
        app_c = np.empty((rows, steps, r))
        forced = np.zeros(steps, dtype=np.bool_)
        forced[code.k :] = True
        _get_numba_trellis("bcjr_blocks")(
            host, trellis.next, trellis.outputs, trellis.tail, forced, 0, 0, method.exact,
            app_u, app_c,
        )  # fmt: skip
        bits = (app_u[:, : code.k] < 0).astype(np.int8)
        soft = app_c.reshape(rows, -1).astype(np.float32)
    device = "cpu" if xp is np else "gpu"
    return _Decoded(
        to_device(bits, device),
        None if soft is None else to_device(soft, device),
        xp.ones(rows, dtype=bool),
        xp.ones(rows, dtype=np.int32),
    )


_register(
    ConvolutionalCode,
    _Codec(
        encode=_encode,
        decode=_decode,
        default_method=lambda code: Viterbi(),
        methods=(Viterbi, BCJR),
    ),
)
