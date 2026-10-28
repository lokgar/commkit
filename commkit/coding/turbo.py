"""Turbo codes: :class:`TurboCode` (LTE) and :class:`TurboIterations`.

The rate-1/3 turbo code of LTE (3GPP TS 36.212 clause 5.1.3.2): two
recursive systematic convolutional encoders with transfer function
``[1, g1(D) / g0(D)]``, ``g0 = 1 + D^2 + D^3`` (feedback) and ``g1 = 1 + D +
D^3``, the second encoding the block permuted by the quadratic permutation
polynomial interleaver ``Pi(i) = (f1 i + f2 i^2) mod K`` of Table 5.1.3-3,
each terminated by three tail bits.  The transmitted streams ``d0``
(systematic), ``d1`` and ``d2`` (parities) carry ``K + 4`` bits each, the
twelve tail bits distributed as clause 5.1.3.2.2 says; they are
transmitted one after the other, ``n = 3 K + 12`` (no rate matching).

:class:`TurboIterations` decodes iteratively with two BCJR decoders
exchanging extrinsic LLRs, the forward-backward recursion of
:mod:`commkit.coding.convolutional`.  Encoding and decoding are sequential
per block: Numba over blocks on the CPU, and GPU input makes one round
trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from .base import _Code, _Codec, _Decoded, _register
from .convolutional import _get_numba_trellis, _Trellis, _trellis

__all__ = ["TurboCode", "TurboIterations"]

# TS 36.212 Table 5.1.3-3: (K, f1, f2) of the QPP interleaver.
# fmt: off
_QPP = (
    (40, 3, 10), (48, 7, 12), (56, 19, 42), (64, 7, 16), (72, 7, 18), (80, 11, 20),
    (88, 5, 22), (96, 11, 24), (104, 7, 26), (112, 41, 84), (120, 103, 90), (128, 15, 32),
    (136, 9, 34), (144, 17, 108), (152, 9, 38), (160, 21, 120), (168, 101, 84),
    (176, 21, 44), (184, 57, 46), (192, 23, 48), (200, 13, 50), (208, 27, 52),
    (216, 11, 36), (224, 27, 56), (232, 85, 58), (240, 29, 60), (248, 33, 62),
    (256, 15, 32), (264, 17, 198), (272, 33, 68), (280, 103, 210), (288, 19, 36),
    (296, 19, 74), (304, 37, 76), (312, 19, 78), (320, 21, 120), (328, 21, 82),
    (336, 115, 84), (344, 193, 86), (352, 21, 44), (360, 133, 90), (368, 81, 46),
    (376, 45, 94), (384, 23, 48), (392, 243, 98), (400, 151, 40), (408, 155, 102),
    (416, 25, 52), (424, 51, 106), (432, 47, 72), (440, 91, 110), (448, 29, 168),
    (456, 29, 114), (464, 247, 58), (472, 29, 118), (480, 89, 180), (488, 91, 122),
    (496, 157, 62), (504, 55, 84), (512, 31, 64), (528, 17, 66), (544, 35, 68),
    (560, 227, 420), (576, 65, 96), (592, 19, 74), (608, 37, 76), (624, 41, 234),
    (640, 39, 80), (656, 185, 82), (672, 43, 252), (688, 21, 86), (704, 155, 44),
    (720, 79, 120), (736, 139, 92), (752, 23, 94), (768, 217, 48), (784, 25, 98),
    (800, 17, 80), (816, 127, 102), (832, 25, 52), (848, 239, 106), (864, 17, 48),
    (880, 137, 110), (896, 215, 112), (912, 29, 114), (928, 15, 58), (944, 147, 118),
    (960, 29, 60), (976, 59, 122), (992, 65, 124), (1008, 55, 84), (1024, 31, 64),
    (1056, 17, 66), (1088, 171, 204), (1120, 67, 140), (1152, 35, 72), (1184, 19, 74),
    (1216, 39, 76), (1248, 19, 78), (1280, 199, 240), (1312, 21, 82), (1344, 211, 252),
    (1376, 21, 86), (1408, 43, 88), (1440, 149, 60), (1472, 45, 92), (1504, 49, 846),
    (1536, 71, 48), (1568, 13, 28), (1600, 17, 80), (1632, 25, 102), (1664, 183, 104),
    (1696, 55, 954), (1728, 127, 96), (1760, 27, 110), (1792, 29, 112), (1824, 29, 114),
    (1856, 57, 116), (1888, 45, 354), (1920, 31, 120), (1952, 59, 610), (1984, 185, 124),
    (2016, 113, 420), (2048, 31, 64), (2112, 17, 66), (2176, 171, 136), (2240, 209, 420),
    (2304, 253, 216), (2368, 367, 444), (2432, 265, 456), (2496, 181, 468), (2560, 39, 80),
    (2624, 27, 164), (2688, 127, 504), (2752, 143, 172), (2816, 43, 88), (2880, 29, 300),
    (2944, 45, 92), (3008, 157, 188), (3072, 47, 96), (3136, 13, 28), (3200, 111, 240),
    (3264, 443, 204), (3328, 51, 104), (3392, 51, 212), (3456, 451, 192), (3520, 257, 220),
    (3584, 57, 336), (3648, 313, 228), (3712, 271, 232), (3776, 179, 236),
    (3840, 331, 120), (3904, 363, 244), (3968, 375, 248), (4032, 127, 168), (4096, 31, 64),
    (4160, 33, 130), (4224, 43, 264), (4288, 33, 134), (4352, 477, 408), (4416, 35, 138),
    (4480, 233, 280), (4544, 357, 142), (4608, 337, 480), (4672, 37, 146), (4736, 71, 444),
    (4800, 71, 120), (4864, 37, 152), (4928, 39, 462), (4992, 127, 234), (5056, 39, 158),
    (5120, 39, 80), (5184, 31, 96), (5248, 113, 902), (5312, 41, 166), (5376, 251, 336),
    (5440, 43, 170), (5504, 21, 86), (5568, 43, 174), (5632, 45, 176), (5696, 45, 178),
    (5760, 161, 120), (5824, 89, 182), (5888, 323, 184), (5952, 47, 186), (6016, 23, 94),
    (6080, 47, 190), (6144, 263, 480),
)
# fmt: on

_NUMBA_TURBO: dict[str, Any] = {}


@dataclass(frozen=True, eq=False)
class TurboCode(_Code):
    """The LTE turbo code for one of the 188 block sizes of TS 36.212.

    Parameters
    ----------
    block_length : int
        ``K``, information bits per block: a size of Table 5.1.3-3 (40 to
        6144).

    Notes
    -----
    ``n = 3 K + 12``: the streams ``d0, d1, d2`` of ``K + 4`` bits each,
    one after the other.
    """

    block_length: int

    def __post_init__(self) -> None:
        if self.block_length not in _QPP_PARAMETERS:
            raise ValueError(
                f"block_length must be a K of TS 36.212 Table 5.1.3-3 (40, 48, ..., 6144), "
                f"got {self.block_length!r}."
            )

    @property
    def n(self) -> int:
        """Transmitted bits per block, ``3 K + 12``."""
        return 3 * self.block_length + 12

    @property
    def k(self) -> int:
        """Information bits per block, ``K``."""
        return self.block_length

    def __repr__(self) -> str:
        return f"TurboCode.lte({self.block_length})"

    @classmethod
    def lte(cls, block_length: int) -> TurboCode:
        """The LTE turbo code (TS 36.212 clause 5.1.3.2) of block size ``K``.

        Parameters
        ----------
        block_length : int
            ``K``, one of the 188 block sizes of TS 36.212 Table 5.1.3-3
            (40 to 6144).

        Returns
        -------
        TurboCode
        """
        return cls(block_length)

    @cached_property
    def interleaver(self) -> np.ndarray:
        """``Pi``, ``(K,)``: the second encoder's input bit ``i`` is ``c[Pi(i)]``."""
        f1, f2 = _QPP_PARAMETERS[self.block_length]
        i = np.arange(self.block_length, dtype=np.int64)
        pi = (f1 * i + f2 * i * i) % self.block_length
        pi.setflags(write=False)
        return pi


_QPP_PARAMETERS = {k: (f1, f2) for k, f1, f2 in _QPP}
# 1 + D^2 + D^3 (feedback) and 1 + D + D^3, MSB the current input.
_RSC: _Trellis = _trellis((0b1101,), 4, feedback=0b1011, systematic=True)


@dataclass(frozen=True)
class TurboIterations:
    """Iterative turbo decoding with two BCJR decoders.

    Parameters
    ----------
    iterations : int, default 8
        Full iterations (both decoders).
    exact : bool, default False
        Log-MAP constituents instead of max-log-MAP.
    scale : float, optional
        Factor on the exchanged extrinsic LLRs; ``None`` takes 0.7 for
        max-log-MAP (which overestimates them) and 1 for log-MAP.
    """

    iterations: int = 8
    exact: bool = False
    scale: float | None = None

    def __post_init__(self) -> None:
        it = self.iterations
        if isinstance(it, bool) or not isinstance(it, int) or it < 1:
            raise ValueError(f"iterations must be an integer >= 1, got {it!r}.")
        if self.scale is not None and not 0 < self.scale <= 1:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}.")


def _get_numba_turbo(name: str) -> Any:
    """Numba RSC encoder and turbo decoder, parallel over blocks."""
    if name not in _NUMBA_TURBO:
        import numba

        bcjr = _get_numba_trellis("bcjr")

        @numba.njit(cache=True, nogil=True)
        def rsc(bits, nxt, outputs, tail, parity, tail_bits):  # type: ignore[no-untyped-def]
            """Parity of ``bits`` and the 3 tail (systematic, parity) pairs."""
            s = 0
            for t in range(bits.shape[0]):
                u = bits[t]
                parity[t] = outputs[s, u, 1]
                s = nxt[s, u]
            for t in range(3):
                u = tail[s]
                tail_bits[t, 0] = u
                tail_bits[t, 1] = outputs[s, u, 1]
                s = nxt[s, u]

        @numba.njit(cache=True, nogil=True, parallel=True)
        def encode(bits, pi, nxt, outputs, tail, out):  # type: ignore[no-untyped-def]
            k = bits.shape[1]
            for row in numba.prange(bits.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
                p1 = np.empty(k, dtype=np.int8)
                p2 = np.empty(k, dtype=np.int8)
                t1 = np.empty((3, 2), dtype=np.int8)
                t2 = np.empty((3, 2), dtype=np.int8)
                permuted = np.empty(k, dtype=np.int8)
                for i in range(k):
                    permuted[i] = bits[row, pi[i]]
                rsc(bits[row], nxt, outputs, tail, p1, t1)
                rsc(permuted, nxt, outputs, tail, p2, t2)
                d0 = out[row, : k + 4]
                d1 = out[row, k + 4 : 2 * k + 8]
                d2 = out[row, 2 * k + 8 :]
                d0[:k] = bits[row]
                d1[:k] = p1
                d2[:k] = p2
                # TS 36.212 clause 5.1.3.2.2.
                d0[k], d1[k], d2[k] = t1[0, 0], t1[0, 1], t1[1, 0]
                d0[k + 1], d1[k + 1], d2[k + 1] = t1[1, 1], t1[2, 0], t1[2, 1]
                d0[k + 2], d1[k + 2], d2[k + 2] = t2[0, 0], t2[0, 1], t2[1, 0]
                d0[k + 3], d1[k + 3], d2[k + 3] = t2[1, 1], t2[2, 0], t2[2, 1]

        @numba.njit(cache=True, nogil=True, parallel=True)
        def decode(llrs, pi, nxt, outputs, tail, iterations, exact, scale, out):  # type: ignore[no-untyped-def]
            k = pi.shape[0]
            steps = k + 3
            forced = np.zeros(steps, dtype=np.bool_)
            forced[k:] = True
            for row in numba.prange(llrs.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
                d0 = llrs[row, : k + 4]
                d1 = llrs[row, k + 4 : 2 * k + 8]
                d2 = llrs[row, 2 * k + 8 :]
                ch1 = np.empty((steps, 2))
                ch2 = np.empty((steps, 2))
                for t in range(k):
                    ch1[t, 0] = d0[t]
                    ch1[t, 1] = d1[t]
                    ch2[t, 0] = d0[pi[t]]
                    ch2[t, 1] = d2[t]
                ch1[k, 0], ch1[k, 1], ch1[k + 1, 0] = d0[k], d1[k], d2[k]
                ch1[k + 1, 1], ch1[k + 2, 0], ch1[k + 2, 1] = (
                    d0[k + 1],
                    d1[k + 1],
                    d2[k + 1],
                )
                ch2[k, 0], ch2[k, 1], ch2[k + 1, 0] = d0[k + 2], d1[k + 2], d2[k + 2]
                ch2[k + 1, 1], ch2[k + 2, 0], ch2[k + 2, 1] = (
                    d0[k + 3],
                    d1[k + 3],
                    d2[k + 3],
                )
                la1 = np.zeros(steps)
                la2 = np.zeros(steps)
                app1 = np.empty(steps)
                app2 = np.empty(steps)
                app_c = np.empty((steps, 2))
                for _ in range(iterations):
                    bcjr(ch1, la1, nxt, outputs, tail, forced, 0, 0, exact, app1, app_c)
                    for t in range(k):
                        # Extrinsic of decoder 1, kept in la1 (natural order).
                        la1[t] = scale * (app1[t] - la1[t] - ch1[t, 0])
                    for t in range(k):
                        la2[t] = la1[pi[t]]
                    bcjr(ch2, la2, nxt, outputs, tail, forced, 0, 0, exact, app2, app_c)
                    for t in range(k):
                        la1[pi[t]] = scale * (app2[t] - la2[t] - ch2[t, 0])
                for t in range(k):
                    out[row, pi[t]] = 1 if app2[t] < 0 else 0

        _NUMBA_TURBO["encode"] = encode
        _NUMBA_TURBO["decode"] = decode
    return _NUMBA_TURBO[name]


def _encode(code: TurboCode, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    host = np.ascontiguousarray(to_device(bits, "cpu"), dtype=np.int8)
    out = np.empty((host.shape[0], code.n), dtype=np.int8)
    _get_numba_turbo("encode")(
        host, code.interleaver, _RSC.next, _RSC.outputs, _RSC.tail, out
    )
    return to_device(out, "cpu" if xp is np else "gpu")


def _decode(code: TurboCode, llrs: ArrayType, method: TurboIterations) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    host = np.ascontiguousarray(to_device(llrs, "cpu"), dtype=np.float64)
    rows = host.shape[0]
    scale = method.scale if method.scale is not None else (1.0 if method.exact else 0.7)
    bits = np.empty((rows, code.k), dtype=np.int8)
    _get_numba_turbo("decode")(
        host, code.interleaver, _RSC.next, _RSC.outputs, _RSC.tail, method.iterations,
        method.exact, scale, bits,
    )  # fmt: skip
    device = "cpu" if xp is np else "gpu"
    return _Decoded(
        to_device(bits, device),
        None,
        xp.ones(rows, dtype=bool),
        xp.full(rows, method.iterations, dtype=np.int32),
    )


_register(
    TurboCode,
    _Codec(
        encode=_encode,
        decode=_decode,
        default_method=lambda code: TurboIterations(),
        methods=(TurboIterations,),
        host=True,
    ),
)
