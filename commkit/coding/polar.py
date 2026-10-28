"""Polar codes: :class:`PolarCode` and the decoders :class:`SC` and :class:`SCL`.

A polar code of length ``N = 2^n`` transmits ``x = u G_N`` with ``G_N`` the
``n``-fold Kronecker power of ``[[1, 0], [1, 1]]`` (no bit reversal, as 5G
NR), the information bits on the most reliable positions of ``u`` and
zeros on the others (frozen).  :meth:`PolarCode.nr` takes the positions from
the reliability sequence of 3GPP TS 38.212 (Table 5.3.1.2-1) and can attach
a CRC to the information bits, whose check selects among the candidates of
list decoding (CA-SCL).  Encoding is ``n`` butterfly passes on the data's
device.

Decoding is successive cancellation (:class:`SC`) or its list version
(:class:`SCL`), with min-sum check-node updates and the path metric of
Balatsoukas-Stimming et al. (2015): sequential per codeword, so Numba over
codewords on the CPU, and GPU input makes one round trip.  LLRs and partial
sums are kept level by level in buffers of ``2N`` (level ``s`` at offsets
``2^s ... 2^(s+1) - 1``), so a list path is copied in ``O(N)``.

Scope: the mother code with ``E = N`` transmitted bits; NR's rate
matching, parity-check bits and input interleaving are not included.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from .base import _Code, _Codec, _Decoded, _encode_blocks, _register
from .crc import CRC

__all__ = ["PolarCode", "SC", "SCL"]

# 3GPP TS 38.212 Table 5.3.1.2-1: bit indices in ascending reliability (N = 1024).
# fmt: off
_NR_SEQUENCE = np.array([
    0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128,
    12, 33, 65, 20, 256, 34, 24, 36, 7, 129, 66, 512, 11, 40, 68, 130,
    19, 13, 48, 14, 72, 257, 21, 132, 35, 258, 26, 513, 80, 37, 25, 22,
    136, 260, 264, 38, 514, 96, 67, 41, 144, 28, 69, 42, 516, 49, 74, 272,
    160, 520, 288, 528, 192, 544, 70, 44, 131, 81, 50, 73, 15, 320, 133, 52,
    23, 134, 384, 76, 137, 82, 56, 27, 97, 39, 259, 84, 138, 145, 261, 29,
    43, 98, 515, 88, 140, 30, 146, 71, 262, 265, 161, 576, 45, 100, 640, 51,
    148, 46, 75, 266, 273, 517, 104, 162, 53, 193, 152, 77, 164, 768, 268, 274,
    518, 54, 83, 57, 521, 112, 135, 78, 289, 194, 85, 276, 522, 58, 168, 139,
    99, 86, 60, 280, 89, 290, 529, 524, 196, 141, 101, 147, 176, 142, 530, 321,
    31, 200, 90, 545, 292, 322, 532, 263, 149, 102, 105, 304, 296, 163, 92, 47,
    267, 385, 546, 324, 208, 386, 150, 153, 165, 106, 55, 328, 536, 577, 548, 113,
    154, 79, 269, 108, 578, 224, 166, 519, 552, 195, 270, 641, 523, 275, 580, 291,
    59, 169, 560, 114, 277, 156, 87, 197, 116, 170, 61, 531, 525, 642, 281, 278,
    526, 177, 293, 388, 91, 584, 769, 198, 172, 120, 201, 336, 62, 282, 143, 103,
    178, 294, 93, 644, 202, 592, 323, 392, 297, 770, 107, 180, 151, 209, 284, 648,
    94, 204, 298, 400, 608, 352, 325, 533, 155, 210, 305, 547, 300, 109, 184, 534,
    537, 115, 167, 225, 326, 306, 772, 157, 656, 329, 110, 117, 212, 171, 776, 330,
    226, 549, 538, 387, 308, 216, 416, 271, 279, 158, 337, 550, 672, 118, 332, 579,
    540, 389, 173, 121, 553, 199, 784, 179, 228, 338, 312, 704, 390, 174, 554, 581,
    393, 283, 122, 448, 353, 561, 203, 63, 340, 394, 527, 582, 556, 181, 295, 285,
    232, 124, 205, 182, 643, 562, 286, 585, 299, 354, 211, 401, 185, 396, 344, 586,
    645, 593, 535, 240, 206, 95, 327, 564, 800, 402, 356, 307, 301, 417, 213, 568,
    832, 588, 186, 646, 404, 227, 896, 594, 418, 302, 649, 771, 360, 539, 111, 331,
    214, 309, 188, 449, 217, 408, 609, 596, 551, 650, 229, 159, 420, 310, 541, 773,
    610, 657, 333, 119, 600, 339, 218, 368, 652, 230, 391, 313, 450, 542, 334, 233,
    555, 774, 175, 123, 658, 612, 341, 777, 220, 314, 424, 395, 673, 583, 355, 287,
    183, 234, 125, 557, 660, 616, 342, 316, 241, 778, 563, 345, 452, 397, 403, 207,
    674, 558, 785, 432, 357, 187, 236, 664, 624, 587, 780, 705, 126, 242, 565, 398,
    346, 456, 358, 405, 303, 569, 244, 595, 189, 566, 676, 361, 706, 589, 215, 786,
    647, 348, 419, 406, 464, 680, 801, 362, 590, 409, 570, 788, 597, 572, 219, 311,
    708, 598, 601, 651, 421, 792, 802, 611, 602, 410, 231, 688, 653, 248, 369, 190,
    364, 654, 659, 335, 480, 315, 221, 370, 613, 422, 425, 451, 614, 543, 235, 412,
    343, 372, 775, 317, 222, 426, 453, 237, 559, 833, 804, 712, 834, 661, 808, 779,
    617, 604, 433, 720, 816, 836, 347, 897, 243, 662, 454, 318, 675, 618, 898, 781,
    376, 428, 665, 736, 567, 840, 625, 238, 359, 457, 399, 787, 591, 678, 434, 677,
    349, 245, 458, 666, 620, 363, 127, 191, 782, 407, 436, 626, 571, 465, 681, 246,
    707, 350, 599, 668, 790, 460, 249, 682, 573, 411, 803, 789, 709, 365, 440, 628,
    689, 374, 423, 466, 793, 250, 371, 481, 574, 413, 603, 366, 468, 655, 900, 805,
    615, 684, 710, 429, 794, 252, 373, 605, 848, 690, 713, 632, 482, 806, 427, 904,
    414, 223, 663, 692, 835, 619, 472, 455, 796, 809, 714, 721, 837, 716, 864, 810,
    606, 912, 722, 696, 377, 435, 817, 319, 621, 812, 484, 430, 838, 667, 488, 239,
    378, 459, 622, 627, 437, 380, 818, 461, 496, 669, 679, 724, 841, 629, 351, 467,
    438, 737, 251, 462, 442, 441, 469, 247, 683, 842, 738, 899, 670, 783, 849, 820,
    728, 928, 791, 367, 901, 630, 685, 844, 633, 711, 253, 691, 824, 902, 686, 740,
    850, 375, 444, 470, 483, 415, 485, 905, 795, 473, 634, 744, 852, 960, 865, 693,
    797, 906, 715, 807, 474, 636, 694, 254, 717, 575, 913, 798, 811, 379, 697, 431,
    607, 489, 866, 723, 486, 908, 718, 813, 476, 856, 839, 725, 698, 914, 752, 868,
    819, 814, 439, 929, 490, 623, 671, 739, 916, 463, 843, 381, 497, 930, 821, 726,
    961, 872, 492, 631, 729, 700, 443, 741, 845, 920, 382, 822, 851, 730, 498, 880,
    742, 445, 471, 635, 932, 687, 903, 825, 500, 846, 745, 826, 732, 446, 962, 936,
    475, 853, 867, 637, 907, 487, 695, 746, 828, 753, 854, 857, 504, 799, 255, 964,
    909, 719, 477, 915, 638, 748, 944, 869, 491, 699, 754, 858, 478, 968, 383, 910,
    815, 976, 870, 917, 727, 493, 873, 701, 931, 756, 860, 499, 731, 823, 922, 874,
    918, 502, 933, 743, 760, 881, 494, 702, 921, 501, 876, 847, 992, 447, 733, 827,
    934, 882, 937, 963, 747, 505, 855, 924, 734, 829, 965, 938, 884, 506, 749, 945,
    966, 755, 859, 940, 830, 911, 871, 639, 888, 479, 946, 750, 969, 508, 861, 757,
    970, 919, 875, 862, 758, 948, 977, 923, 972, 761, 877, 952, 495, 703, 935, 978,
    883, 762, 503, 925, 878, 735, 993, 885, 939, 994, 980, 926, 764, 941, 967, 886,
    831, 947, 507, 889, 984, 751, 942, 996, 971, 890, 509, 949, 973, 1000, 892, 950,
    863, 759, 1008, 510, 979, 953, 763, 974, 954, 879, 981, 982, 927, 995, 765, 956,
    887, 985, 997, 986, 943, 891, 998, 766, 511, 988, 1001, 951, 1002, 893, 975, 894,
    1009, 955, 1004, 1010, 957, 983, 958, 987, 1012, 999, 1016, 767, 989, 1003, 990, 1005,
    959, 1011, 1013, 895, 1006, 1014, 1017, 1018, 991, 1020, 1007, 1015, 1019, 1021, 1022, 1023,
], dtype=np.int64)
# fmt: on


@dataclass(frozen=True, eq=False)
class PolarCode(_Code):
    """Polar code of length ``N = 2^n`` with given information positions.

    Parameters
    ----------
    length : int
        ``N``, a power of two from 2 to 1024 (and larger for arbitrary
        positions).
    info_positions : array_like
        Indices of ``u`` that carry information (and CRC) bits, distinct;
        the others are frozen to zero.
    crc : CRC, optional
        A CRC over the ``k`` information bits, appended before placing them
        (``len(info_positions) = k + crc.degree``); list decoding keeps the
        best candidate that passes it.

    Notes
    -----
    The information (then CRC) bits fill ``info_positions`` in increasing
    index order, as TS 38.212 places them.
    """

    length: int
    info_positions: np.ndarray
    crc: CRC | None = None

    def __post_init__(self) -> None:
        n = self.length
        if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n & (n - 1):
            raise ValueError(f"length must be a power of two >= 2, got {n!r}.")
        p = np.unique(np.asarray(self.info_positions))
        if p.size != np.size(self.info_positions) or p.size == 0:
            raise ValueError("info_positions must be distinct and not empty.")
        if p.dtype.kind not in "iu" or p[0] < 0 or p[-1] >= n:
            raise ValueError(f"info_positions must be integers in [0, {n}).")
        p = p.astype(np.int64)
        p.setflags(write=False)
        object.__setattr__(self, "info_positions", p)
        if self.crc is not None:
            if not isinstance(self.crc, CRC):
                raise TypeError("crc must be a CRC or None.")
            if self.crc.n != p.size:
                raise ValueError(
                    f"crc covers {self.crc.k} bits plus {self.crc.degree}; the code has "
                    f"{p.size} information positions."
                )

    @property
    def n(self) -> int:
        """Transmitted bits per codeword, ``N``."""
        return self.length

    @property
    def k(self) -> int:
        """Information bits per codeword (without the CRC)."""
        return int(
            self.info_positions.size - (0 if self.crc is None else self.crc.degree)
        )

    def __repr__(self) -> str:
        crc = "" if self.crc is None else f", crc={self.crc.degree}"
        return f"PolarCode(n={self.n}, k={self.k}{crc})"

    @classmethod
    def nr(cls, k: int, n: int, *, crc: str | None = None) -> PolarCode:
        """Polar code with the 5G NR information set (TS 38.212 clause 5.3.1.2).

        Parameters
        ----------
        k : int
            Information bits per codeword.
        n : int
            Code length, a power of two from 32 to 1024.
        crc : {"6", "11", "16", "24A", "24B", "24C"}, optional
            Attach this NR CRC (``"24C"`` for downlink control, ``"11"`` and
            ``"6"`` for uplink control); the most reliable ``k + degree``
            positions carry the bits.

        Returns
        -------
        PolarCode
        """
        if n not in (32, 64, 128, 256, 512, 1024):
            raise ValueError(f"n must be a power of two from 32 to 1024, got {n!r}.")
        check = None if crc is None else CRC.nr(crc, k)
        size = k + (0 if check is None else check.degree)
        if not 1 <= size < n:
            raise ValueError(
                f"k = {k} with the CRC needs fewer than n = {n} positions."
            )
        sequence = _NR_SEQUENCE[_NR_SEQUENCE < n]
        return cls(n, np.sort(sequence[-size:]), check)

    @classmethod
    def from_frozen(cls, frozen: Any, *, crc: CRC | None = None) -> PolarCode:
        """Polar code from a frozen-position mask.

        Parameters
        ----------
        frozen : array_like
            ``(N,)`` bool, True at the frozen positions; ``N`` a power of two.
        crc : CRC, optional
            A CRC over the information bits, appended before placing them
            (the unfrozen positions number ``k + crc.degree``); list decoding
            keeps the best candidate that passes it.

        Returns
        -------
        PolarCode
        """
        mask = np.asarray(frozen, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("frozen must be a 1-D mask.")
        return cls(int(mask.size), np.flatnonzero(~mask), crc)

    @cached_property
    def _frozen(self) -> np.ndarray:
        mask = np.ones(self.length, dtype=np.bool_)
        mask[self.info_positions] = False
        return mask


@dataclass(frozen=True)
class SC:
    """Successive-cancellation decoding (min-sum check nodes)."""


@dataclass(frozen=True)
class SCL:
    """Successive-cancellation list decoding, CRC-aided when the code has a CRC.

    Parameters
    ----------
    list_size : int, default 8
        Paths kept, ``>= 1`` (1 is SC).
    """

    list_size: int = 8

    def __post_init__(self) -> None:
        s = self.list_size
        if isinstance(s, bool) or not isinstance(s, int) or s < 1:
            raise ValueError(f"list_size must be an integer >= 1, got {s!r}.")


def _transform(u: ArrayType) -> ArrayType:
    """``u G_N`` (mod 2) for rows of ``u``, by ``log2 N`` butterfly passes."""
    x = u.copy()
    rows, n = x.shape
    h = 1
    while h < n:
        view = x.reshape(rows, n // (2 * h), 2, h)
        view[:, :, 0, :] ^= view[:, :, 1, :]
        h *= 2
    return x


def _encode(code: PolarCode, bits: ArrayType) -> ArrayType:
    _, xp, _ = dispatch(bits)
    payload = bits if code.crc is None else _encode_blocks(code.crc, bits)
    u = xp.zeros((bits.shape[0], code.length), dtype=np.int8)
    u[:, xp.asarray(code.info_positions)] = payload
    return _transform(u)


def _get_numba_polar() -> Any:
    """Numba SC list decoder over codewords (``list_size = 1`` is SC).

    Per path: ``alpha`` (LLRs) and ``beta`` (partial sums of left children)
    in level order, offset ``2^s`` for level ``s``; ``u`` the decisions and
    ``pm`` the path metric.  Bit ``i`` recomputes the LLRs from level
    ``ctz(i) + 1`` (``n`` for ``i = 0``) down: ``g`` with the left child's
    partial sums where bit ``s - 1`` of ``i`` is 1, ``f`` (min-sum) where it
    is 0.
    """
    from . import _numba_loops

    return _numba_loops.polar_scl


def _decode(code: PolarCode, llrs: ArrayType, method: SC | SCL) -> _Decoded:
    _, xp, _ = dispatch(llrs)
    host = np.ascontiguousarray(to_device(llrs, "cpu"), dtype=np.float32)
    rows = host.shape[0]
    size = 1 if isinstance(method, SC) else method.list_size
    if code.crc is None:
        crc_g = np.zeros((0, 0), dtype=np.int8)
        crc_c = np.zeros(0, dtype=np.int8)
    else:
        g, c = code.crc._tables
        crc_g, crc_c = g.astype(np.int8), c.astype(np.int8)
    u = np.empty((rows, code.length), dtype=np.int8)
    valid = np.empty(rows, dtype=np.bool_)
    _get_numba_polar()(
        host, code._frozen, size, crc_g, crc_c, code.info_positions, code.k, u, valid
    )
    if code.crc is None:
        valid[:] = True
    device = "cpu" if xp is np else "gpu"
    bits = u[:, code.info_positions[: code.k]]
    return _Decoded(
        to_device(bits, device),
        None,
        to_device(valid, device),
        xp.ones(rows, dtype=np.int32),
    )


_register(
    PolarCode,
    _Codec(
        encode=_encode,
        decode=_decode,
        default_method=lambda code: SCL(),
        methods=(SC, SCL),
    ),
)
