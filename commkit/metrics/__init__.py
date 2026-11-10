"""Performance metrics: EVM, SNR, BER, FER, SER, GMI, NGMI and MI, and their
closed forms over AWGN.

:mod:`~commkit.metrics.quality` holds :func:`evm`, :func:`snr` and
:func:`estimate_noise_variance`; :mod:`~commkit.metrics.error_rates`
:func:`ber`, :func:`fer` and :func:`ser`; :mod:`~commkit.metrics.information`
:func:`mi`, :func:`gmi` and :func:`ngmi`; :mod:`~commkit.metrics.awgn` the
expected values over AWGN at an Es/N0, :func:`awgn_ser`, :func:`awgn_ber`,
:func:`awgn_mi` and :func:`awgn_gmi`.  Import them from
:mod:`commkit.metrics`.

Metrics are the reporting layer.  They take received symbols (or bits, or
LLRs) at one sample per symbol and their reference, as arrays or as a 1-SPS
:class:`~commkit.core.Signal` whose ``reference`` holds the transmitted
symbols and bits.  They return host values: a ``float`` for ``(N,)`` input
and a ``(C,)`` NumPy array for ``(C, N)`` input.

Metrics never align, rotate or permute the received symbols against the
reference: synchronization, phase-ambiguity and channel-permutation
resolution are explicit receiver steps.  Lengths that disagree, empty
selections and frame Signals (use :func:`~commkit.core.extract_payload`)
raise.

Units and scaling
-----------------
Units are carried by argument names (``esn0_db``, ``snr_db``,
``noise_var``); there is no units object.  Three noise quantities, each
with one meaning:

* ``noise_var`` is the variance of the noise in *your samples*, what
  :func:`estimate_noise_variance` measures and
  :func:`~commkit.mapping.compute_llr` takes;
* ``snr_db`` is a power ratio in *your samples*, what :func:`snr` measures
  and ``apply_awgn(snr_db=)`` sets;
* ``esn0_db`` is the *channel's* textbook Es/N0, the same whether the
  samples are complex or real: error rates at an Es/N0 do not depend on
  the dtype.

For complex samples at one sample per symbol, ``snr_db = esn0_db`` and
``noise_var = Es / (Es/N0)``.  Real samples carry the in-phase half of
the noise: ``snr_db = esn0_db + 3.01 dB`` and ``noise_var = Es / (2
Es/N0)``.  The library converts where it takes ``esn0_db``
(``apply_awgn``, ``compute_llr``, :func:`mi`, :func:`gmi`, :func:`ngmi`,
:func:`commkit.coding.error_rate_curve`), so the conversion is never
written by hand.

==========================  =================================================
Quantity                    Definition in commkit
==========================  =================================================
Sample power ``P``          ``E[|x|^2]`` per sample.  At ``sps``
                            samples per symbol a normalized waveform has
                            ``P = 1/sps``; at the symbol rate ``P = 1``.
Symbol energy ``Es``        ``sps * P``: 1 for normalized waveforms and for
                            symbols on a constellation's scale.  Every
                            factory constellation, shaped ones included, has
                            unit average power under its pmf.
``Es/N0`` (``esn0_db``)     Symbol energy over noise spectral density
                            ``N0``.  ``apply_awgn`` adds noise of variance
                            ``N0 = sps * P / (Es/N0)`` per complex sample
                            (``N0 / 2`` in I and in Q) and ``N0 / 2`` per
                            real sample, which is ``Es/N0`` per symbol
                            after a matched filter and decimation.
Noise variance              ``noise_var = sigma^2 = E[|n|^2]`` per symbol on
(``noise_var``)             the constellation's scale: ``10 ** (-esn0_db /
                            10)`` at unit ``Es`` for complex symbols, half
                            of it for real ones.  The LLRs and information
                            rates use the likelihood of complex or real
                            noise after the symbols' dtype.
SNR (:func:`snr`, dB)       ``Es / E[|r/g - s|^2]`` at the symbol rate, with
                            ``g`` the data-aided gain (below): a power ratio
                            in the samples, equal to ``Es/N0`` for complex
                            AWGN after a matched filter, 3.01 dB above it
                            for real samples.
EVM (:func:`evm`, %)        ``100 sqrt(E[|r/g - s|^2] / Es)``, RMS, relative
                            to the reference's average power; ``= 100 *
                            10 ** (-SNR_dB / 20)``.  Blind EVM is relative
                            to the constellation's average power.
MI, GMI (bits/symbol)       At most ``H(X)``: ``log2(M)`` when uniform, the
                            entropy of the pmf when shaped.  GMI is the
                            bit-metric decoding rate
                            ``H(X) - sum_b H(B_b | Y)``.
NGMI                        ``1 - (H(X) - GMI) / m`` for ``m`` bits per
                            symbol: the FEC threshold metric of soft
                            decoders, 1 for error-free LLRs.
==========================  =================================================

Pre-FEC and post-FEC.  ``ber``, ``gmi`` and ``ngmi`` of a Signal measure
the channel and demapper before decoding: they compare with
``reference.bits``, the bits on the air.  Post-FEC, ``ber`` and ``fer``
take the :class:`~commkit.coding.DecodeResult` of
:func:`commkit.coding.decode` and the transmitted Signal (or its
``Reference``): ``ber(res, sig)`` compares with ``reference.info_bits``,
and ``fer(res, sig)`` counts blocks of the code's ``k`` bits.
:func:`estimate_noise_variance` gives the ``noise_var`` of ``compute_llr``
and the information rates.  :func:`commkit.coding.error_rate_curve`
measures a code's error rates against Es/N0 by Monte Carlo.

Which scale each metric reads:

* ``evm`` and ``snr`` normalize the reference to unit average power and
  divide the received symbols by the data-aided gain
  ``g = |<r s*>| / <|s|^2>`` (per channel, after ``num_skip_symbols``).
  Unlike normalizing to unit total power, which shrinks the symbols by
  ``1/sqrt(1 + 1/SNR)``, this is unbiased at any SNR.  Only the magnitude
  is used, so a rotation stays an error.
* ``mi`` and ``gmi`` of a Signal with a reference divide by the same gain,
  which puts the symbols on the constellation's scale whatever their
  normalization; ``noise_var`` is then on that scale.  Arrays are read as
  given.
* ``ser`` and ``ber`` decide on the symbols as given, against the
  constellation's points: decision thresholds are the receiver's.
"""

from .awgn import awgn_ber, awgn_gmi, awgn_mi, awgn_ser
from .error_rates import ber, fer, ser
from .information import gmi, mi, ngmi
from .quality import estimate_noise_variance, evm, snr

__all__ = [
    "awgn_ber",
    "awgn_gmi",
    "awgn_mi",
    "awgn_ser",
    "ber",
    "estimate_noise_variance",
    "evm",
    "fer",
    "gmi",
    "mi",
    "ngmi",
    "ser",
    "snr",
]
