"""
Shared coding interfaces and conventions (scaffold).

Defines the contracts every code in this package will share - ``Encoder`` /
``Decoder`` protocols, a ``CodewordResult`` dataclass, and the hard/soft
interface conventions - so that algebraic, convolutional, and modern
capacity-approaching codes present a uniform surface.

Soft (LLR) convention - agreed with :func:`commkit.mapping.compute_llr`:

    LLR_k = log[ P(b_k = 0 | r) / P(b_k = 1 | r) ]

so a **positive** LLR means bit 0 is more likely, a **negative** LLR means bit
1, and the magnitude is the confidence.  Soft-input decoders here consume LLRs
in exactly this sign/scale; soft-output decoders (BCJR, belief propagation)
emit them in the same convention.  No implementation yet.
"""
