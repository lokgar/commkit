"""
Finite-field arithmetic (scaffold).

GF(2) / GF(2^m) element and polynomial arithmetic - the algebraic foundation
shared by the BCH and Reed-Solomon codes.  GF lookup tables are small and stay
host-side (NumPy); backend dispatch is reserved for the hot array math only.
No implementation yet.
"""
