"""Finite-field arithmetic (scaffold).

GF(2) / GF(2^m) element and polynomial arithmetic - the algebraic foundation
shared by the BCH and Reed-Solomon codes.  GF lookup tables are small and stay
on the host (NumPy); only the hot array math will run on the data's device.
No implementation yet.
"""
