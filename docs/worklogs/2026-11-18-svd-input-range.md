# Dense SVD input scaling

## Decisions

- Apply the safe range from LAPACK xGESVD at the shared dense SVD entry,
  using the largest absolute real or imaginary component from `norm_max`.
  Choose an exact power of two as the eigensolver does. Component magnitudes
  remain representable even when the modulus of a complex entry would overflow.
- Reuse the existing bidiagonalization or tall QR copy; restore the singular
  values once in the common exit path. There is no new scratch space or heap
  allocation. An ordinary input pays for the maximum reduction but incurs no
  scaling pass. Nonfinite input or a nonfinite singular value after scale-back
  produces `NoConvergence`.

## Verification conclusions and constraints

- Tests cover all four native dtypes, both sides of the QR-reduction threshold,
  wide transposition, general strides, and values-only and thin SVD. Residuals
  are checked after rescaling the factors to ordinary magnitude so the check
  itself does not overflow.
- Near the very bottom of the range the returned singular values still round
  to the original dtype's subnormal grid. Unrelated bidiagonal entry points
  and the polar QDWH path are unchanged. No performance or upstream comparison.
