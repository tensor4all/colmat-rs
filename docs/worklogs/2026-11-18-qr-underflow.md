# Householder reflectors near underflow

## Decisions

- Rescale a tiny nonzero reflector in the shared generator rather than copying
  the whole QR matrix. Following the small-vector scaling in LAPACK's xLARFG,
  lift the head and tail by the reciprocal of `MIN_POSITIVE / eps`, recompute
  their norms, then form the reflector. Scale the returned norm and diagonal
  back; the stored vector and tau are dimensionless.
- The out-of-place reflector copies the scaled tail into its existing output
  buffer, leaving its source unchanged. Check whether the tail is really zero
  when its norm is zero: older norm kernels can give zero for a nonzero
  subnormal vector. An actual zero tail leaves the head, including a subnormal
  diagonal, unchanged.
- Only dense QR opts in. In the existing rank-one SVD regression, the fused
  bidiagonalization encountered a rounded-zero row norm, a finite tau, and an
  unrepresentable original-scale reciprocal. QR needs the normalized vector
  and tau, while that SVD update also consumes the original-scale reciprocal.
  The SVD and the other reflector callers retain the previous behavior.

## Verification conclusions and constraints

- Public QR checks reconstruction, orthonormality and tau for the issue's tall,
  square and wide shapes in four native dtypes. A reflector-level test also
  covers the out-of-place path at the smallest subnormal scale. The existing
  rank-one SVD regression passes without changing its assertions.
- This does not scale trailing QR updates or address overflow; at the smallest
  scales, rounding of the stored factor at the original scale is unavoidable.
  No performance or upstream comparison.
