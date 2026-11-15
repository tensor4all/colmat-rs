# Public bidiagonal dqds for LAPACK adapters

## Decisions

The LASQ1 connection in AtelierArith/Colmat.jl#306 needs the existing dqds
operation independently of the full bidiagonal SVD. The full public SVD always
forms vectors and its internal values-only route may recover with QR; neither
provides an explicit dqds boundary. The new public entry uses the existing
private algorithm without changing that algorithm or existing dispatch.

The public contract accepts real contiguous diagonal/off-diagonal slices and
caller-owned linear scratch. It commits descending finite values only on
success. Staging both arrays makes the immutable off-diagonal boundary and
error rollback explicit. Empty/singleton inputs avoid squared-factor range
issues; larger zero-diagonal inputs report nonconvergence and remain available
for recovery. LAPACK-specific iteration statuses and intermediate factor formats
remain the adapter's responsibility, through its preserved reference routine.

## Verification conclusions and constraints

The focused public tests cover both real precisions, exact-query scratch,
analytic singular values, independent full-SVD comparison, singleton subnormals,
invalid/nonfinite input and bitwise error rollback. A dedicated allocation
check covers both precisions beyond the former sorting-allocation boundary.
Both real-precision public contracts and the zero-allocation check pass. The
previous pinned native build rejects imports of both public entries with E0432;
the new public tests exercise the added boundary. Both executable documentation examples also pass. Sequential-only compilation
and hosted checks are still pending. No performance result or campaign adoption is claimed.
The existing decomposition kernels and dispatch are unchanged, so this is an
API extension rather than a decomposition performance experiment.
