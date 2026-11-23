# faer-047: Unpivoted QR returns a wrong factorization for input scaled to `MIN_POSITIVE`

**What happens:** `MatRef::qr` (no pivoting) reports success with a factorization that does not reconstruct the input: the relative residual `||A - Q R||_F / ||A||_F` is `0.52` to `0.56` for a well-conditioned matrix whose entries are scaled to `f64::MIN_POSITIVE`. LAPACK's `?geqrf` returns `2e-16` to `4e-16` on the same matrices. At `16 * MIN_POSITIVE` and above the factorization is accurate.
**Which input:** `A_ij = ((7 i + 13 j + 3) mod 11) / 11 - 0.45 + delta_ij`, scaled by `s = f64::MIN_POSITIVE`, in the shapes `6 x 4`, `4 x 4` and `4 x 6`.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-23.

## Reproduce

```sh
cargo run
```

For each shape and scale the program factors `A`, rebuilds `Q R / s`, and compares it with `A / s`, so the check itself does not underflow. It prints the residual and exits with status 1 when any residual exceeds `1e-13`.

| shape | scale | expected | `faer` 0.24.4 |
|---|---|---|---|
| `6 x 4` | `1` | `~1e-16` | `1.59e-16` |
| `6 x 4` | `16 * MIN_POSITIVE` | `~1e-16` | `1.52e-16` |
| `6 x 4` | `MIN_POSITIVE` | at most `1e-13` | **`5.43e-1`** |
| `4 x 4` | `1` | `~1e-16` | `1.59e-16` |
| `4 x 4` | `16 * MIN_POSITIVE` | `~1e-16` | `1.54e-16` |
| `4 x 4` | `MIN_POSITIVE` | at most `1e-13` | **`5.19e-1`** |
| `4 x 6` | `1` | `~1e-16` | `1.94e-16` |
| `4 x 6` | `16 * MIN_POSITIVE` | `~1e-16` | `2.01e-16` |
| `4 x 6` | `MIN_POSITIVE` | at most `1e-13` | **`5.59e-1`** |

The table was recorded on an aarch64 macOS host with `cargo run --locked`. The issue that reports this case recorded the same 0.5 relative residual on x86_64 Linux at the same scale.

## Cause

The unpivoted QR driver calls `make_householder_imp` (`src/linalg/householder.rs:59-76` in 0.24.4) for each column. That function forms the reflector from the column at its original scale:

```rust
let tail_norm = tail.norm_l2();
let mut head_norm = (*head).abs();
...
let norm = head_norm.hypot(&tail_norm);
let ref signed_norm = sign * norm.to_cplx::<T>();
let ref head_with_beta = &*head + signed_norm;
let head_with_beta_inv = head_with_beta.recip();
```

For a column whose entries are `2.2e-308`, the stored reflector vector is of that order and `head_with_beta_inv` is of the order of `1e308`. Applying the reflector to the trailing columns multiplies those tiny entries by each other, so the products (about `1e-616`) underflow. The diagonal of `R` is returned at the original scale and the call reports success, but `R` no longer carries the trailing update. LAPACK's `?larfg` avoids this by lifting the vector by `1 / (safmin / eps)` before forming the reflector, and scales `beta` back afterwards.

## How the fork fixed it

The shared generator in colmat rescales a tiny nonzero reflector (`faer/src/linalg/householder.rs:67-137`): it lifts the head and the tail by the reciprocal of `min_positive / eps`, recomputes their norms, forms the reflector, then scales the returned norm and diagonal back. The stored vector and `tau` are then dimensionless. Only the dense QR callers opt in: the fused bidiagonalisation's rank-one update also consumes the original-scale reciprocal, so the SVD keeps the previous behaviour there.

Fork regression tests check reconstruction, orthonormality and `tau` for the tall, square and wide shapes in four native dtypes, and a reflector-level test covers the out-of-place path at the smallest subnormal scale.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: `make_householder_imp` is unchanged in upstream `main` at `0ecd846` (2026-10-22), `faer/src/linalg/householder.rs:59-76`; this program run against a path checkout of `main` exits 1 with the same rows.

## References

* Fork fix: tensor4all/colmat-rs#231 (merge commit `02fe96cb`), for tensor4all/colmat-rs#228.
* Where it was found: tensor4all/colmat-rs#228, found together with the thin SVD's range problem (`faer-048`) while testing the polar decomposition (tensor4all/colmat-rs#226). The issue notes that the same class of defect exists in `self_adjoint_evd` (tensor4all/faer#159).
