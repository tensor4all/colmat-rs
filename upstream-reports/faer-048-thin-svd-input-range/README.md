# faer-048: `thin_svd` returns `NoConvergence` near overflow and wrong factors near underflow

**What happens:** `MatRef::thin_svd` fails on finite, well-conditioned matrices near either end of the `f64` range. With the largest entry at `0.25 * MAX` or `0.5 * MAX` it returns `Err(NoConvergence)` although every singular value is representable; with the entries scaled to `MIN_POSITIVE` or `16 * MIN_POSITIVE` it returns `Ok` with a relative reconstruction residual of `0.23` to `0.64`. LAPACK's `?gesvd`/`?gesdd` returns a residual of about `1e-16` on the same matrices.
**Which input:** `A_ij = ((7 i + 13 j + 3) mod 11) / 11 - 0.45 + delta_ij`, scaled by `1`, `0.25 * MAX`, `0.5 * MAX`, `16 * MIN_POSITIVE` and `MIN_POSITIVE`, in the shapes `6 x 4`, `4 x 4` and `4 x 6`.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-23.

## Reproduce

```sh
cargo run
```

For each case the program factors `A`, rebuilds `U (S / s) V^H` and compares it with `A / s`, so the check itself neither overflows nor underflows. It prints the residual, or `Err(NoConvergence)`, and exits with status 1 when a case is not `Ok` with a residual of at most `1e-13`.

| shape | scale | expected | `faer` 0.24.4 |
|---|---|---|---|
| `6 x 4` | `1` | `~1e-16` | `6.46e-16` |
| `6 x 4` | `0.25 * MAX` | `Ok`, `~1e-16` | **`Err(NoConvergence)`** |
| `6 x 4` | `0.5 * MAX` | `Ok`, `~1e-16` | **`Err(NoConvergence)`** |
| `6 x 4` | `16 * MIN_POSITIVE` | `~1e-16` | **`2.72e-1`** |
| `6 x 4` | `MIN_POSITIVE` | `~1e-16` | **`6.39e-1`** |
| `4 x 4` | `1` | `~1e-16` | `1.05e-15` |
| `4 x 4` | `0.25 * MAX` | `Ok`, `~1e-16` | **`Err(NoConvergence)`** |
| `4 x 4` | `0.5 * MAX` | `Ok`, `~1e-16` | **`Err(NoConvergence)`** |
| `4 x 4` | `16 * MIN_POSITIVE` | `~1e-16` | **`2.69e-1`** |
| `4 x 4` | `MIN_POSITIVE` | `~1e-16` | **`5.69e-1`** |
| `4 x 6` | `1` | `~1e-16` | `7.03e-16` |
| `4 x 6` | `0.25 * MAX` | `Ok`, `~1e-16` | **`Err(NoConvergence)`** |
| `4 x 6` | `0.5 * MAX` | `Ok`, `~1e-16` | **`Err(NoConvergence)`** |
| `4 x 6` | `16 * MIN_POSITIVE` | `~1e-16` | **`2.34e-1`** |
| `4 x 6` | `MIN_POSITIVE` | `~1e-16` | **`6.41e-1`** |

The table was recorded on an aarch64 macOS host with `cargo run --locked`. The issue that reports this case tested on x86_64 Linux, where `0.25 * MAX` was accurate for the `6 x 4` case and `0.5 * MAX` already failed; the `6 x 4` `0.25 * MAX` row above therefore fails on this host but not there. The issue did not test the other shapes at `0.25 * MAX`. Which of the two reductions is used for a given shape can differ between the hosts, so the failure's first scale may move; the failure at the ends of the range does not.

## Cause

`Svd::new_thin` (`faer/src/linalg/solvers.rs:1332`) dispatches to `svd` (`src/linalg/svd/mod.rs:530`) and then to `svd_imp` (`:326`). `svd_imp` copies the input into its work matrix (`bid.copy_from(matrix)`, `:355`) and runs `bidiag::bidiag_in_place` (`:356`) on it as given. There is no equivalent of LAPACK's `?lascl` step, which scales the input into `[smlnum, bignum]` before `?gesvd`/`?gesdd` and scales the singular values back:

* near the top of the range, the Householder norms and the products in the bidiagonalisation overflow, and the iteration reports `NoConvergence`;
* near the bottom, the same products underflow or lose their digits, and `U`, `S` and `V` no longer reconstruct the input, although the call returns `Ok`.

The singular values themselves are representable in every row of the table: they are the values at scale `1` multiplied by the scale.

## How the fork fixed it

The shared dense SVD entry in colmat applies LAPACK `?gesvd`'s safe range (`faer/src/linalg/svd/mod.rs:664-727`): it takes the largest absolute real or imaginary component from `norm_max`, chooses an exact power of two that brings the input into `[min_positive, 1 / min_positive]`, copies the input scaled by that power, and scales the singular values back in the common exit path. An ordinary input pays for the maximum reduction but no scaling pass. Nonfinite input, or a nonfinite singular value after the scale-back, produces `NoConvergence`.

Fork tests cover all four native dtypes, both sides of the tall-QR threshold, wide transposition, general strides, and the values-only and thin entry points.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same `svd_imp` copies the input unscaled in upstream `main` at `0ecd846` (2026-10-22), `faer/src/linalg/svd/mod.rs:326-356`, `:530`; this program run against a path checkout of `main` exits 1 with the same rows.

## References

* Fork fix: tensor4all/colmat-rs#232 (merge commit `2f764d2a`), for tensor4all/colmat-rs#227.
* Where it was found: tensor4all/colmat-rs#227, found together with the QR reflector's underflow (`faer-047`) while testing the polar decomposition (tensor4all/colmat-rs#226).
