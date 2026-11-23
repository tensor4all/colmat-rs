# faer-049: Generalized eigenvectors are NaN for a 1 x 1 block at extreme scale

**What happens:** the generalized eigenproblem `S u = lambda T u` returns `Ok` with `[NaN, NaN]` as the right eigenvector of `lambda = 2`, for a pencil whose factors are O(1) except for one diagonal entry of `2e-200`. The exact eigenvector is `(1, 1) / sqrt(2)`.
**Which input:** the upper triangular pencil `S = [[1, 1], [0, s11]]`, `T = diag(1, s11 / 2)` with `s11 = 2e-200`. The eigenvalues are `1` and `2`. The program also runs `s11 = 2`, `2e-8`, `2e-100` and `2e-150` as controls.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-23.

## Reproduce

```sh
cargo run
```

The program calls the public `gevd::gevd_real` with the pencil, takes the column whose `alpha / beta` is closest to `2`, and checks `|S v beta - T v alpha|` relative to the larger of the two terms. It exits with status 1 when the column is not finite or the residual exceeds `1e-10`.

| `s11` | expected `v` for `lambda = 2` | `faer` 0.24.4 |
|---|---|---|
| `2` | `[0.7071067811865475, 0.7071067811865475]`, residual 0 | same |
| `2e-8` | same | same |
| `2e-100` | same | same |
| `2e-150` | same | same |
| `2e-200` | finite `[0.7071, 0.7071]`, residual 0 | **`[NaN, NaN]`, residual NaN** |

The `s11 = 2e-150` row is the boundary: the pair's `temp^2` is then `4e-300`, still representable; at `2e-200` it is `4e-400` and flushes to zero. The failure at `2e-200` is recorded as NaN here; the fork observed the same underflow as an `[0, 1]` null column at `e7381104`, which is the same zero coefficient pair reached through a different normalization.

## Cause

The conversion stage after the QZ iteration computes the coefficient pair of a 1 x 1 block as

```rust
let temp = safmin.fmax((A[(je, je)].abs() * ascale).fmax(B[(je, je)].abs() * bscale));
let salfar = A[(je, je)].mul_real(temp).mul_real(ascale);
let sbeta = B[(je, je)].real() * temp * bscale;
```

(`src/linalg/gevd/mod.rs:568-573` in 0.24.4; the same lines recur at `:801`, `:1069` and `:1178`). LAPACK's `?tgevc` uses the **reciprocal** of that maximum, so the pair is normalized to O(1); here the pair carries an extra factor `temp` on each side, i.e. `temp^2`, and the pair is already scaled by `ascale` and `bscale`. At `s11 = 2e-200` both `salfar` and `sbeta` underflow to zero, and the normalization of the column then divides zero by zero and returns NaN with `Ok`.

## How the fork fixed it

The four 1 x 1 sites in `faer/src/linalg/gevd/mod.rs` (the first at `:1019-1030`) take the reciprocal of the maximum, as `?tgevc` does; 2 x 2 blocks keep their own `generalized_eigval_2x2` path. The fork regression covers this pencil for real and complex, `f32` and `f64`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same `temp = safmin.fmax(...)` and the same two products are in upstream `main` at `0ecd846` (2026-10-22), `faer/src/linalg/gevd/mod.rs:568-573` (and the three later copies); this program run against a path checkout of `main` exits 1 with the same NaN.

## References

* Fork fix: tensor4all/colmat-rs#224 (merge commit `ba864a24`), for tensor4all/colmat-rs#223.
* Where it was found: tensor4all/colmat-rs#223, a follow-up to the saved-Schur-pencil work in tensor4all/colmat-rs#219.
