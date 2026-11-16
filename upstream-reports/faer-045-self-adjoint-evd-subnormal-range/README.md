# faer-045: The dense self-adjoint EVD loses a subnormal Hermitian matrix and returns `Ok`

**What happens:** `self_adjoint_eigenvalues` returns `Ok` with an all-zero spectrum for a nonzero symmetric matrix whose off-diagonal entries are subnormal. The exact spectrum of `s * (J - I)` is `s * [-1, -1, 2]`; at `s = 2e-308` and `s = 1e-310` every returned eigenvalue is `0`.
**Which input:** `A = s * (J - I)`, the 3 x 3 matrix with zero diagonal and `s` in every off-diagonal entry; `s = 2e-308` and `s = 1e-310`.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-16.

## Reproduce

```sh
cargo run
```

The program calls the public `self_adjoint_eigenvalues` with the lower triangle, then does the same for the diagonal matrix `diag(-s, -s, 2s)` as a control: those values are representable and come back exactly, so the loss is in the dense reduction rather than in the storage of the eigenvalues. It exits with status 1 when a case is wrong.

| input | expected | `faer` 0.24.4 |
|---|---|---|
| `s * (J - I)`, `s = 1` | `Ok`, [-1, -1, 2] | `Ok`, [-1.0000000000000004, -1, 2.0000000000000004] |
| `s * (J - I)`, `s = 2e-308` | `Ok`, [-2e-308, -2e-308, 4e-308] | **`Ok`, [0, 0, 0]** |
| `s * (J - I)`, `s = 1e-310` | `Ok`, [-1e-310, -1e-310, 2e-310] | **`Ok`, [0, 0, 0]** |
| `diag(-s, -s, 2s)`, `s = 2e-308` | `Ok`, [-2e-308, -2e-308, 4e-308] | `Ok`, [-2e-308, -2e-308, 3.999999999999999e-308] |

## Cause

`self_adjoint_eigenvalues` calls `linalg::evd::self_adjoint_evd` (`src/linalg/evd/mod.rs:270` in 0.24.4). It copies the lower triangle and reduces it to tridiagonal form with `tridiag::tridiag_in_place`; the reflectors are built by `make_householder_imp` (`src/linalg/householder.rs:59-76`):

```rust
let mut head_norm = (*head).abs();
if head_norm < min_positive() {
    *head = zero();
    head_norm = zero();
}
if tail_norm < min_positive() {
    return HouseholderInfo { tau: infinity(), ... };
}
```

Both the head `A[(1, 0)]` and the tail norm are `s = 2e-308`, just below `f64::MIN_POSITIVE` = 2.2250738585072014e-308, so the reduction zeroes the head and treats the tail as absent. The tridiagonal matrix is left with no coupling; the tridiagonal solver's own range normalization runs afterwards and cannot recover a spectrum from it. The complex Hermitian path shares this reduction.

## How the fork fixed it

`self_adjoint_evd` now normalizes the copied lower triangle before the reduction (`normalize_lower_range`, `faer/src/linalg/evd/mod.rs:336`, called at line 414). It multiplies the entries by powers of two until the largest one is at least `sqrt(min_positive) / eps`, and multiplies the computed eigenvalues back at the end; the eigenvectors are unchanged. A power of two keeps the scaling exact, and the diagonal control above shows that the storage was never the problem.

Fork regression tests, in `faer/src/linalg/evd/mod.rs`: `dense_symmetric_near_underflow`, `complex_hermitian_near_underflow` and `diagonal_control_round_trips_exactly`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same `make_householder_imp` is in upstream `main` at `0ecd846` (2026-10-22), and this program run against a path checkout of `main` exits 1 with the same zeros.

## References

* Fork fix: tensor4all/faer#358 (merge commit `58c764d7`, PR #359).
* Where it was found: tensor4all/faer#358.
