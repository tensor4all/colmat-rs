# faer-046: `norm_l2` of a contiguous vector returns inf or 0 at the ends of the `f64` range

**What happens:** `norm_l2` of an owned column returns `inf` for a one-entry vector whose exact norm is `1.348e308`, and `0` for a one-entry vector of `5e-320` and for sixteen copies of `5e-320`, although every one of those norms is finite and representable.
**Which input:** a `1 x 1` column holding `0.75 * f64::MAX`, a `16 x 1` column holding `5e-320`, and a `1 x 1` column holding `5e-320`. The program also runs `0.5 * f64::MAX` and `16 x 1e-300` as controls.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-23.

## Reproduce

```sh
cargo run
```

The program builds an owned column for each case, calls `norm_l2`, prints the exact norm and the actual value, and exits with status 1 when the result is not finite or disagrees by more than a relative `1e-12`. An owned column takes the contiguous SIMD path; `faer-008` covers the strided fallback.

| input | expected | `faer` 0.24.4 |
|---|---|---|
| `1 x (0.5 * MAX)` | `8.988465674311579e307` | `8.988465674311579e307` |
| `1 x (0.75 * MAX)` | `1.3482698511467367e308` | **`inf`** |
| `16 x 1e-300` | `4e-300` | `4e-300` |
| `16 x 5e-320` | `2e-319` | **`0`** |
| `1 x 5e-320` | `5e-320` | **`0`** |

## Cause

`norm_l2` keeps the sum of squares in three bands and scales each term before squaring (`src/linalg/reductions/norm_l2.rs` in 0.24.4): the SIMD kernel multiplies by `sqrt_min_positive()` and `sqrt_max_positive()` (`:19-20`, accumulated at `:26-34`), and the result is selected at `:157-165`. The two scales are the square roots of the range limits with no margin:

* the large band squares `x * sqrt(min_positive)`; that square overflows when `x` is above about `9e307` (just over `0.5 * MAX`), so the selected result is `inf`;
* the small band squares `x * sqrt(1 / min_positive)`; that square underflows to zero when `x` is below about `3.3e-316`, so the selected result is `0`.

Both thresholds are representable norms: `0.75 * MAX` is finite, and a subnormal vector's norm is a subnormal number rather than zero.

## How the fork fixed it

`norm_l2` in colmat keeps the same three bands but widens both scales by machine epsilon (`faer/src/linalg/reductions/norm_l2.rs:7-13`): the down scale is `sqrt_min_positive * eps` and the up scale is `sqrt_max_positive / eps`, so the smallest subnormal can be squared without underflowing and `MAX` can be squared without overflowing (`:30`, `:156`). The strided fallback accumulates the same three sums and scales before squaring, so it no longer needs the running `hypot` (`:158`).

Fork regression tests in `faer/src/linalg/reductions/norm_l2.rs` cover real and imaginary inputs in both native precisions, contiguous and strided views, the pairwise split, subnormal inputs, finite norms whose squares overflow, true overflow, and NaN mixed with infinity.

## Affected versions

Confirmed by running this program: `faer` 0.24.4.
Not fixed upstream: the same three-band kernel and the same two scales are in upstream `main` at `0ecd846` (2026-10-22), `faer/src/linalg/reductions/norm_l2.rs:19-34`, `:157-165`; this program run against a path checkout of `main` exits 1 with the same rows.

## References

* Fork fix: tensor4all/colmat-rs#230 (merge commit `195e625a`), for tensor4all/colmat-rs#229.
* Where it was found: tensor4all/colmat-rs#229, a follow-up to the strided case in tensor4all/faer#13 (`faer-008`).
