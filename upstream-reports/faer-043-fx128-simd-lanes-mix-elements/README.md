# faer-043: the `fx128` SIMD hooks mix the high and low parts of different elements

**What happens:** `fx128`'s SIMD load reads AoS memory (`[h0 l0 h1 l1 …]`) into a register pair whose arithmetic treats it as SoA (high lanes, low lanes), with no deinterleave. Every lane therefore combines parts of different elements, and `simd_mul` of `x` with itself returns `0.5556` for `x[0]^2 = 0.1111` and `0` for `x[1]^2 = 0.4444`.
**Which input:** any `fx128` slice loaded through `ComplexField::simd_mask_load_raw`, which is the route `SimdCtx::read` takes; here, `k / 3` for `k = 1..8`.
**Which version:** `faer` 0.24.4 with `faer-traits` 0.24.0, the latest releases on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-13.

## Reproduce

```sh
cargo run --locked
```

The program loads the first `LANES` values through the hooks, squares them with `simd_mul`, stores them through the hooks, and compares each with the scalar product. It exits with status 1 when a lane is wrong. Output on aarch64 (NEON, `LANES = 2`):

| value | expected `x^2` | `faer` 0.24.4 hooks |
|---|---|---|
| `x[0] = 1/3` | `(1.111e-1, 6.17e-18)` | **`(5.556e-1, 1.71e-33)`**, relative error 4.0 |
| `x[1] = 2/3` | `(4.444e-1, 2.47e-17)` | **`(0, 0)`**, relative error 1.0 |

The program also prints the loaded registers. The register read as the high lanes holds `[h0, l0]` and the one read as the low lanes holds `[h1, l1]`.

## Cause

- **Memory is AoS.** `Mat<fx128>` stores `fx128 = qd::Quad(f64, f64)` contiguously. `simd_mem_mask_between` agrees with that: it scales the element range by 2 into `f64` units (`faer-traits` 0.24.0 `src/lib.rs:5259`, `let start = start * 2;`).
- **Registers are SoA.** `SimdVec<S> = Quad<S::f64s>`, and the arithmetic is `qd::simd`'s, which reads `.0` as the high lanes and `.1` as the low lanes (`qd` 0.8.0 `src/simd/mod.rs:94` `add_estimate`, `:113` `mul`).
- **Nothing converts between the two.** `SimdCtx::read` and `write` cast the element pointer to `*const T::SimdVec<S>` (`faer` 0.24.4 `src/utils/simd.rs:69, 101, 134`). `fx128::simd_mask_load_raw` then loads `.0` from the first `LANES` `f64` and `.1` from the next `LANES` (`faer-traits` 0.24.0 `src/lib.rs:5279`; the store at `:5293` is the mirror image).

Not checked: which high-level routines reach these hooks with `fx128`. In the same build, `Col<fx128>` dot products and `norm_l2` with nonzero low parts, `n` from 1 to 101, agree with a scalar loop to about `1e-32` relative, so those routes did not show the defect. The dense product does not reach the hooks either: it panics first (faer-044).

## How the fork fixed it

Not fixed yet: tensor4all/colmat-rs#173. The plan, recorded in tensor4all/colmat-rs#171, keeps the storage AoS and makes the conversion part of the hook contract. A load deinterleaves (NEON `vld2q_f64`, or x86 loads plus `unpacklo/hi_pd`), a store interleaves, and the hooks take an element pointer instead of `*const SimdVec`.

## Affected versions

Confirmed by running this program: `faer` 0.24.4 (with `faer-traits` 0.24.0, `qd` 0.8.0, `pulp` 0.22.3) on aarch64. Also confirmed against upstream `main` at `0ecd846` as a path dependency: same output, exit 1. The code there is the same, at `faer-traits/src/lib.rs:5249` (mask) and `:5269`/`:5283` (load and store), and `faer/src/utils/simd.rs:69, 101, 134`.
Not run on x86_64, where `LANES` is 4 or 8; the same pointer cast applies.

## References

* Fork issue: tensor4all/colmat-rs#173.
* Where it was found: tensor4all/colmat-rs#171, while comparing colmat's `fx128` with the double-double GEMM of `SpM-lab/sparse-ir-rs` (`sparse-ir-basis/src/ddgemm.rs`), which splits its inputs into high and low arrays before the kernel.
