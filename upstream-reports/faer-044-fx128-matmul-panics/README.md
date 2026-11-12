# faer-044: the dense product of `fx128` matrices panics outside `cfg(test)`

**What happens:** `&a * &b` for `Mat<fx128>` panics with `explicit panic` in the SIMD micro-kernel. The kernels refuse every non-native scalar unless `faer` itself is compiled with `cfg(test)`, so `faer`'s own `fx128` matmul tests pass and a downstream crate panics.
**Which input:** a `13 x 17` times `17 x 11` `fx128` product; the same shape in `f64` (the control) succeeds.
**Which version:** `faer` 0.24.4, the latest release on crates.io, and upstream `main` at `0ecd846` (2026-10-22).
**Added to this list:** 2026-11-13.

## Reproduce

```sh
cargo run --locked
```

The program runs the `f64` control, then the `fx128` product inside `catch_unwind`, and exits with status 1 when the product panics (or, once it does not, when it is inaccurate).

| call | expected | `faer` 0.24.4 |
|---|---|---|
| `f64` `13x17 * 17x11` | shape `(13, 11)` | `(13, 11)` |
| `fx128` `13x17 * 17x11` | a product within `1e-29` relative | **panic** at `linalg/matmul/mod.rs:93`: `explicit panic` |

Backtrace (`RUST_BACKTRACE=1`): `matmul_vertical::matmul_simd::<qd::Quad>` ← `matmul_imp::<qd::Quad>` ← `matmul::<qd::Quad, …>` ← `Mul for &Mat`.

## Cause

Both micro-kernels of `src/linalg/matmul/mod.rs` start with the same gate (`faer` 0.24.4, lines 86-93 and 349-356):

```rust
if const {
    !(T::IS_NATIVE_F64 || T::IS_NATIVE_F32 || T::IS_NATIVE_C64 || T::IS_NATIVE_C32)
        && !cfg!(test)
} {
    panic!();
}
```

`cfg!(test)` is true only while `faer`'s own unit tests are compiled, so the in-tree `fx128` product tests take the `else` branch and pass, and no test exercises what a user runs. `fx128` is a supported, exported scalar (`faer::fx128`), and the product dispatches to these kernels for it.

Removing the gate alone would not be enough: the kernels would then run `fx128` through the SIMD hooks of faer-043, which mix the parts of different elements.

## How the fork fixed it

Not fixed yet: tensor4all/colmat-rs#174. It is to be fixed together with tensor4all/colmat-rs#173 (faer-043). For the longer term, tensor4all/colmat-rs#171 plans a level-3 kernel for double-double scalars that packs its panels into separate high and low parts.

## Affected versions

Confirmed by running this program: `faer` 0.24.4 (with `qd` 0.8.0) on aarch64. Also confirmed against upstream `main` at `0ecd846` as a path dependency: panic at `faer/src/linalg/matmul/mod.rs:93`, exit 1. The gate is at the same lines there (86-93 and 349-356).
Not run on x86_64. The gate is a `const` condition that does not depend on the architecture.

## References

* Fork issue: tensor4all/colmat-rs#174.
* Related: faer-043 (tensor4all/colmat-rs#173), and the plan in tensor4all/colmat-rs#171.
