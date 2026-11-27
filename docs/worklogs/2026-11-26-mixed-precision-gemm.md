# Mixed-precision GEMM first slice (colmat-rs#293)

## Decisions

- Add `mixed_matmul` only for `f32 × f32 → f64` and `f32 × f64 → f64`, with `Dst = Acc = f64`.
  Homogeneous `matmul` is unchanged. CCR, an `f32` destination, `f64 × f32`, numeric `beta`,
  tcontract planning, and performance claims are outside this change.
- `t4a-gemm` owns typed packing and blocked execution. Its homogeneous packer remains the identity
  path; every `f32` mixed operand is converted exactly into packed `f64`, and existing `f64`
  microkernels perform every product and partial sum. There is no whole-matrix conversion or
  per-pair microkernel. The colmat dependency is pinned to companion commit
  `5a10f8f72d8790b8136acea59d04389d691b72e8` during coordinated development.
- `mixed_matmul` bypasses homogeneous nano-GEMM, GEMV, rank-update, and native-x86 routes. `Par::Seq`
  makes one packed call. `Par::Team` reuses the existing cost model and disjoint output tiles, with
  one sequential mixed call per tile and no `K` split or partial-result allocation. Scratch is packed
  `f64`; colmat exposes no new public scratch parameter.
- The sealed public trait rejects unsupported pairs at compile time. The crate-private dynamic
  dispatcher in `t4a-gemm` also returns an error before output access; a dedicated test proves the
  destination is bitwise unchanged. Shape failures and short caller stacks are atomic as well.
- Products are computed after promotion, with `f64` partial sums, one product-`alpha` application,
  and one old-output addition per output. `K` blocking can change summation order, so results are not
  block-size independent and sequential/team bitwise equality is not promised. Tests use naive `f64`
  and independent `fx128` references with tolerances. Exact cases separately cover early-product
  rounding and extreme `f32` values.
- Develop and validate both repositories together, then open `t4a-gemm` first. After it is merged and
  released, replace the draft Git pin with that release before opening colmat's PR. No crate is
  published by this work, and no upstream-faer issue or PR is created.

## Verification conclusions and constraints

Host: Linux x86_64, Rust 1.96.0. The companion facade tests used `RUST_FONTCONFIG_DLOPEN=1` because
the host has no `pkg-config`; no dependency source or registry cache was edited.

- The `t4a-gemm` full workspace suite, no-default-features build, mixed default and `x86-v4` tests,
  dynamic-rejection test, caller-stack tests, and public rustdoc examples pass.
- Colmat's nine downstream integration tests pass: both supported pairs, exact early-product
  discriminators, independent `fx128` comparisons, `Replace`/`Add`, zero and non-unit `alpha`, empty
  and zero-depth inputs, blocked `K`, row/column-major, padded and reversed layouts, destination
  guards, fixed-team execution, extreme `f32` inputs, and shape-failure atomicity.
- The homogeneous matmul control passes 86 tests with its existing long test ignored. Public
  doctests, including both intentional unsupported-pair `compile_fail` examples, pass. The
  no-default-features `std` build and the existing zero-allocation parallel-region control pass,
  as do all nine mixed tests with `x86-v4`.
- Full colmat validation passes on the integrated branch: 867 library tests with 162 existing
  ignores, every integration-test target, and 99 doctests, with no failures. The final companion
  lint-corrected head `5a10f8f72d8790b8136acea59d04389d691b72e8` also passes all nine downstream
  mixed integration tests; the full-suite run used the previous companion head, whose delta is
  style-only.
- Non-x86 targets, Apple SME/AMX, wasm, and every SIMD path have not been executed. A feature build
  is not treated as proof that a particular instruction set ran.
- No benchmark or performance-adoption result is claimed. The temporary Git revision must be
  replaced by a released `t4a-gemm` version before the colmat PR is opened.
