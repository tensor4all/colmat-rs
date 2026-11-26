# Mixed-precision GEMM: first real slice

Status: draft implementation design for [colmat-rs#293](https://github.com/tensor4all/colmat-rs/issues/293).

## Goal and non-goals

Add a separate `mixed_matmul` entry point for these two operations:

```text
f32 × f32 → f64, computation and accumulation in f64
f32 × f64 → f64, computation and accumulation in f64
```

The existing homogeneous `matmul` signature and behavior are unchanged. The first slice has no
`f32` destination, no separately exposed accumulator type, no numeric `beta`, no whole-matrix
conversion allocation, no new microkernel, and no performance claim. tcontract planning and its
double-double representation are outside this change.

`c64 × f64 → c64` (CCR) is a follow-up only. Its candidate route is to pack each real value as
`(re, 0)` and reuse complex arithmetic, with an explicit type-safe account of real conjugation.
It is not implemented here.

## Public API

The proposed name and signature for draft review are:

```rust
pub fn mixed_matmul<
    LhsT: 'static + Copy + Send + Sync,
    RhsT: 'static + Copy + Send + Sync,
    M: Shape,
    N: Shape,
    K: Shape,
>(
    dst: impl AsMatMut<T = f64, Rows = M, Cols = N>,
    accum: Accum,
    lhs: impl AsMatRef<T = LhsT, Rows = M, Cols = K>,
    rhs: impl AsMatRef<T = RhsT, Rows = K, Cols = N>,
    alpha: f64,
    par: Par,
) where
    (LhsT, RhsT): MixedRealGemmToF64;
```

`MixedRealGemmToF64` is a public sealed marker trait owned by `t4a-gemm` and re-exported by colmat.
Downstream users cannot implement it for another pair. The two inputs are real, so this slice has no
explicit-conjugation variant and does not change `Conjugate` or its layout safety contract.

| `Lhs` | `Rhs` | `Dst` | `Acc` | Packed/compute type | Status |
| --- | --- | --- | --- | --- | --- |
| `f32` | `f32` | `f64` | `f64` | `f64` | Supported |
| `f32` | `f64` | `f64` | `f64` | `f64` | Supported |
| `f64` | `f32` | `f64` | `f64` | — | Compile-time rejection |
| any other pair | — | `f64` | `f64` | — | Compile-time rejection |
| `c64` | `f64` | `c64` | `c64` | `c64` candidate | Follow-up only |

Support is an explicit list, not inferred from matrix-product commutativity.

## Numerical contract

For each output element:

1. Convert each source value while packing it. `f32 → f64` conversion is exact. “Once” means once
   per packed-panel instance, not a stronger whole-call count independent of blocking and reuse.
2. Compute every product in `f64` after promotion. An `f32` product followed by widening is a
   different operation and is forbidden.
3. Hold partial sums in `f64`, including across `K` blocks. Fused multiply-add contraction and
   route-dependent summation order are allowed.
4. Apply `alpha` in `f64` to the completed product contribution, once per output rather than once
   per depth block.
5. `Accum::Replace` computes `C = alpha * A * B` and does not read the old destination.
   `Accum::Add` computes `C = C + alpha * A * B` and reads and adds old `C` exactly once.
6. Store directly into `f64`. There is no narrowing store in this slice.

Empty `M` or `N` returns without output access. `K = 0` writes zero for Replace and leaves the
output unchanged for Add. `alpha = 0` consequently writes zero for Replace and preserves old output
for Add, subject to the ordinary floating-point behavior of the implementation.

**The result is not block-size independent.** Changing `K` blocking, the execution route, or the
thread count can change `f64` summation order. Bitwise equality across those choices is not
promised. Correctness is judged against a naive `f64` reference with a scale-aware tolerance at the
low level and an independent `fx128` reference at the public colmat API. Exact tests separately
reject an early `f32` product and cover the minimum positive `f32` subnormal,
`f32::MIN_POSITIVE`, and `f32::MAX` products, all finite and nonzero in `f64`.

Bitwise repeatability follows the repository's existing contract: same machine and build at a fixed
thread count only. Tests must not assert bitwise sequential-versus-team equality.

## Route ownership

`t4a-gemm` owns source-to-packed conversion, blocked execution, runtime kernel selection, and packed
scratch. A typed `PackOperand<T>` abstraction gives homogeneous sources their original packer and
`f32 → f64` sources an exact converting packer. Any operand whose source type differs from the
kernel's `f64` type is forced through packing; a raw source pointer reaches a microkernel only when
its type is already `f64`.

The mixed facade calls the blocked driver directly. Homogeneous nano-GEMM, GEMV, rank-update,
horizontal, and native-x86 routes are deliberately not entered. Existing `f64` microkernels are
reused; there is no kernel specialized for either input pair. Runtime SIMD selection has a scalar
fallback, and mixed dispatch does not transpose `f32 × f64` into unsupported `f64 × f32`.

Colmat owns the public API, shape validation, `Accum` mapping, and `Par` policy. It maps the public
operation to the low-level `dst := old_dst_alpha * dst + product_alpha * lhs * rhs` convention with
`old_dst_alpha = 1` for Add and `0` for Replace, and `product_alpha = alpha`.

`Par::Seq` makes one mixed call with `Parallelism::None`. `Par::Team` uses the existing cost model
and disjoint output tiles, making one sequential mixed call per tile on the team's members. `K` is
never split, so there are no partial-result buffers or reductions. A team may choose a narrower
width or the sequential route without changing the numerical contract.

## Scratch and allocation

Packed buffers are `f64` regardless of source storage type. `t4a-gemm`'s `gemm_mixed_scratch`
returns an upper bound for the selected shape and all supported kernels, and
`gemm_mixed_with_stack` takes caller-owned storage for `Parallelism::None`. A short stack panics
before destination mutation. With Rayon, scratch remains owned by each worker's reusable
thread-local buffers and the scratch query is empty.

The first colmat API does not expose a scratch parameter. Its sequential and team-worker calls use
the same reusable per-thread packing storage as `t4a-gemm`, so a warm steady-state call allocates
nothing. No caller-created whole-matrix conversion matrices are permitted.

## Rejection and failure atomicity

There are two rejection levels:

- **Type level:** an unsupported pair cannot satisfy `MixedRealGemmToF64`, so the public call fails
  during type checking. Intentional `compile_fail` examples cover `f64 × f32` and complex input.
- **Dynamic dispatch:** the crate-private dispatcher checks the pair before shape handling, stride
  normalization, packing, or output access and returns `UnsupportedMixedGemm`. A unit test invokes
  it with `f64 × f32` and proves the destination is bitwise unchanged. Supported pairs always have
  a correct scalar fallback, so they have no representable unsupported-runtime error.

Colmat shape preconditions run before multiplication begins; a shape-panic test proves sentinel
output is unchanged. The low-level short-stack test likewise proves output atomicity.

## Verification plan

The low-level `t4a-gemm` tests cover both pairs, exact promotion, `Replace`/`Add` roles, empty and
zero-depth inputs, non-unit scalars, vector and irregular shapes, `K = 1031` across the 512-depth
small-shape block, row/column-major, padded and reversed views, destination guards, Rayon,
caller-stack equality and allocation, short-stack atomicity, dynamic rejection, and exact extreme
values. Existing homogeneous and caller-stack tests are regression controls.

Colmat's downstream integration tests use an independent `fx128` oracle. They cover both pairs,
`Replace`/`Add`, zero/unit/non-unit `alpha`, empty/zero-depth shapes, exact early-product
discriminators, blocked `K`, supported layouts and guards, `Par::Seq`, a fixed team, shape-failure
atomicity, and exact extreme values. Public doctests execute supported examples, and `compile_fail`
examples cover unsupported pairs. The homogeneous matmul suite remains a control. No benchmark is
required because no performance claim is made.

## Integration ordering and limitations

Develop and validate the repositories together. Open the `t4a-gemm` PR first after the integrated
checks pass. During coordinated development, colmat pins that branch's exact Git commit so its
build is reproducible. After `t4a-gemm` is merged and released, replace the Git pin with the release
and rerun colmat's focused checks before opening the colmat PR. Do not publish a crate without the
maintainer's per-release approval.

Remaining limitations are the two-type-pair matrix, no narrower destination, no mixed-domain
support, no public colmat scratch parameter, and no performance claim. Apple SME/AMX, wasm, and
non-x86 execution require their own validation; a feature build alone is not evidence that every
SIMD route was exercised.
