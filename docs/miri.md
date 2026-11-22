# Miri

Miri runs weekly (`.github/workflows/miri.yml`, Sunday 18:00 UTC) in two lanes. Each lane fails its
own job on a Miri violation, a failing test or a timeout, and a final `report` job files (or comments
on) one open issue labelled `miri-failure` that names the lanes that failed.

| lane | command | runner | job cap |
|---|---|---|---|
| `traits` | `cargo miri test -p colmat-traits --lib` (whole suite) | `ubuntu-latest` (x86_64) | 120 min |
| `colmat` | `scripts/run-miri-colmat.sh`: `cargo miri test -p colmat --lib -- --exact <names>` | `ubuntu-24.04-arm` (aarch64) | 60 min |

Install Miri once with `rustup component add --toolchain nightly miri rust-src`, then reproduce a lane
locally with `cargo +nightly miri test -p colmat-traits --lib` or, on an aarch64 host,
`RUSTUP_TOOLCHAIN=nightly scripts/run-miri-colmat.sh`.

## The main crate's selected test set

The `colmat` lane does not run the whole `--lib` suite. It runs the tests named in
`scripts/miri-colmat-tests.txt`, one libtest name per line, passed as `--exact` filters to a single
`cargo miri test` so the crate is built once. The runner checks that the run passed exactly the listed
tests with none ignored, so a renamed, deleted or newly `#[ignore]`d test fails the lane instead of
dropping out of it silently.

**Selection rule.** A test is in the set when it exercises the owned, indexed or aliasing APIs that
hold most of the crate's own `unsafe`: the owned `Mat` (`resize_with`, `push_row`/`push_col`, dropping
its elements when one `Drop` panics), strided views and `IndexMut`, `cloned`, `zip!` and the
elementwise operators built on it, `Perm` and diagonal application, sparse index construction, the
raw-pointer index sort, and the iterators and reductions over `Col`, `Row` and `Mat` in their owned,
`Ref` and `Mut` forms. The list file groups the 53 names by these headings.

**Excluded:**

- Numeric kernels: matmul (including `linalg::mat_ops::test::test_matmul_col_row`), the
  factorizations and solvers (`linalg::{cholesky, lu, qr, lq, svd, evd, gevd, …}`), `matrix_free`,
  the dense `sparse::linalg` factorizations, `stats` and `linalg::reductions`, by the selection rule
  of #267: the set targets the crate's container and index `unsafe`, not its kernels. Exclusion is
  not a cost necessity for all of them (`test_matmul_col_row` ran in 0.84 s under Miri on
  2026-11-08), and the kernels' own pointer handling is **not** covered by the tests above (see the
  gaps below).
- Threading, partition and cost-model tests (`utils::team`, `utils::thread`, `utils::cost`,
  `utils::partition`) and `linalg::tests` (layout copies on a team, factorization bit-for-bit
  checks): they spawn worker teams or run kernels, which Miri runs slowly and which are not in the
  selection rule.
- `io::npy`: byte reinterpretation of a file buffer, outside the selection rule.
- Every `#[ignore]`d test: 162 test functions at the time of writing
  (`cargo test -p colmat --lib -- --list --ignored`), from 43 `#[ignore]` attributes, most of them in
  the per-type (`f32`/`f64`/`c32`/`c64`) LAPACK validation macros, plus e.g.
  `linalg::matmul::tests::test_matmul` (`#[ignore = "takes too long"]`). The runner rejects one if
  it is listed.

**Gaps the set does not close.** The lane runs `--lib` only, as #267 specifies, so the integration
tests in `faer/tests/` are outside it. One of them, `faer/tests/owned_capacity.rs`, calls
`Mat::truncate` and was Miri-clean when run by hand
([worklog](worklogs/2026-11-07-zst-reservation-division.md); now
`cargo +nightly miri test -p colmat --test owned_capacity`); adding it to this lane is a separate
decision. Within `--lib`, no test calls `Mat::truncate` (shrinking), the
`Array` API in `utils::bound` (`as_mut`, range indexing), `PermRef` with `N = One`, `ZipEq::new_unchecked`
or `for_each_triangular_*` on a transposed and reversed operand. These are exactly the paths of
#254, #255 and #256, which external reports found by running Miri on reproducers outside the crate.
Their fixes are expected to bring regression tests; add those names to the list when they land, so the
lane guards the fixes.

No listed test writes through an uninitialized matrix destination: those paths are exercised by
`linalg::matmul::tests::test_matmul_with_conj_uninit*` (`faer/src/linalg/matmul/mod.rs`), which are
matmul tests and therefore outside the selection rule. The lane does reach `sort`'s uninitialized
offset scratch through `sort::tests::test_quicksort_big`, but not `Mat`'s uninit destinations;
widening the rule to those small tests is a separate decision.

**Cost.** The `finished in …s` that libtest prints under Miri is **not** wall-clock time: with
isolation on, Miri gives the program a virtual clock, so the figure is deterministic (`184.79s` on
every M5 run, `180.74s` on both CI runs) and says nothing about the budget. Wall-clock times:

- M5 (aarch64, 2026-11-22, `miri 0.1.0 (daf2e5e18b 2026-07-13)`), crate already built for Miri:
  `scripts/run-miri-colmat.sh` takes 70 s for all 53 tests. The heaviest single test,
  `mat::matref::tests::test_cloned_from_a_strided_source`, takes 54 s as its own invocation, against
  14 s for a trivial one (`sort::tests::test_quicksort`), so about 40 s of the set is that test.
- `ubuntu-24.04-arm` (runs 44965688890 and 44972192686, 2026-11-22): the whole job, toolchain
  install, `cargo miri setup` and the Miri build of the crate included, took 5 min 36 s and
  5 min 20 s; the test step (build plus run) took 4 min 50 s in the second run.

The 60-minute cap leaves room for the regression tests above and for slower runner days.

## Why aarch64: x86_64 is not Miri-covered

Miri executes no inline assembly. On x86_64, `pulp` 0.22.2 implements its masked u32 load/store with
an unguarded `core::arch::asm!` (`pulp-0.22.2/src/x86.rs:263` and `:281`). On aarch64 every `asm!` in
`pulp-0.22.2/src/aarch64.rs` sits behind `#[cfg(not(miri))]` next to a plain-Rust `#[cfg(miri)]`
replacement (e.g. `vcmlaq_0_f64` at `:7-30`), so a Miri build has no `asm!` to reach, and the set
above runs there. The `colmat` lane therefore runs on an aarch64 runner, and **the main crate's x86_64
code paths (`pulp`'s x86 SIMD arms and the crate's x86-specific dispatch) are not covered by Miri**.
Whether an x86_64 run would reach those `asm!` blocks at all (`Arch::new()` falls back to
`Arch::Scalar` when it detects neither AVX2 nor AVX-512, `pulp-0.22.2/src/x86.rs:313-323`) has not
been tested; the lane does not depend on the answer.

The aarch64 lane runs `pulp`'s `#[cfg(miri)]` replacements, not the NEON `asm!` the release build
executes, so it checks the crate's own pointer and index handling around the SIMD calls, not the
`asm!` itself.

The `traits` lane stays on x86_64: `colmat-traits` runs under Miri there today.

## Float precision under Miri

Three float tests in `colmat-traits` are `#[cfg_attr(miri, ignore)]` because they assert the last bit
of a subnormal or near-maximum result, which Miri's floating point does not reproduce; they are a
precision check, not a UB report.
