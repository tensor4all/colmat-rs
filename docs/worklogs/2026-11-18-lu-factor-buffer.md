# LU: an explicit factor-buffer API (`PartialPivLu::factor_into`)

Issue: [tensor4all/colmat-rs#191](https://github.com/tensor4all/colmat-rs/issues/191). Follow-up to
#210 (both owning factors in one allocation) and #213, which measured where the owning route's
remaining 1024/1536 gap is and left this decision to the maintainer.

## Decisions

**The reusable buffer is the `PartialPivLu` itself, not a second type.** `PartialPivLu` already owns
exactly one factor allocation and the two permutation arrays, so `PartialPivLu::factor_into(&mut
self, A, par)` is the caller's buffer: `new` is now `empty()` + one `factor_into`, i.e. the
convenience wrapper over the buffer route, and there is one code path, so the two cannot drift. A
separate `PartialPivLuBuffer` plus a borrowing result was rejected because the result would have to
borrow the buffer, and then either the borrowed type re-implements `ShapeCore`/`SolveCore`/
`DenseSolveCore` (or a caller cannot solve with the factors it just computed) or the buffer never
outlives the factors. `PartialPivLu::empty()` is public so a caller can hold a factor buffer before
it has a matrix, which is the design the issue approved.

The reuse rule, in the doc comment and in `partial_piv_lu_factor_into_regrows_the_buffer`: the factor
buffer is kept when it already holds `col_stride(m) * n + col_stride(size) * size` elements for the new
shape (the number #210's single-allocation layout needs), and reallocated exactly once when it does
not; the permutation arrays are `Vec`s and keep their storage while the row count does not grow.

`P` changes representation from `perm::Perm<usize>` to two `Vec<usize>`, and `P()` builds the
`PermRef` from them. `lu_in_place` writes the permutation through `&mut [usize]`, and the owned `Perm`
offers only `&[..]`; without the change a factorization into a reused buffer would allocate both
permutation arrays on every call, which is the allocation this API exists to remove.

**Panic state.** `factor_imp` reports `0 x 0` before it writes anything and commits the dims after the
last fill, so a panic leaves an empty factorization rather than one whose `P()` would have to hand out
a half-written permutation. `PermRef` indexes rows with its entries, so that trade is about soundness,
not tidiness, and it costs two stores per call.

**What this deliberately does not do.** The transposition scratch
(`min(m, n) * size_of::<usize>()` bytes) is still allocated per call, as in `new` and in the other
owning decompositions in `solvers.rs`; keeping it would need manual `Clone`/`Debug` impls
(`dyn_stack::MemBuffer` has neither) for 8 KiB that faults no page once the arena is warm.
A conjugated `A` is still materialised by `to_owned` first, as `new` does. Row-major input on a team
at `m * n >= 256 * 256` still takes `lu_in_place`'s own whole-matrix copy allocation per call
(`factor.rs`); this record's numbers for order C show the factor-buffer pages are still what the reuse
removes, and that copy is untouched.

## Verification conclusions and constraints

Measured, not inferred. Yardstick: the same library's own routes on one host, one process per route,
`benchmarks/scripts/pinned.sh` from `tensor4all/colmat-benchmark` (`1876a51`) with the idle gate
accepted, one measurement at a time, measured CPUs 1T `4` / 8T `4-11` (one L3 domain), observer
pinned to `0-3,12-15`. Host: EVO-X1Pro, AMD Ryzen AI 9 HX 470 (Zen 5), 24 logical CPUs, two L3
domains, THP `madvise`. Probe: `~/work/colmat-191buf-probe` (a copy of #213's
`~/work/colmat-191lu/probe` with the `colmat` path pointed at this revision and one route,
`lib_reuse`, added: a `PartialPivLu::empty()` kept across iterations and refreshed by `factor_into`).
It times `restore` + one call and reads minor faults per call from `/proc/self/stat` field 10, exactly
as #213's `profiles/probe.txt` did, so the two records are comparable. Median of the printed calls
(60 at 1024, 30 at 1536), two independent sets at the headline cell.

| n | dtype | order | T | `lib` (`new`) | `lib_reuse` (`factor_into`) | `owning_reuse` | recovered | faults/call `lib` -> reuse |
|---|---|---|---|---|---|---|---|---|
| 1024 | c64 | F | 8 | 25.09 | **20.48** | 20.26 | **4.61** | 8193 -> 0 |
| 1024 | c64 | F | 8 | 25.19 | **20.53** | — | **4.66** | 8193 -> 0 |
| 1024 | c64 | C | 8 | 25.49 | **21.21** | — | **4.28** | 8193 -> 0 |
| 1024 | c64 | F | 1 | 118.71 | **105.73** | — | **12.98** | 8193 -> 0 |
| 1536 | f64 | F | 8 | 23.68 | **18.86** | 18.77 | **4.82** | 9217 -> 0 |
| 1536 | c64 | F | 8 | 68.12 | **59.26** | 57.86 | **8.85** | 18433 -> 0 |
| 1024 | f64 | F | 8 | 6.60 | 6.57 | 6.90 | 0.04 | 68 -> 0 |

ms per call, one set unless a row is repeated; `owning_reuse` is #213's hand-written
"factor and split into already-resident buffers" composite, which has no copy of its own, so
`lib_reuse` landing within 0.25 ms of it at 1024 c64 is the check that the API reaches the prize.

Against #213's record (`lib` 24.76 / `owning_reuse` 20.38 at 1024 c64 8T, faults 8193), the 4.4-4.8 ms
first touch it priced is now 4.6 ms recovered by the shipped API, and the factor buffer's pages are
fault-free on every repeat call. 1024 f64 is the control where the buffer is 16 MiB: 68 faults, a
0.04 ms gap, and no reuse claim. The C-order cell keeps the win (4.28 ms) because `lu_in_place`'s copy
comes from a warm arena; that copy is still charged to both routes.

Steady-state allocation count (`faer/tests/no_step_alloc.rs`, n = 256, widths 4/8/70):
`factor_into` allocates **once, 2048 bytes** = `min(m, n) * size_of::<usize>()`, the transposition
scratch; the 1 MiB factor buffer is never reallocated (the check fails with "allocated 1048576 bytes"
when the buffer is forced to reallocate on every call). Bit identity: the factor into a reused buffer
is `new`'s for the same `A` and the same `par`, for f32/f64/c64, square/tall/wide, column-major and
row-major input, on `Par::Seq` and a team of three, and the solve after it is `new`'s solve.

Upstream revision: **not measured**. This is a library-internal ownership API; the vs-MKL question
belongs to the harness, and the harness's LU owning rows still time the hand-written composite
`explicit_ops::partial_lu_of` rather than `PartialPivLu` (#213's follow-up, still open, not in this
change).

Limits of this record: one host (EVO-X1Pro); no vendor arm; sizes 1024/1536 only; 1T, 4T is not
in the pinned set for this change (1024 c64 1T and 8T are); the numbers are a focused probe, not a
campaign, and must not enter the published site (`R-performance-95`).

Evidence: `~/work/colmat-191buf-bench/run1/` (`probe.txt`, `probe.log` with the observer pin, the
binary sha256 and one idle-guard line per run; no attempt was rejected) and
`~/work/colmat-191buf-bench/run2/` (the repeat of the headline cell);
`~/work/colmat-191buf-bench/probe-factor-buffer.sh` is the driver, `~/work/colmat-191buf-probe` the
probe source.

Gates: `cargo test -p colmat --lib --features std,rayon,threadteam` (839 passed, 162 ignored),
`... linalg::lu` (55 passed, 12 ignored), `... solvers::tests::partial_piv_lu` (3 passed),
`cargo test -p colmat --test no_step_alloc --features std,rayon,threadteam` (1 passed),
`cargo build -p colmat --no-default-features --features std` (0 errors, the 25 pre-existing warnings),
`cargo clippy -p colmat --all-targets --features std,rayon,threadteam -- -D warnings` (none),
`cargo fmt --all -- --check` (the only file it reports is `evd/tridiag_evd.rs`, 10 hunks, already
reported at `b5558b56`; this change's files are clean), and `cargo doc` adds no diagnostic (the four
unresolved links it reports are `team.rs`/`cost.rs`/`lib.rs` at `b5558b56` too).

## Review

Plan gate: `ollama-cloud/glm-5.3-flash:high`, read-only, **VERDICT: CHANGES** (7 findings, all
folded into the plan: `empty()` must be public, the focused filter is `solvers::tests::partial_piv_lu`,
acceptance bars must be same-run relative, the fault bar is `<= 8` because the transposition scratch
is still per call, assert the parallel dispatch in the allocation test, add an `f32` case, two wrong
line numbers).

Finished diff: the repository's default reviewer, `openai/gpt-6.1-sol`, **could not run** on this
host - its OAuth access token expired 2026-11-02 and the refresh returns `invalid_grant`, which needs
an interactive re-login, and `commandcode/gpt-6.1-sol` answers `MODEL_NOT_IN_PLAN`. Substituted, and
said so here rather than silently: `ollama-cloud/glm-5.3-flash:high` on the same diff,
**VERDICT: APPROVE** with 7 MINOR findings. Applied: pin the per-call allocation count to exactly one
(now also what the record claims), run the shrink-then-grow chain (130x70 -> 33x257 -> 130x70) in the
regrow test, one `LU_SHAPES` for the module, delete the dead `new_without_default` allow (clippy is
clean without it), drop the host-specific ms figure from the public rustdoc and say "the buffer keeps
its address" instead of "every address in it", which is what the test pins. Not applied: converting
the older test's inline c64 `bits` closure to `digest` (unrelated churn in #213's test; the two
compare different things, `u64` pairs of `L`/`U` versus raw bytes), and asserting `U`'s block offset
from `packed_at`/`square_at` (internal layout, not a contract; bit identity with `new` covers what a
caller sees).

Plan: the pre-implementation plan and the review that approved it live in the working tree's
`IMPLEMENTATION-PLAN.md`; #213's plan for the previous phase is at `git show
b5558b56:IMPLEMENTATION-PLAN.md`, which this change does not overwrite.
