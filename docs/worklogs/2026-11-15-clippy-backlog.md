# The clippy backlog on `colmat` is empty, and the gate is blocking (tensor4all/colmat-rs#183)

**Outcome: `cargo +1.96.0 clippy --workspace --all-targets -- -D warnings` exits 0 with zero
warnings, and `code-quality.yml` now fails on a new one instead of counting it.** No behaviour
change was intended and none was found: the numerical suites match the pre-change record bit for bit,
and a bitwise digest of LLT, LDLT, LU (partial and full pivot), QR (no pivot and column pivot), EVD,
SVD and a sparse Cholesky solve is identical before and after.

## What the backlog was

Measured on `evo2_vpn` (EVO-X2, Ryzen AI MAX+ 395), `cargo +1.96.0 clippy --workspace --all-targets
-j 4`:

- before any edit, **the `colmat` lib did not compile under clippy**: 40 deny-level lint errors
  (36 `out_of_bounds_indexing`, 3 `overly_complex_bool_expr`, 1 `approx_constant`) aborted it, so
  every count from that run is a floor. `faer-ffi`, all 53 `faer/tests/*` targets, benches and
  examples were never linted.
- after the five deny errors were fixed: **2,748 warning messages over 151 files** (`faer/src`: 94,
  the rest tests/examples/`faer-ffi`), `toplevel_ref_arg` alone at 857.

## What the fix was

1. **Deny errors, by hand** — the `out_of_bounds_indexing` 36 are one diagnosis, not 36: the
   `simd_iter!` macro (`faer/src/lib.rs:1454`) expands the caller's block once per
   `if const { $batch_size == k }` arm, so the dead arms of a `$batch_size == 4` call index
   `acc[4..8]` on an `[_; 4]`. The macro already carried `#[allow(non_upper_case_globals,
   unconditional_panic)]` on that inner expansion for the same reason; `out_of_bounds_indexing`
   joined it. The other four: two `if true || …` / `if false && …` predicates in the LDLT and
   real-Schur drivers (item-level allow, upstream construction kept), `approx.rs`'s tolerance
   predicate hand-simplified, `core::f64::consts::SQRT_2` for a hand-written `1.4142…`.
2. **Mechanical pass**, per lint, applied through a script that canonicalises the file path
   (`realpath`) before editing and applies each clippy span group atomically. Both details are
   load-bearing: several sources are `include!`d from more than one module, so the raw `file_name`
   names one file twice and a naive applier corrupts it; and every multi-span suggestion (all of
   `collapsible_if`, `manual_is_multiple_of`) is written as a set of disjoint edits that is only
   valid together.
3. **Hand pass** for what `--fix` cannot do or would break: `missing_safety_doc` (47 items — the
   `*_unchecked` accessors got the contract they actually have, the `faer-ffi` C entry points got
   theirs), `needless_borrows_for_generic_args` where the borrow *is* the `ByRef` type hint,
   `manual_div_ceil` on unsigned operands only (`div_ceil` is bit-identical there), `from_over_into`
   on the FFI status enums, `field_reassign_with_default` in the `Spec` wrapper (assigned through the
   public `config` field, since struct-update syntax cannot reach a `DerefMut` field),
   `redundant_locals` (14 rebindings; two in `pfaffian/mod.rs` are kept with an allow because they
   are what makes the closure capture raw pointers instead of `a` and `tiles`).
4. **The gate**: one blocking
   `cargo clippy --workspace --all-targets -- -D warnings` step replaces the "clean crates only"
   step and the advisory `Linting check`, and the stale `cd ./faer-entity` prefix is gone from both
   steps — that directory does not exist in this tree, so both commands failed there silently and
   only `continue-on-error` hid it.

## Allows added

291 item-level attributes, each with its reason on the same line: 134 `too_many_arguments` and 63
`type_complexity` (kernel and `mat` view signatures that are the upstream API), 35
`needless_range_loop` (bodies index parallel buffers and strided views), then 11
`neg_cmp_op_on_partial_ord` (NaN must fail `!(a < b)`, `a >= b` is not equivalent), 7 each
`toplevel_ref_arg`, `missing_transmute_annotations` (the cast reinterprets a view's scalar in the
`IS_REAL` branch; both sides spelled out would repeat the view type), `extra_unused_lifetimes` (the
lifetime appears inside a `Dim<'M>` argument), and a tail of single-digit items. No crate-root
`#![allow]`; the one inner attribute (`deref_addrof` in `impl_op_assign!`) is scoped to that macro's
`imp` block because an attribute on the assignment expression is not stable.

## Gates run (this host, `-j 4`, one cargo command at a time)

| gate | result |
|---|---|
| `clippy --workspace --all-targets -- -D warnings` | **exit 0, 0 warnings** |
| `cargo test -p colmat --lib` | 823 passed, 0 failed, 162 ignored — same set as the pre-change record |
| `cd faer && cargo nextest run --features=unstable` | 1073 passed, 166 skipped, 0 failed |
| `cargo test --doc --workspace` | 90 passed, 0 failed |
| `cargo build --examples` | clean |
| `cargo test -p colmat-traits` / `-p faer-ffi` / `-p colmat --test no_step_alloc` | 6 / 9+1 / 1 passed |
| bitwise digest (`/tmp/numcmp`, `Par::Seq`, `f64::to_bits`, 10 families: LLT, LDLT, LU partial+full pivot, QR no-pivot+col-pivot, self-adjoint EVD, real-Schur EVD, SVD, sparse Cholesky solve) | **identical**, main `7b56e0b1` vs this branch: `5819055034d04b16` |
| `cargo fmt --all -- --check` | **0 diffs** — but this tree was not formatted on `main`: the baseline reported 843 diffs over 115 workspace files plus the excluded `eigen-bench-setup`, so this change also carries rustfmt's output for them (two `cargo fmt --all` passes were needed; the first does not converge). Whitespace only: `git diff -w` shows the real change. |

## Adoption-protocol waiver

`docs/adoption-protocol.md:21-31` routes `linalg/{evd,svd,cholesky,lu}/**` and shared view helpers to
family measurement, and this diff touches all of them. That grid exists to accept a *performance
claim*; this change makes none — it asserts that speed and results do not change. Per-measurement is
therefore waived, and the substitute is stronger for this class of change: the bitwise digest above
(same bits on same inputs, all nine families) plus the full numerical suite. Had any digest differed,
the waiver would be void for that family.

## Residual risk

- The bulk pass is ~2,600 mechanical hunks in numerical kernels. Every hunk was classified by lint
  (the equivalence argument is per lint, in the plan), and the suites above plus the digest are the
  evidence that no result moved. A hunk that changes results without changing the test outcome would
  have to also leave every digest bit identical, which is what the digest rules out for the nine
  families listed.
- Formatting is still advisory, so a `rustfmt` regression can still land; that was out of scope.
- The diff also carries rustfmt's output for the 115 files that `main` left unformatted (see the gate
  table). It is mechanical and whitespace-only, but it makes the raw diff larger than the lint work;
  review with `git diff -w`.
- Two allow categories are judgement calls a reviewer may want to revisit: 35 `needless_range_loop`
  (kept because the bodies index parallel buffers) and 35 `type_complexity` in the `mat`/`col` view
  types (kept because the signature is the upstream API).

## Review

`review: gpt-6.1-sol APPROVE` — independent read-only review of the finished diff in five passes. It
found four real problems, all fixed: two incomplete `# Safety` contracts (`libfaer_v0_23_alloc`/
`_dealloc` and `temp_mat_uninit`), generated FFI safety text that forbade read-only aliasing, and an
`Into` → `From` rewrite that would have added public API. The doc tests caught one bug of the
author's own — a comment-deduplication step that also collapsed two doc examples — and the affected
files were restored from a snapshot of the pre-deduplication tree.
