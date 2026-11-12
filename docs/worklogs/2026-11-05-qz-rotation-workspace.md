# Generalized-Hessenberg rotation range and AED workspace

Issues: [#256](https://github.com/tensor4all/faer/issues/256),
[#260](https://github.com/tensor4all/faer/issues/260).

The current base is main `1be2ace`, merged into this branch by merge commit
`d800f78` (the earlier base was `e4c6d57e`). This PR owns the generalized-Hessenberg
rotation correction and AED Sylvester index reservation. The even-rounded
window query (#253, related to #252) and row-zero sentinel correction (#255)
are already in main and are not changes contributed by this PR.

As [the maintainer clarified](https://github.com/tensor4all/faer/pull/258#issuecomment-6638555959),
the earlier closure followed deletion of the #255 base branch; it was not a
review rejection or a decision to abandon this change. The PR now targets main
and its rebase conflicts have been resolved, preserving main's regressions
and this PR's additional CI registrations.

## Design

The old generalized-Hessenberg Givens helper forms a reciprocal of the norm or
cosine. A finite subnormal can make that reciprocal infinite, producing
nonfinite factors even for a finite 3x3 pencil with identity B. The public
four-type reproducer fails on unmodified 19053d8 and succeeds with the candidate.

Reuse existing arithmetic at the owning rotation layer. For real scalars,
`JacobiRotation::make_givens` computes the coefficients from bounded ratios;
adjust its sign convention to keep the cosine nonnegative. Compute the returned
eliminated entry by dividing by the dominant coefficient, whose magnitude is
at least 1/sqrt(2). This avoids both tiny reciprocals and individually rounded
subnormal products. For complex scalars, reuse the scaled complex `rotg` and
convert its sine convention explicitly. Keep the g=0 identity/returned-f branch.
No per-call allocation, dynamic dispatch or matrix-size loop is introduced.

The AED local Sylvester solve owns four usize permutation arrays. Its query
listed one. [#260](https://github.com/tensor4all/faer/issues/260) independently
reproduces this at n=18/blocking_threshold=15 after #253 plus the exact #255
sentinel correction: 2,112 queried bytes panic requesting a 24-byte LU record.
Changing only the array count gives 2,304 bytes and completes QZ; its retained
numerical assertion still fails at ratio 61.65 (tracked with #257). Reserve
the actual four arrays rather than depending on padding or
another scratch branch accidentally covering the shortfall. A child-module
regression executes a full 2x2/2x2 swap with the AED query's workspace and checks
both pencil reconstructions.

## Historical evidence

The results below describe the original candidate and its recorded baseline,
not a rerun of all checks on current main. Current-base evidence is separated
below.

Four public finite-pencil tests cover f32/f64/c32/c64, subnormals, zero/nonzero
components, complex phase, signs and large finite entries. They verify finite
factors, Hessenberg/triangular structure, Q/Z unitarity and both reconstructions.
Large inputs are normalized componentwise before reconstruction checks to avoid
vacuous overflowing tolerances.

Exact public-query QZ tests cover real single/double precision, n=18/78/96,
default/maximum valid windows, both/no vector accumulators and a dedicated
2-thread token. Outputs start as NaN sentinels, and final real blocks do not
overlap. Independent block polynomials check homogeneous values and finite
spectrum for nonsingular B. The n=258 recursive sweep is opt-in, is executed
locally, and is wired into both scheduled numerical workflows; its runtime
belongs outside the default PR test budget.

The related library selection passes 56 tests, including the bounded 1,520-case
QZ reorder family. Existing public generalized Schur contract/allocation tests
also pass. Final compile/format and stacked-base checks are recorded in the PR.
The original n=96 workspace MWE completes with 40,960 queried bytes after the
query correction, versus its original 40,448-byte allocation panic. Its debug
execution took minutes; that is a local runtime observation, not a speed claim.

## Remaining limits

This is a range/bounds correction, not a proof of all QZ numerical accuracy.
[#257](https://github.com/tensor4all/faer/issues/257) remains open: the n=78
maximum-window fixture has a DGET54-style coupled residual ratio 436.6 on
the recorded baseline `19053d8` versus 0.7724 from LAPACK. The local range-safety candidate
reduces the ratio to 178.6 but does not meet the diagnostic threshold 20.
No tolerance is raised and no full-window accuracy fix is claimed here.

There is no performance adoption or benchmark claim. Existing Jacobi arithmetic
is reused; real/complex routes specialize at compile time. Hosted CI still owns
the full feature/MSRV/no_std matrix and coverage. The local checks are scoped to
the changed generalized-Hessenberg/QZ paths; unrelated numerical families are
unchanged. Colmat retains its current wrapper correction until merged upstream
fixes and the original reproductions plus wrapper regressions pass.


## Parent regression integration

At the cc317ac checkpoint, #255 added its six-dimension public window-32 regression and
runs both sentinel/query tests in CI. Integration preserves those checks and
this candidate's rotation/workspace checks in the same CI step. Production
`faer/src` and `faer-traits/src` trees remain identical to 12bd4bc; the parent
update changes only tests/workflow. No gate, tolerance or timeout is weakened.

The integrated focused run passes eight tests: four rotation tests, the row-zero
regression, the six-dimension window-32 query regression and both real-precision
exact-query suites. Recursive cases retain their opt-in scheduled coverage; the
unchanged production code was already validated for those cases and on both
aarch64 numerical jobs. The new head still runs its required hosted checks.


## Parent test-runtime adjustment

The later #255 head 68ac9eb reduces its window-32 public regression from
31/32/95/96/97/191 to 31/32/96/97. The original reported n=96 case remains;
n=95 and n=191 are no longer covered by that sweep. This is a reduction in
coverage, inherited from the parent's test-runtime adjustment, rather than a
numerical repair. This candidate's separate maximum/default-window suites at
n=18/78/96 and opt-in recursive n=258 suites remain unchanged; they do not
replace the removed window-32 cases.

The merged parent changes only that test. Production sources, manifests and
Cargo.lock are identical to cc317ac and the previously validated 12bd4bc.
The integrated focused run passes all eight ordinary tests (four scalar-range,
row-zero, four-dimension window-32 and both real-precision exact-query suites).
The two recursive tests retain their scheduled opt-in coverage. Required hosted
CI runs again for this combined head; no performance claim is added.


## Main integration after #255

Maintainers merged #255 as e6bb7be and retargeted this PR to main after its
base branch was removed. The only merge conflict was the CI test command;
the resolved command retains every main regression and both rotation/workspace
suites. The merge's complete tree is identical to 470360d, so the eight local
focused tests and existing numerical evidence apply unchanged. This record is
the only subsequent content change; numerical sources, manifests and Cargo.lock
remain identical to the validated candidate. Required checks run for the new
ancestry/head; no main merge or registry publication is performed here.


## Current main compatibility

The candidate was rebased onto main `e4c6d57e`; main `1be2ace` was later merged
in as `d800f78` (no force push, so reviewed SHAs remain). The sentinel correction and
public QZ regressions are supplied by main; this PR retains only its rotation
and four-array workspace corrections, tests and validation registrations.
The rotation helper and the new public regression sources are unchanged from
`e475baf`; main's existing even-rounded AED workspace assertion is retained.
Historical validation above describes its recorded revisions; current-head
hosted validation must complete again before adoption.

All validation statements in the sections above, including each "eight tests
pass" statement, the residual ratios (436.6, 178.6, 61.65, 0.7724) and the
workspace byte counts (2,112/2,304 and 40,448/40,960), are records from the time
they were written. Their command output is not attached to this PR, and they
were not re-measured for the current head. The next section records what was
re-run for the current head.

## Current-head validation (2026-11-10)

Run locally on the tree of `2c7bf5c` (Darwin arm64, rustc 1.96.0). The commit
after it changes only this record.

- CI focused command, `cargo test -p t4a-faer --no-default-features --features
  "std,linalg" --test lsmr_initial_residual --test generalized_schur_reorder
  --test generalized_schur_reorder_no_alloc --test public_qz_aed --test
  public_qz_scratch --test generalized_hessenberg_rotations --test qz_scratch`:
  43 passed, 0 failed, 2 ignored. The eight focused tests are among them:
  `generalized_hessenberg_rotations` 4/4, `public_qz_aed` 1/1,
  `public_qz_scratch` 1/1 and the exact-query suites in `qz_scratch` 2/2. The
  2 ignored tests are the opt-in recursive n=258 tests, which were not run.
- `cargo test -p t4a-faer --lib givens`: 5 passed (`test_givens` and the four
  `test_givens_subnormal_*` tests added here).
- `cargo test -p t4a-faer --lib qz`: 12 passed, 0 failed, 16 ignored (the
  ignored tests are opt-in LAPACK reorder validation).
- Mutation check. `make_givens` was temporarily changed to return `r = 0`
  whenever both inputs are nonzero subnormals, leaving `c` and `s` unchanged.
  `--lib givens` then failed all four `test_givens_subnormal_*` tests at the
  `c f + s g = r` assertion (f64: error 7.49e-23 > bound 5.29e-23 at `f = g =
  5e-324` after scaling by 2^1000; f32: 2.63e-9 > 1.86e-9). The public
  `generalized_hessenberg_rotations` tests stayed 4/4 green under the same
  mutation, which confirms that their absolute reconstruction bound misses this
  failure. With the mutation reverted, all five `givens` tests pass.
- `cargo clippy -p t4a-faer --all-targets --keep-going -- -D warnings` fails
  identically on `1446210` and on `2c7bf5c`: 2399 errors in the lib and 2632 in
  the lib test target, none in the added lines. CI's blocking clippy step covers
  only `t4a-faer-traits`.

Not re-measured: the residual ratios, the workspace byte counts, the recursive
n=258 tests and hosted CI.

## After merging main `1be2ace` (merge commit `d800f78`)

The only conflict was the focused test line in `ci.yml`: main added
`public_householder_upgrade` (and separate `faer-ffi` / `lu_perf_warn` steps);
the resolution keeps main's line and appends `generalized_hessenberg_rotations`
and `qz_scratch`. Re-run locally on the tree of `d800f78` (Darwin arm64,
rustc 1.96.0); the commit after it changes only this record and PATCHSET.

- The resolved CI focused command (main's line plus the two targets above):
  50 passed, 0 failed, 2 ignored (`generalized_hessenberg_rotations` 4,
  `generalized_schur_reorder` 31, `generalized_schur_reorder_no_alloc` 1,
  `lsmr_initial_residual` 4, `public_householder_upgrade` 6, `public_qz_aed` 1,
  `public_qz_scratch` 1, `qz_scratch` 2 passed / 2 ignored recursive, not run).
- `cargo test -p t4a-faer --lib givens`: 5 passed.
- `cargo test -p t4a-faer --lib qz`: 13 passed, 0 failed, 16 ignored.

The mutation check and the clippy comparison above were not repeated on
`d800f78`.

## After merging main `c820e447` and tightening the subnormal allowance

Main `c820e447` was merged into `fix/qz-safe-rotations` as merge commit
`8d88e34e`. The single conflict was again the focused-test line in `ci.yml`:
main had meanwhile added `--test gevd_no_alloc` to it, `--test evd_perf_warn` to
the `perf-warn` line and a `schur_eigenvectors_team_no_alloc` step. The
resolution keeps all of main's additions and appends this PR's two
registrations, so the focused step is main's line plus
`--test generalized_hessenberg_rotations --test qz_scratch`. `git diff
origin/main -- .github/workflows/ci.yml` shows exactly those two added names,
and the recursive-workspace steps in `lapack-validation{,-aarch64}.yml` are
retained.

The earlier review's NEEDS_CHANGES finding is corrected in the same commit as
this record: the subnormal allowance in `check_subnormal_givens` was
`from_f64::<T::Real>(0.75 * tiny) * &scale`, which rounds `0.75 * tiny` back up
to a full subnormal unit before the exact power-of-two scaling, so the check
admitted one unit rather than the documented 0.75. It is now formed after
scaling, `(from_f64::<T::Real>(tiny) * &scale) * from_f64::<T::Real>(0.75)`,
where every step is exact (f32: 2^-149 * 2^120 = 2^-29, * 0.75 = 3*2^-31; f64:
2^-1074 * 2^1000 = 2^-74, * 0.75 = 3*2^-76). The bound is genuinely tighter: the
f64 image bound in the mutation below drops from 5.29e-23 to 3.97e-23. The doc
comment above the helper now says 0.75 units instead of one.

### Validation at this record's tree (Linux x86_64, rustc 1.96.0, `-j 4`)

`target` was warm from the merged tree; every command below ran on the tree of
`8d88e34e` plus this commit's allowance change.

- Focused command from the PR body (the eight `--test` targets, `--no-default-features
  --features "std,linalg"`): 50 passed, 0 failed, 2 ignored, 2m50s wall.
  `generalized_hessenberg_rotations` 4, `generalized_schur_reorder` 31,
  `generalized_schur_reorder_no_alloc` 1, `lsmr_initial_residual` 4,
  `public_householder_upgrade` 6, `public_qz_aed` 1, `public_qz_scratch` 1,
  `qz_scratch` 2 passed / 2 ignored.
- The same command with `--test gevd_no_alloc` added, i.e. the resolved
  `ci.yml` focused line: the same counts plus `gevd_no_alloc` 1, so 51 passed,
  0 failed, 2 ignored.
- `cargo test -p t4a-faer --test qz_scratch recursive -- --ignored`: **2 passed,
  0 failed, 174.01 s** in-test (4m20s including the one-off test-profile build).
  This is the recursive n=258 AED workspace suite for f32 and f64, which the
  earlier reviews recorded as registered but never executed at this head. It
  passes here.
- `cargo test -p t4a-faer --test qz_scratch dedicated_team`: 1 passed, 23.01 s.
  That test needs `threadteam`, so the `--no-default-features` focused line does
  not compile it and the `recursive` filter above does not select it; it was run
  separately here so every test in the added file has current-head evidence.
- `cargo test -p t4a-faer --lib givens`: 5 passed. `cargo test -p t4a-faer --lib
  qz`: 13 passed, 0 failed, 16 ignored. `cargo build -p t4a-faer`: success
  (1m09s). `rustfmt --check` on the changed file reports no diff in the lines
  this change touches (one pre-existing diff at line 174, from main).

### Mutation check at this record's tree

`make_givens` was changed to return `r = 0` whenever both inputs are nonzero and
below 1e-30, leaving `c` and `s` as computed, then reverted.

- `cargo test -p t4a-faer --lib givens`: all four `test_givens_subnormal_*`
  tests fail, `test_givens` passes. f64: `image_error = 7.486784261149287e-23`
  against `image_bound = 3.9704669402545464e-23` at `f = g = 5e-324`. f32:
  `2.634178e-9` against `1.3969864e-9` at `f = g = 1e-45`. Both complex variants
  fail the same way.
- `cargo test -p t4a-faer --no-default-features --features "std,linalg" --test
  generalized_hessenberg_rotations`: 4 passed under the same mutation, which is
  why the direct helper regression exists.
- With the mutation reverted, `--lib givens` is 5 passed again.

Not re-measured, still historical: the residual ratios (436.6, 178.6, 61.65,
0.7724) and the workspace byte counts (2,112/2,304 and 40,448/40,960). The
clippy comparison from the earlier section was not repeated here; this change
adds no new code path, and hosted CI owns its own matrix.
