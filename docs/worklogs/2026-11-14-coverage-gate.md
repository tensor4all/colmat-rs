# Coverage gate for the workspace

## Decisions

- The floor is the measured total line coverage of the whole workspace, taken below the third decimal
  (`82.694 %`, 79,272 of 95,861 lines), not a rounded or aspirational number. At an otherwise
  unchanged total the gate is one-line sensitive: a line that stops being covered, or a new line no
  test executes, trips it. There is no margin, because the measurement repeated exactly.
- The report is published to the job summary as well as the log, so a red check shows which file moved
  without a rerun. Uncovered line numbers stay out of CI (16,589 of them at the baseline) and are a
  local rerun with `--show-missing-lines`, which `docs/coverage.md` records.
- The number was measured on the development host at 32 threads and re-checked on four CPUs, because
  the hosted runner has four. Coverage counts are integers, so the check is decided by the ratio, not
  by float ordering or host speed; a test that branched on thread count would have changed the counts,
  and none does (no test reads `available_parallelism`).
- The gate lives in `ci.yml` with its toolchain pinned twice: the action installs 1.96.0 with
  `llvm-tools-preview`, and the job also sets `RUSTUP_TOOLCHAIN`, which is what `rust-toolchain.toml`
  would otherwise override. Coverage counts move with the compiler, so the job must not follow a
  channel bump silently; a bumped pin re-measures the floor in the same change. cargo-llvm-cov and
  cargo-nextest are pinned versions, the actions that install them are not.
- Workspace and all targets, with no exclusions. `faer-macros` is reported at 0 % because proc-macro
  code runs inside rustc; it stays in the total as a constant offset instead of being hidden behind an
  ignore regex. `faer-ffi` and `faer-traits` are measured by their own tests. Doctests are not in the
  number: nextest does not run them, and `ci.yml` runs them separately.
- The inherited upstream codecov upload stays untouched. It is a second, non-gating coverage
  measurement covering `faer` alone on the moving `stable` toolchain, so it cannot be the floor.
  Retiring it is its own change, not a hunk in this one.
- `README.md` is not touched: the README-badge change is in flight in another pull request (it adds
  the `ci.yml` badge, which covers this job; `README.md` has no CI badge today), and this change needs
  no badge of its own.
- No `PATCHSET.md` row: this is CI infrastructure, and the ledger tracks the library's delta from
  upstream.
- The design was reviewed before implementation by an independent model
  (`ollama-cloud/glm-5.3-flash`); its findings are why the toolchain pin is explicit
  (`RUSTUP_TOOLCHAIN`) rather than two pins coinciding, why the floor is `82.694` and not the printed
  `82.69`, and why the timeout is 90 minutes.

## Verification conclusions and constraints

- Two full runs gave identical integer counts, so the floor is reproducible on a fixed toolchain and
  the same target set. Run time is dominated by a handful of long numerical tests, not by the
  instrumented build.
- The threshold is the JSON total's line ratio (`covered * 100 / count`), and cargo-llvm-cov signals a
  trip only through its exit status: the report table and its `TOTAL` row are printed first, and no
  message follows. A red run is read by comparing that row's `Lines` cover with the floor in the
  workflow.
- A line floor cannot detect churn: deleting an `#[ignore]`d test raises the ratio, and a change that
  adds covered lines creates headroom below the floor. That is a known, documented limit, not a claim
  of test-count protection.
- Only one instrumented run can own the llvm-cov target directory at a time; probing the threshold
  while a run is in flight reads partially rewritten profile data and gives wrong boundaries.
- Making the check *required* is not part of this change and cannot be: the repository has no
  ruleset and `main` is unprotected, so enabling `coverage` (and the clippy check from the warning
  cleanup) is a settings edit by the maintainer once the job has existed and failed correctly.
- Unverified: the hosted run time under the 90-minute job timeout, and whether the first hosted run
  reproduces the four-CPU counts. Both are settled by the first run of the job itself.
