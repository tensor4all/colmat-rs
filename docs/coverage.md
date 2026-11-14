# Coverage gate

`.github/workflows/ci.yml` runs one coverage job (`coverage`) that measures the whole workspace and
fails below a checked-in floor. It is the only coverage *gate* in CI; the inherited upstream codecov
upload in `run-tests.yml` is non-gating and covers `faer` alone (retiring it is an open follow-up, see
`docs/worklogs/2026-11-14-coverage-gate.md`).

## The number

Measured 2026-11-14 on an x86_64 32-thread host with cargo 1.96.0, cargo-llvm-cov 0.9.1 and
cargo-nextest 0.9.151, twice with identical integer counts and once on 4 CPUs:

| metric | value |
| --- | --- |
| lines | 79,272 / 95,861 = **82.694 %** (the floor: `--fail-under-lines 82.694`) |
| functions | 73.61 % |
| regions | 85.21 % |
| tests | 1089 run, 166 `#[ignore]`d |

The floor is the measurement, not a target: it is the measured ratio below the third decimal, so at an
otherwise unchanged total, one line that stops being covered — or one new line that no test executes —
fails the job. It is an aggregate, so a change that also adds covered lines can raise the ratio past
one new uncovered line, and a coverage improvement creates headroom below the checked-in floor until
someone raises it. Raise the floor when coverage improves. Lowering it needs a maintainer's decision
stated in the PR, and changing the toolchain pin in the job means re-measuring it in the same PR.

## Reproduce it locally

```sh
rustup component add llvm-tools-preview --toolchain 1.96.0   # required by cargo-llvm-cov
cargo install cargo-llvm-cov --version 0.9.1 --locked
cargo install cargo-nextest --version 0.9.151 --locked      # `cargo llvm-cov nextest` calls it
RUSTUP_TOOLCHAIN=1.96.0 cargo llvm-cov nextest --workspace --all-targets --locked \
    --fail-under-lines 82.694
```

The text report lists every file with its `Lines` / `Missed Lines` and a `TOTAL` row; the exit status
is non-zero when the total is below the floor. The job publishes the same report to its summary as
well as to the log, and cargo-llvm-cov prints no message of its own when the floor trips, so a red run
is read from the `TOTAL` row. For the uncovered lines themselves add `--show-missing-lines` (16,589
lines on the baseline, so keep it out of CI), or generate `--lcov --output-path lcov.info` / `--html`
and open `target/llvm-cov/html/index.html`; the lcov and html formats replace the text table rather
than adding to it.

Only one instrumented run can own `target/llvm-cov-target` at a time: a second `cargo llvm-cov` run
started while one is measuring rewrites the profile data underneath it, and the numbers it reports are
not the ones either run measured.

## What is not measured, and why

- **Doctests.** nextest does not run them, and they are not in the number; `ci.yml` runs them
  separately.
- **`faer-macros`** reads 0 %: proc-macro code runs inside rustc, not in the test process, so no test
  can cover it. It stays in the total rather than being hidden behind an `--ignore-filename-regex`, and
  it is a constant offset (~251 lines). `faer-ffi` (47.94 %) and `faer-traits` (67.80 %) are measured by
  their own tests; no crate is excluded.
- **Examples and benches** are built by `--all-targets` but not run.
- **`#[ignore]` tests** do not run, so the code only they reach counts as uncovered. Their number can
  go down without changing the ratio: deleting one raises coverage, which a line floor cannot tell
  apart from an improvement — the floor is not churn detection.
- **Non-Linux / non-x86_64 hosts.** Like the rest of the fork's PR lanes, the gate is
  `ubuntu-latest` x86_64.
