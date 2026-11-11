# evo2 evidence: `eigh` against MKL at n = 64..128 (tensor4all/faer#391, increment 4)

The record these files back is [`../../2026-11-12-eigh-vs-mkl-evo2.md`](../../2026-11-12-eigh-vs-mkl-evo2.md).
Nothing here is part of the library: the probe is a standalone binary, and
`instrumentation.patch` is throwaway phase instrumentation applied to a scratch copy of the tree.

## Files

| file | what it is |
| --- | --- |
| `probe/` | the driver: fork and MKL arms in one process, plus the timing window and the accuracy checks |
| `build.sh` | builds the probe against a given checkout (stands outside that checkout's workspace) |
| `run.py` | runs the cells: one process per cell, arms interleaved, idle gate before each process |
| `aggregate391.py` | every table of the record from the raw CSVs (`call`, `stage`, `phase`, `mstage`, `spread`, `pairs`) |
| `instrumentation.patch` | per-(level, phase) timing of the divide-and-conquer and the MKL-free stage marks |
| `run1-call.csv` | table A: whole call, 3 arms x f64/c64 x n = 64/96/128 x 1/4/8T, 5 processes per cell |
| `run1-phases.csv` | tables B and C: the 3 wall-time stages and the phase groups, 5 processes per cell |
| `run1-mstage.csv` | table D: MKL `?sytrd` / `?stedc` on the fork's tridiagonal, 3 processes per cell |
| `run2-width.csv` | table E: the root split at its own width against a serial loop (probe-only hook), 2T and 3T included |
| `run3-minwidth.csv` | table F: a min-width-4 variant of the same gate against it, 2T and 3T included |
| `run4-counts.txt` | table G: raw stdout of the count-only build, one process per cell |

`run2` and `run3` were produced by `run.py`'s loop with two differences, because `run.py` takes
neither: the extra 2T (CPUs 0-1) and 3T (CPUs 0-2) cells, and a per-arm environment variable
(`SEC_FORCE_SERIAL`, `SEC_MIN4`). The record shows that loop. Their variant labels are written as
given on the command line, so `aggregate391.py` reads them like any other arm.

The CSVs are the raw per-process records: one row per `(arm, dtype, n, T, repetition, key)`, with
`key` being `call_mean_ns`, `ns`/`nodes` per (level, phase), `stage_mean_ns.<stage>`,
`residual_neps`, `orth_neps`, `digest`, `mkl_max_threads`, `busy_after`, or `cnt` for the counters.
Lines starting with `#` are the runner's idle-gate log and are not data.

## Protocol

- One cell per process. All arms of a cell are interleaved within a repetition (forward on even
  repetitions, reversed on odd ones), so no arm owns a warm-up or a cool-down of the host.
- `taskset` pins each cell: 1T on CPU 0, 4T on 0-3, 8T on 0-7 (one L3 domain). The harness and
  its idle gate run on CPUs 8-15/24-31, outside the measured domain.
- The environment is stripped of `RAYON_*`, `OMP_*`, `MKL_*`, `GOMP_*`; only
  `OMP_NUM_THREADS=<T>` is set. The MKL arm's effect is read back with `mkl_get_max_threads`.
- Idle gate: no process starts while any of CPUs 0-7/16-23 is above 3% busy over 300 ms. A wait is
  logged, never a rejected attempt.
- Every arm restores its input outside the clock, and reports residual and orthogonality in units
  of `n * eps`, plus an FNV digest of `(s, U)`.

## The probe

```
u391 <fork|mkl> <call|mstage> <f64|c64> <n> <threads> <thr|default> [prime_ms] [window_ms]
```

`fork` calls `self_adjoint_evd` (with `SelfAdjointEvdParams { recursion_threshold: thr }`), `mkl`
calls `LAPACKE_?syevd`, and `mstage` times `LAPACKE_dsytrd` and `LAPACKE_dstedc` separately on the
fork's tridiagonal (f64 only). `prime_ms` runs untimed before the window, so the window measures a
warm process. `threads = 1` runs on `Par::Seq`, otherwise inside `faer::team::with_team`.

`probe/.cargo/config.toml` links the installed MKL 2026.2 (`/opt/intel/oneapi`) with its Intel
OpenMP runtime and an rpath; a different MKL install needs those paths changed.

## Instrumentation

`instrumentation.patch` applies to `4d140a9dd5c7ca6e9d520e426b158432afc5b50e` and adds:

- `faer/src/dc_probe.rs`: process-wide atomics per (recursion level, phase) - wall time, call
  count, allocations - so time spent on any team member is counted, plus the `dc-count` counters
  of the secular equation (`cnt` is a no-op unless that feature is on, which is what keeps the
  timing build free of them);
- `lap` marks in `tridiag_evd.rs` (the merge's phases) and three stage marks in `evd/mod.rs`;
- the two probe-only hooks the A/B tables need: `SEC_FORCE_SERIAL` (run every secular loop
  serial) and `SEC_MIN4` (turn the split off below width 4). Both are environment variables read
  at the top of `secular_width`, and both exist so that one binary can be its own baseline.

The counting build is a third variant:

```
cargo build --release --features instr,count --features faer/dc-count
```

Its counters are indexed by the *thread-local* recursion level, so only the 1T rows are usable
per call; the multi-thread rows of `run4-counts.txt` are kept as raw output, not as evidence.
