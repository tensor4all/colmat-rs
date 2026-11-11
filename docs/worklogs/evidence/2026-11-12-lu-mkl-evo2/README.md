# evo2 evidence: `lu` / `solve` against MKL at n = 128..1024 (tensor4all/faer#123)

The record these files back is [`../../2026-11-12-lu-mkl-evo2.md`](../../2026-11-12-lu-mkl-evo2.md).
Nothing here is part of the library: `probe/` is a standalone binary that only calls `faer`'s
public API (plus the same `lu_in_place` the harness uses, to price the phases of the owning
route), and no file outside this directory is touched.

## Files

| file | what it is |
| --- | --- |
| `probe/` | the phase decomposition of the owning route: one process per cell, primed, 5 repetitions, time and minor page faults per phase |
| `run_probe.sh` | runs the probe grid (1T/4T/8T x f64/c64 x c/f x n = 128..1024), one cell per process, through `pinned.sh` |
| `run_cells.sh` | runs the campaign's own cells: `fbench`, 1T/4T/8T x n = 128..1024, f64/c64, `lu` and `solve`, rows `faer-fork`/`faer-upstream`/`lapack-mkl`, 3 process sets |
| `run_alloc_diag.sh` | table D: the same cells under four glibc growth policies, plus each faer row alone in its own process |
| `aggregate_lu123.py` | tables A and B (and the whole-grid medians) from `cells/`; nothing derived is committed, so the command reproduces the tables |
| `aggregate_probe.py` | tables C and E from `probe-runs/` |
| `aggregate_alloc.py` | table D from `alloc/` |
| `cells/cells.csv` | the campaign cells: one row per case and route, 3 processes per cell, samples included; `set` names the process set |
| `probe-runs/probe-phases.csv` | the phase decomposition, one row per phase per cell (cell = `threads`/`dtype`/`order`/`n`), 5 repetitions per phase |
| `alloc/alloc.csv` | the allocator diagnostics. `run` names the process: `alloc-policy-<policy>-<dtype>` is one `fbench` process per glibc growth policy with the rows `faer-fork`+`lapack-mkl`, `toppad-faer-upstream-<dtype>` the control row under the same padding, `row-isolated-set<N>-<row>-<dtype>` one arm alone in its own process (`lu` only) |

The probe's `owning` and `owning_no_split` reproduce the campaign's two `fbench` rows to within
~3% (1.545 against 1.580 and 0.724 against 0.703 at f64 n = 512 8T), which is what says the
decomposition is of the same code; the probe hoists the permutation vectors while `new_imp`
allocates two 8-KiB `Vec`s per call, so those are inside the campaign row and outside the probe's
phases.

The three CSVs are the runners' per-cell files concatenated, which is why each carries a column for
the run it came from; nothing else was changed, so re-deriving them from the runners and re-merging
reproduces the tables:

```
# cells.csv: 'set' is the process set, '#' guard lines dropped (the record treats them as not data)
{ echo "set,$(head -1 cells/run1-1t-n128.csv)"
  for f in $(ls cells/run*.csv | sort); do
    set=$(basename "$f" | sed 's/^run\([0-9]*\)-.*/\1/')
    grep -v '^#' "$f" | awk -v s="$set" 'NR>1{print s","$0}'
  done; } > cells/cells.csv
# probe-phases.csv: the probe's own rows already identify their cell
{ head -1 probe-runs/probe-1t-c64-c-n1024.csv
  for f in $(ls probe-runs/probe-*.csv | sort); do tail -n +2 "$f"; done; } > probe-runs/probe-phases.csv
# alloc.csv: 'run' is the per-cell file's name
{ echo "run,$(head -1 alloc/alloc-policy-base-f64.csv)"
  for f in $(ls alloc/*.csv | sort); do
    r=$(basename "$f" .csv); grep -v '^#' "$f" | awk -v r="$r" 'NR>1{print r","$0}'
  done; } > alloc/alloc.csv
```

Everything except the concatenation is the runners' own output. The `#` lines the merging drops
were `pinned.sh`'s idle-gate log, which `fbench` copies into its own file; the probe writes its
measurements only, so `probe-runs/` carries no guard log - those verdicts were printed by
`run_probe.sh` during the session (48 cells, all accepted on attempt 1) and are not part of these
files.

## Protocol

The full statement is in the record's "Verification conclusions and constraints"; what a reader of
these CSVs needs:

- one cell per process; in `cells/` the rows of a cell are interleaved inside every repetition
  (`--rotate-rows 3`), in `probe-runs/` each phase is primed 250 ms and then timed 5 times on the
  same buffers and the same thread budget;
- `taskset` 1T = CPU 0, 4T = 0-3, 8T = 0-7 (one L3 domain of this host), harness and idle gate on
  CPUs 8-15/24-31, outside the measured domain;
- `benchmarks/scripts/pinned.sh`: 3 s idle window, 5% busy, 3 retries, verdict recorded per
  process; every process in every run here was accepted on attempt 1;
- the campaign's `zen5-cpu` profile is a different machine (24 logical / 12 cores) and
  `record_run.py` refuses this host, so the CPU sets are the ones
  `docs/worklogs/2026-11-12-eigh-vs-mkl-evo2.md` fixed for evo2.

## Revisions

- fork arm: `c820e447` (`t4a-faer` 0.32.0, `origin/main`), the checkout the probe's
  `Cargo.toml` points at (`../../../../../faer`)
- upstream control row: `0539947f` (`faer-v0.24.4`), built by the harness from
  `extern/upstream-faer`
- harness: `tensor4all/faer-benchmark` `3b80b90`; MKL 2026.2 from `/opt/intel/oneapi`

## The probe

```
lu123 --threads T --dtype f64|c64 --order c|f --n N --reps 5 --prime-ms 250 --csv OUT
```

Every phase runs `--reps` times after the prime, on the same input and the same thread budget, and
records its median wall time and the minor page faults it took (`/proc/self/stat`, field 10). The
phases are the ones table C names.

`split_std_threads` spawns its own threads (`std::thread::scope`) and exists only to answer
"would a parallel split pay?"; a library change would have to use faer's own team, so that phase
is a bound, not a proposal.

The probe needs nothing beyond the checkout: its `Cargo.toml` declares an empty `[workspace]` so it
does not join faer's, and its only dependency is `faer` by path (`../../../../../faer`, the checkout
it sits in).
