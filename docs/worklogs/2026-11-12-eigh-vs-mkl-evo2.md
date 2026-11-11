# `eigh` at n = 64..128 on evo2 against MKL: the 8T gap is gone, the 1T one is the secular solve

Issue: [tensor4all/faer#391](https://github.com/tensor4all/faer/issues/391), increment 4.

Measured revision: `4d140a9dd5c7ca6e9d520e426b158432afc5b50e` for **both** arms, clean tree,
built `--no-default-features --features std,rayon,threadteam` (the campaign's feature set; the
instrumented copy adds only the phase probe). MKL 2026.2 (`/opt/intel/oneapi/mkl/2026.2`,
`libmkl_rt.so.3`, Intel OpenMP `iomp5`). One host, one revision, one session: **EVO-X2**, AMD
Ryzen AI MAX+ 395 (Zen 5), 32 logical / 16 cores, SMT on (sibling `n + 16`), two 64-MiB L3
domains (CPUs 0-7/16-23 and 8-15/24-31), `performance` governor, boost on, no
`x86-v4`/`nightly`. Nothing here is comparable with the MKL `?stedc` figure of the 2026-11-11
records: that one came from another host and another fork revision.

Every number below is recomputed by
[`evidence/2026-11-12-eigh-mkl-evo2/aggregate391.py`](evidence/2026-11-12-eigh-mkl-evo2/aggregate391.py)
from the raw CSVs in that directory (one row per process per key, 5 processes per cell for the
call and the phases, 3 for the MKL stages). The script prints the number of processes it kept
per cell and the spread; it never assumes either.

## Decisions

- **The 8T premise does not hold on this host: there is no 8T gap to close.** With the
  divide-and-conquer reachable (`recursion_threshold` 16), f64 n = 64/96/128 at 8T is
  **0.94 / 0.92 / 0.95x** MKL's `?syevd`, and c64 is 0.65 / 0.65 / 0.74x at 8T and
  0.65-0.81x at *every* thread count (table A). The rotations are not the 8T lever: at 8T
  f64 n = 128 the whole (c) group (Löwner + eigenvector columns + Householder apply) is 90.6 us
  of a 705 us call, while the secular root finder is 171.5 us (table C). Chasing the rotations
  at 8T would optimise the third-largest phase of that call - after the secular solve and the
  leaf QR - which is already ahead of MKL.
- **The gap that does exist is 1T and 4T (+10..28% and +12..18%), and it is one stage.** At 1T
  f64 n = 128, `tridiagonal_self_adjoint_evd` is 1025.1 us against MKL's `?stedc` 602.4
  (**+422.7 us, 1.70x**), while the tridiagonalization is at parity with `?sytrd` (235.5 vs
  219.6) and the reflector apply costs 104.5 (table B). The whole call is only 118.2 us slower,
  because MKL's own `?syevd` spends 377.7 us outside `?sytrd + ?stedc` at 1T while faer's apply
  is 104.5 - the same structure the 2026-11-10 record found, and the reason a stage-level
  comparison overstates the whole-call gap. At 4T the same split holds: +268.8 us in stage 2,
  offset by MKL's remainder shrinking from 377.7 to 105.8.
- **Inside stage 2, the secular root finder is the largest group at 1T and it is at the pass
  count its algorithm dictates** (table C, table G). At 1T f64 n = 128 level 0 it is 378.0 us
  of the 1025.1, 37% of the stage and 28% of the call; the next is (c) at 90.0. The count build
  gives, per call, 128 roots, 4.92 evaluation loops per root - two of them the 4-point
  bracketing ladders of `secular_eq_root_finder` (`faer/src/linalg/svd/bidiag_svd.rs:10`), the
  rest secant steps - plus bisection for 19.5% of roots at 4.3 evaluations each. So the "fewer
  passes per root" direction has about one removable pass in five (the duplicated `f_mid` of the
  first ladder at `bidiag_svd.rs:23` and `:171`), and each pass is one division and one multiply
  per pole: 378.0 us / (128 roots x 4.92 passes x 128 poles) = **4.7 ns per pole-evaluation**.
  Reassociating that sum into four accumulators was implemented and measured - 1319.1/1324.6 us
  against 1319.1/1317.7 us of the baseline over two rounds at 1T f64 n = 128 - and changed
  nothing, because the sum is not the cost. A vectorised residual cannot be reached from the
  safe API either: `SimdCtx` has no `div`, and the residue is `z * (z / x)` per pole. **This
  increment therefore does not take the 1T root-finder direction**: the measured headroom is one
  pass in five, not a factor.
- **The largest measured lever is a default, not a kernel** (table A, columns 4 and 6).
  `SelfAdjointEvdParams::recursion_threshold` is 128 (`faer/src/linalg/evd/mod.rs:91`), the
  entry gate is `n < params.recursion_threshold -> qr_algorithm` (`mod.rs:274`), and the same
  value is handed to the divide-and-conquer as its fallback threshold (`mod.rs:287` ->
  `tridiag_evd.rs:389-390`, `n <= qr_fallback_threshold`). So **every n <= 128 call of the
  published `eigh` runs the QR algorithm and never enters the divide-and-conquer**, at any thread
  count; that is what the "fork default" column measures, and it is 1.24-2.18x behind MKL at
  4T/8T while the same binary at 16 is 0.92-1.28x. Changing it is a numerics decision - a
  different algorithm on the default path for a whole size band - so this record measures it and
  leaves the decision; no library code is changed here.
- **The #397 root split is mis-calibrated at width 2, and this is a separate defect** (tables E
  and F). Forcing the loop serial (`SEC_FORCE_SERIAL`, a probe-only hook) prices the split
  itself: at 1T it is free, at 4T it buys 3% of the call, at 8T 27% - but at **2T it costs 22%**
  of the call, because the level-0 secular phase goes from 381.1 to 617.2 us (+62%) on two
  participants. `secular_width` (`tridiag_evd.rs:273`) admits that width from
  `cost::par_width_upto` of `k^2` units at `rate::secular_roots` (35.2 ns per `k^2`), so the
  model prices k = 128 as 577 us of serial work and a section of 2 as paying. At 4T the same effect appears one level
  down: level-1 (a) is 317.4 us there against 208.5 at 1T, because each level-1 merge's sub-team
  is two threads wide (table C). The obvious fix - never split below width 4 - was measured:
  it wins 14-17% at 2T and **loses 1.8-7% at 4T and 8T for n = 64/96** (table F), because the
  deeper levels' width-2 and width-3 splits do pay for their `k`. So the widths the gate picks at
  4T/8T are right, and what needs explaining is why k = 128 at width 2 is 62% slower than
  serial. This record does not explain it and does not change the gate.

## What this record does not do

No library change, no version bump, no API change: the measurement contradicts the premise the
next stage was to be chosen from, so this is the record and the decision points, not the change.

The measured next stages, in the order the numbers rank them, for the maintainer to choose:

1. **`recursion_threshold` on the default path** (`mod.rs:91`). Effect if set per thread count
   to what table A measures: f64 n = 128 8T 1430.8 -> 624.7 us, 4T 1438.0 -> 822.4, 1T 1485.6 ->
   1315.9; c64 n = 128 8T 1597.8 -> 829.1. That is 2.3x at 8T and the whole published gap
   against MKL in the measured band. Risk: the D&C becomes the algorithm of record below 129, so
   its accuracy and its determinism across thread counts become the public behaviour - table B/
   the residuals below say the D&C is *more* accurate here (0.164 against 0.461 n-eps at f64
   n = 128) but its output differs between 1T/4T and 8T, which the QR path's does not.
2. **the width-2 behaviour of the #397 split** (`tridiag_evd.rs:273`): worth 22% of the call at
   2T and, through the level-1 sub-teams, part of the 4T cell that is the worst one against MKL
   (1.18x). Needs a mechanism before a gate tweak: the min-width-4 variant is measured and is
   not it.
3. **the 1T secular root finder**: one removable pass in five (~7% of the stage, ~5% of the
   call), or a vectorised residual, which needs a `div` in `traits::SimdCtx` - a trait change,
   not a kernel change.

## Verification conclusions and constraints

- **Both arms are the same revision and the same host**, in one session, one cell per process,
  arms interleaved per repetition (forward on even repetitions, reversed on odd). The report
  process and its idle gate run on CPUs 8-15/24-31 (`taskset -apc 8-15,24-31` on the harness
  pid), so the observer is outside the two L3 domains that contain the measured CPUs; 1T runs on
  CPU 0, 4T on 0-3, 8T on 0-7 - one L3 domain (CPUs 0-7 share `index3/shared_cpu_list`), so a
  cell never spans both domains. `RAYON_*`/`OMP_*`/`MKL_*`/`GOMP_*` are stripped from the child's
  environment and only `OMP_NUM_THREADS=<T>` is set; the MKL arm's own thread count is read back
  through `mkl_get_max_threads` and is 1/4/8 in every cell.
- **Idle gate**: `< 3%` busy over 300 ms on CPUs 0-7 and 16-23 before every process; a cell that
  had to wait is logged with `busy_after` in the CSV and never retried until it looks good.
  Run 2 waited 39 times and run 3 100 times for the host to go quiet; the spreads of the kept
  processes are 0.2-0.7% (table A, and `aggregate391.py spread`), which is what says the gate did
  its job.
- **Nothing is claimed about the output of the two algorithms being equal.** The probe prints an
  FNV digest of `(s, U)`. Within a cell it is one value across all 5 processes; across thread
  counts it is not (`f64 n = 128`: 1T and 4T share a digest, 8T differs), so the D&C path is
  deterministic per thread count only. Residuals are reported in units of `n * eps * max|A|` and
  are 0.16-0.48 on the D&C path and 0.42-0.75 on the QR path (f64 and c64, all cells), i.e. a
  few n-eps either way.
- **The gate is measured before each process, not after it.** The CSV's `busy_after` column is
  the same measurement taken after the process exited and reaches 13% in this session (other work
  on the host); it is recorded and not used to select, because selection happened before the
  process, on the same CPUs.
- The prototype process measured itself: no other measurement ran on the host during these runs,
  and the A/B experiments (tables E and F) were run in one session, alternately, on the same
  binary pair.

## Evidence

### A. the whole call, us - `aggregate391.py call run1-call.csv`

"fork default" is `eigh` with the crate's own `SelfAdjointEvdParams`; "fork thr=16" is
`self_adjoint_evd` with `recursion_threshold` 16, which is what makes the divide-and-conquer
reachable at these sizes at all. MKL is `LAPACKE_?syevd` (column major, jobz V, uplo L) with the
input restored outside the clock.

| dtype | n | T | fork default | fork thr=16 | MKL ?syevd | thr16 / MKL | default / MKL | procs | spread (fork thr=16) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 1 | 318.745 | 317.165 | 253.376 | 1.25 | 1.26 | 5 | 0.5% |
| f64 | 64 | 4 | 319.212 | 194.133 | 174.012 | 1.12 | 1.83 | 5 | 0.4% |
| f64 | 64 | 8 | 320.436 | 157.780 | 168.528 | 0.94 | 1.90 | 5 | 0.4% |
| f64 | 96 | 1 | 774.662 | 737.161 | 576.470 | 1.28 | 1.34 | 5 | 0.2% |
| f64 | 96 | 4 | 761.723 | 428.656 | 379.385 | 1.13 | 2.01 | 5 | 0.4% |
| f64 | 96 | 8 | 765.577 | 339.844 | 369.466 | 0.92 | 2.07 | 5 | 0.6% |
| f64 | 128 | 1 | 1485.586 | 1315.887 | 1197.677 | 1.10 | 1.24 | 5 | 0.5% |
| f64 | 128 | 4 | 1437.987 | 822.394 | 699.398 | 1.18 | 2.06 | 5 | 0.3% |
| f64 | 128 | 8 | 1430.830 | 624.719 | 656.665 | 0.95 | 2.18 | 5 | 0.6% |
| c64 | 64 | 1 | 402.472 | 414.599 | 510.056 | 0.81 | 0.79 | 5 | 0.5% |
| c64 | 64 | 4 | 366.016 | 254.252 | 347.580 | 0.73 | 1.05 | 5 | 0.4% |
| c64 | 64 | 8 | 365.654 | 214.515 | 330.308 | 0.65 | 1.11 | 5 | 2.4% |
| c64 | 96 | 1 | 992.683 | 971.675 | 1322.942 | 0.73 | 0.75 | 5 | 0.2% |
| c64 | 96 | 4 | 865.746 | 556.208 | 763.602 | 0.73 | 1.13 | 5 | 0.7% |
| c64 | 96 | 8 | 853.576 | 447.097 | 690.501 | 0.65 | 1.24 | 5 | 0.9% |
| c64 | 128 | 1 | 1996.484 | 1856.686 | 2850.840 | 0.65 | 0.70 | 5 | 0.8% |
| c64 | 128 | 4 | 1643.865 | 1063.355 | 1336.970 | 0.80 | 1.23 | 5 | 0.3% |
| c64 | 128 | 8 | 1597.775 | 829.138 | 1123.351 | 0.74 | 1.42 | 5 | 0.6% |

c64 n = 64 at 8T is the one cell above 1% (2.4% for the fork arm, 0.6% for MKL); every other
cell and arm is at 1.0% or below, and `aggregate391.py spread run1-call.csv` prints the count,
min, median, max and spread of each cell and arm separately - the ratios above are between two
medians, so both spreads matter.

### B. the three wall-time stages of the call, us - `aggregate391.py stage run1-phases.csv`

Marks inside `self_adjoint_evd` around `tridiag_in_place`, `tridiagonal_self_adjoint_evd` and the
block-Householder apply; the three sum to the call within 0.5% (last two columns).

| dtype | n | T | variant | tridiagonalize | tridiag EVD | apply | sum | call |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| f64 | 64 | 1 | fork16i | 56.045 | 253.130 | 21.016 | 330.191 | 330.498 |
| f64 | 64 | 1 | forkdefi | 56.395 | 250.603 | 21.109 | 328.108 | 328.309 |
| f64 | 64 | 4 | fork16i | 57.736 | 139.081 | 17.961 | 214.778 | 215.011 |
| f64 | 64 | 8 | fork16i | 56.928 | 95.686 | 18.383 | 170.997 | 171.581 |
| f64 | 96 | 8 | fork16i | 132.978 | 207.865 | 32.896 | 373.740 | 374.155 |
| f64 | 128 | 1 | fork16i | 235.539 | 1025.093 | 104.533 | 1365.165 | 1365.685 |
| f64 | 128 | 1 | forkdefi | 235.779 | 1178.701 | 103.509 | 1517.989 | 1518.256 |
| f64 | 128 | 4 | fork16i | 223.973 | 652.538 | 63.641 | 940.153 | 939.039 |
| f64 | 128 | 8 | fork16i | 232.327 | 424.660 | 50.635 | 707.622 | 707.596 |
| c64 | 128 | 1 | fork16i | 483.926 | 1033.896 | 387.190 | 1905.012 | 1905.062 |
| c64 | 128 | 4 | fork16i | 319.013 | 649.880 | 186.482 | 1155.374 | 1156.632 |
| c64 | 128 | 8 | fork16i | 313.451 | 432.343 | 143.371 | 889.166 | 888.901 |

`forkdefi` at n = 128 is the QR path: its "tridiag EVD" is 1178-1189 us at *every* thread count,
because `qr_algorithm` takes no `Par` - the whole reason a per-call default is worth more than
any kernel in this band.

### C. the phase groups of #393/#397, us - `aggregate391.py phase run1-phases.csv --by-level`

Groups unchanged from `2026-11-11-eigh-dc-merge-phases.md`: (a) secular root finder, (b) z build +
merge perm + deflate + sort, (c) Löwner + eigenvector columns + Householder apply, (d) the two
matmuls, (d') copy-out, (e) leaf QR. f64 n = 128, `fork16i`:

| level | T | (a) | (b) | (c) | (d) | (d') | (e) | nodes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 377.985 | 1.572 | 90.005 | 24.206 | 11.127 | - | 1.00 |
| 1 | 1 | 208.538 | 1.640 | 43.444 | 7.070 | 5.699 | - | 2.00 |
| 2 | 1 | 116.909 | 2.005 | 21.437 | 2.740 | 2.988 | - | 4.00 |
| 3 | 1 | - | - | - | - | - | 100.811 | - |
| 0 | 4 | 283.532 | 1.744 | 92.397 | 9.222 | 11.206 | - | 1.00 |
| 1 | 4 | 317.403 | 2.024 | 44.037 | 5.057 | 5.836 | - | 2.00 |
| 2 | 4 | 118.689 | 2.453 | 22.179 | 2.841 | 3.020 | - | 4.00 |
| 3 | 4 | - | - | - | - | - | 100.054 | - |
| 0 | 8 | 171.453 | 1.763 | 90.585 | 7.540 | 11.220 | - | 1.00 |
| 1 | 8 | 144.709 | 2.022 | 44.076 | 5.265 | 5.861 | - | 2.00 |
| 2 | 8 | 60.278 | 2.800 | 22.385 | 3.035 | 3.039 | - | 4.00 |
| 3 | 8 | - | - | - | - | - | 100.755 | - |

Three things this table says. The leaf QR at the bottom is thread-count independent (100.8 us at
1T and at 8T) because 8 leaves already run on 8 threads at 8T; it is 10% of the 1T stage 2 and is
not the gap against `?stedc`. (c) is thread-count independent too (155-158 us summed over levels
at every T) - it is the phase a rotation-level rewrite would address, and at 8T it is an eighth
of the call. And level-1 (a) *rises* from 208.5 at 1T to 317.4 at 4T, which is the width-2
sub-team effect of table E, not noise (spread 0.3%): the per-level sums are thread-time, so above
1T they must be read as total work, not as latency - `aggregate391.py phase` prints their ratio
to the call in its `identity` column (0.75 at 1T, 1.09 at 4T, 0.96 at 8T) rather than pretending
the sum is the call.

### D. MKL's own stages against the fork's, us - `aggregate391.py mstage run1-mstage.csv ...`

`LAPACKE_dsytrd` and `LAPACKE_dstedc` (compz 'I') called on the fork's own tridiagonal output, so
the second column is MKL's divide-and-conquer on the same input the fork's stage 2 gets. f64:

| n | T | MKL ?sytrd | MKL ?stedc | fork stage 1 | fork stage 2 | fork apply | MKL ?syevd | fork call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 35.818 | 160.915 | 56.045 | 253.130 | 21.016 | 253.422 | 317.165 |
| 64 | 4 | 36.370 | 110.057 | 57.736 | 139.081 | 17.961 | 173.900 | 194.133 |
| 64 | 8 | 35.662 | 109.385 | 56.928 | 95.686 | 18.383 | 168.483 | 157.780 |
| 96 | 1 | 100.676 | 334.640 | 126.278 | 589.313 | 52.269 | 576.442 | 737.161 |
| 96 | 4 | 105.275 | 216.076 | 129.417 | 313.645 | 33.840 | 379.275 | 428.656 |
| 96 | 8 | 111.889 | 214.405 | 132.978 | 207.865 | 32.896 | 369.029 | 339.844 |
| 128 | 1 | 219.640 | 602.369 | 235.539 | 1025.093 | 104.533 | 1199.682 | 1315.887 |
| 128 | 4 | 209.354 | 383.706 | 223.973 | 652.538 | 63.641 | 698.901 | 822.394 |
| 128 | 8 | 221.697 | 365.697 | 232.327 | 424.660 | 50.635 | 656.421 | 624.719 |

The gaps, as differences rather than ratios, stage 2 against `?stedc` (rows n = 64, 96, 128; T =
1, 4, 8): **+92.2 / +29.0 / -13.7 / +254.7 / +97.6 / -6.5 / +422.7 / +268.8 / +59.0** us. Above
1T the fork's stage 2 is no longer behind MKL's at n = 64 or 96 - it is ahead at 8T - so the
1T column is where the stage-level loss is. Stage 1 against `?sytrd` is +20.2 / +21.4 / +21.3 /
+25.6 / +24.1 / +21.1 / +15.9 / +14.6 / +10.6: ~20 us at 1T, i.e. not a scaling problem either.
What keeps the *call* near parity at 4T and ahead at 8T is the column after `?stedc`:
`?syevd - ?sytrd - ?stedc` is 377.7 / 105.8 / 69.0 us at n = 128, MKL's unattributed remainder,
against the fork's apply of 104.5 / 63.6 / 50.6. As in the 2026-11-10 record, that remainder is
not attributed to a phase of MKL here, and it is not called the reflector apply.

### E. what the #397 root split itself is worth, us - `aggregate391.py pairs run2-width.csv split serial`

f64 n = 128, `recursion_threshold` 16, one binary; the serial arm forces every secular loop
serial through a probe-only hook, so both arms differ only in the width of that loop.

| T | split call | serial call | serial / split | split secular | serial secular | procs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1364.613 | 1365.057 | 1.000 | 378.2 | 378.2 | 5 |
| 2 | 1316.214 | 1079.879 | 0.820 | 617.2 | 381.1 | 5 |
| 3 | 1090.932 | 1052.324 | 0.965 | 414.0 | 377.1 | 5 |
| 4 | 935.891 | 962.850 | 1.029 | 283.7 | 364.9 | 5 |
| 8 | 705.354 | 969.125 | 1.374 | 171.4 | 381.3 | 5 |

At width 2 the phase is 62% *slower* than serial. Every per-root quantity is unchanged (each
root is independent and its arithmetic is deterministic), so this is the parallel region, not the
root finder, and it is not explained by this record.

### F. the min-width-4 variant, us - `aggregate391.py pairs run3-minwidth.csv base min4`

Same binary pair, `min4` turns the split off whenever the gate asks for fewer than 4
participants.

| n | T | base call | min4 call | min4 / base | base secular | min4 secular |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2 | 315.213 | 270.405 | 0.858 | 145.8 | 100.1 |
| 64 | 3 | 263.497 | 270.320 | 1.026 | 95.9 | 99.8 |
| 64 | 4 | 215.912 | 231.440 | 1.072 | 73.1 | 73.2 |
| 64 | 8 | 171.560 | 174.679 | 1.018 | 37.3 | 37.4 |
| 96 | 2 | 736.578 | 618.826 | 0.840 | 338.0 | 217.1 |
| 96 | 3 | 611.355 | 602.820 | 0.986 | 224.9 | 212.7 |
| 96 | 4 | 478.233 | 509.945 | 1.066 | 163.1 | 163.2 |
| 96 | 8 | 374.034 | 386.352 | 1.033 | 81.8 | 81.9 |
| 128 | 2 | 1320.989 | 1091.936 | 0.827 | 618.8 | 382.6 |
| 128 | 3 | 1091.395 | 1056.288 | 0.968 | 414.7 | 377.8 |
| 128 | 4 | 936.636 | 885.940 | 0.946 | 283.7 | 284.0 |
| 128 | 8 | 706.364 | 725.983 | 1.028 | 171.4 | 171.7 |

The level-0 secular column shows the variant only touches *deeper* levels at n = 64/96 (its 4T/8T
level-0 values are unchanged) - and there it costs 3-7%. So a wider minimum is not the fix for
table E, and the gate's 4T/8T choices are correct as they stand.

### G. what the secular equation does per call, counts - `run4-counts.txt`

Count-only build (`--features instr,count --features faer/dc-count`): the counters are never
timed. f64, `recursion_threshold` 16, prime 250 ms / window 1000 ms, one process per cell, per
call:

| n | T | roots | batches | evaluation loops | bisected roots | bisection evaluations | loops / root |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 1 | 96.00 | 96.00 | 464.00 | 16.00 | 45.00 | 4.83 |
| 128 | 1 | 128.00 | 128.00 | 617.00 - 630.00 | 25.00 | 108.00 | 4.92 |

The T = 4 and T = 8 rows of that file are **not usable**: the counters are indexed by the
thread-local recursion level, so roots solved by a sub-team are counted at another level and no
per-call total can be formed from them. The 1T rows are exact (roots per call = n).

## Reproduction

```
# both arms, from the same revision
git -C <checkout> rev-parse HEAD                 # 4d140a9dd5c7ca6e9d520e426b158432afc5b50e
python3 build.sh <pristine checkout> ~/run pristine             # -> ~/run/u391-pristine
git -C <instrumented copy> apply instrumentation.patch
python3 build.sh <instrumented copy> ~/run instr --instr        # -> ~/run/u391-instr

# tables A, B, C, D: 5 processes per cell (3 for the MKL stages), arms interleaved per process
python3 run.py run1-call.csv   forkdef:default=~/run/u391-pristine \
                               fork16:16=~/run/u391-pristine  mkl=~/run/u391-pristine
python3 run.py run1-phases.csv fork16i:16=~/run/u391-instr forkdefi:default=~/run/u391-instr \
                               mkl=~/run/u391-pristine
python3 run.py run1-mstage.csv mkl=~/run/u391-pristine --reps=3 --mode=mstage

python3 aggregate391.py call|stage|phase [--by-level]|mstage|spread run1-*.csv
python3 aggregate391.py pairs <csv> <armA> <armB>
```

Tables E and F differ from that in exactly two ways: they need the 2T (CPUs 0-1) and 3T (CPUs
0-2) cells, and they need one environment variable on one arm - `run.py` takes neither, so its
loop was run with those two differences made explicit, same one-process-per-cell rule, arms
interleaved per repetition and the gate before each process:

```
for n in 64 96 128; do for T in 1 2 3 4 8; do for rep in 1 2 3 4 5; do
  for var in "" 1; do                      # split, then serial (or base, then min4)
    env -u RAYON_NUM_THREADS ${var:+SEC_FORCE_SERIAL=$var} OMP_NUM_THREADS=$T \
      taskset -c $CPUS ~/run/u391-instr fork call f64 $n $T 16 250 1000
  done
done; done; done                            # table F: SEC_MIN4 instead, and T in 2 3 4 8
```

`run4-counts.txt` is the raw stdout of the counting build (built inside the probe copy with
`cargo build --release --features instr,count --features faer/dc-count`), one process per cell and
no repetitions - it is a count, not a timing, so the gate and the repetitions do not apply:

```
for T in 1 4 8; do OMP_NUM_THREADS=$T taskset -c $CPUS ~/run/u391-count fork call f64 <n> $T 16 250 1000; done
```
