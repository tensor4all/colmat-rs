# SVD target cases against MKL, and the 511/513 speedup (#83)

Issue: [colmat-rs#83](https://github.com/tensor4all/colmat-rs/issues/83).

## Target and evidence contract

The requested target is at least 1.1 times MKL's throughput (colmat elapsed time
at most MKL/1.1) at 1024x64, 128x128 and 256x256. The added 511x511 and 513x513
cases are measured alongside 512x512 to expose leading-dimension boundaries.
The active goal also includes QR at 511/513: prioritize at least 10% lower
elapsed time than the current implementation, with 5% accepted by the user
as the minimum. Report each target case separately in milliseconds.
Primary coverage is f64/c64, C/F layout, values-only/thin SVD, 1/4/8 threads.
The four native types remain numerical and non-regression guards.

Baseline colmat is `a395150c086f`; candidate `a407a0b6`. Harness
`f3cd501` (the local SVD diagnostic with the 511/513 grid and per-case target
columns), MKL 2025.3, Rust 1.96.0, release, Ryzen 8945HS. Both runs are in
sibling colmat-benchmark: baseline `data/probes/svd-docker/20261114T101544Z/`,
candidate `data/probes/svd-docker/20261114T115445Z/`. The host is the
`zen4-8945hs-dev` profile with boost on, one shared L3 and seven physical cores
plus one SMT sibling at 8T, so this is a diagnostic and not formal campaign
acceptance.

MKL is timed through its public LAPACKE API only. No vendor internals or vendor
source are consulted; no numerical implementation is translated in this work.
The reference LAPACK sources in `extern/OpenBLAS/lapack-netlib` are BSD-3 and
are the algorithmic record already cited in the SVD modules; they were read for
the shift/deflation structure, and no constant was carried over without a
measurement on this code.

## Investigation

Temporary stage timers in the blocked panel (removed before the commit) split
the 511 f64 values call into its stages at 1T and 8T. Before the change:

| stage | 1T ms | 8T ms |
| --- | ---: | ---: |
| `Y` product (`A^H u`) plus corrections | 5.15 | 2.27 |
| `X` product (`A r^H`) plus corrections | 5.40 | 2.80 |
| row gather/update/reflect | 0.70 | 1.16 |
| trailing update (two `k = 16` products) | 2.60 | 0.79 |
| column update | 0.26 | 0.35 |
| unblocked tail | 0.26 | 0.39 |
| compact-factor upgrade | 0.23 | 0.21 |
| dqds | 2.48 | 2.50 |
| copy | 0.06 | 0.06 |

The two products are memory-bound (80 GB/s at 1T, 237 GB/s at 8T) and cannot be
fused: `X_i` needs `r_i`, which needs the row update, which needs `Y_i`. The
remaining lever was the layout of the panel's rows and the trailing update.

Rejected, with the number that rejected it (local probe, median of 9):

- Hand-fused correction recurrences: 1T 18.34 against 17.17 ms, 8T 12.60
  against 10.77. The generic `matmul` kernels vectorize better than the
  hand-written `z!` loops.
- Initial dqds zero-shift sweep removed (`have_sweep = true`): 1T 17.16 against
  17.05, 8T 10.84 against 10.77. The shift from the raw factors costs more
  sweeps than it saves.
- 2/3 `dmin` shift variants: 508K and 911K entry updates against 507K.
- Values-only divide-and-conquer instead of dqds: 1T 22.99, 4T 13.11, 8T 13.85
  against 17.16 / 10.23 / 10.76. dqds stays.
- Iterator-formed dqds sweep and `fmin` in the sweep: 1T 15.96 and 15.78
  against 15.68. Only the pre-sweep `dmin` scan became branchless.

Adopted, in `faer/src/linalg/svd/bidiag.rs`, `dqds.rs` and `mod.rs`:

1. **The panel's rows are stored transposed.** A row of `A` has stride `m`,
   and the right reflectors stored in it are read again by the next steps'
   correction recurrences and by the trailing update. The panel now gathers
   each row once, reflects it into a contiguous `rows` buffer (one row per
   column, so a row is contiguous) and reads it from there. `A`'s rows are
   written back only when the compact factors are wanted, so the
   singular-value-only path skips the strided scatter.
2. **The values-only SVD skips the compact-factor upgrade** (`H_left`/
   `H_right` are never read). `BidiagParams::compute_householder_factors`
   defaults to true; `svd_imp` clears it when both output vectors are `None`.
3. **One trailing product of width `2 ib`.** `W = [V | X]` and `Z = [R^H | Y]`
   are assembled in the second half of the correction storage and applied as
   one product, so the trailing block is read and written once per panel. The
   copy is `3 (m + n) ib` scalars per panel against one traversal of the block.
4. **dqds keeps only whether the smallest pivot is the bottom one**, not its
   index, so the pre-sweep scan has no unpredictable branch per entry.

## Verification

- `cargo test -p colmat --release --lib`: 823 passed, 0 failed, 162 ignored.
  `linalg::svd`: 79 passed. The all-ones rank-1 case (`test_special`) stays
  under its tolerance; the transposed row buffer stores the same values as the
  scattered rows, so the vector paths are numerically unchanged.
- The Docker diagnostic verifies the whole timed grid at 1/4/8T before timing
  and checks the measured source identity again at the end.

## Paired result (candidate / baseline, below 1 is faster)

`scripts/docker_mkl.sh exec` with `docker/compare_svd.py --shapes
511,512,513 --families svd_values,svd_thin`, 3 process starts, 5 samples,
250 ms priming, MKL arm. 72 paired cells; the MKL control over all cells is
median 1.003, min 0.977, max 1.087 (the max is `svd_thin f64 F 511 8T`; that
cell's faer ratio 0.935 agrees with its neighbours, so the control moved, not
the candidate).

`svd_values`, median over C/F layout:

| dtype | shape | 1T | 4T | 8T |
| --- | --- | ---: | ---: | ---: |
| f64 | 511x511 | **0.901** | **0.804** | **0.813** |
| f64 | 512x512 | 0.909 | 0.809 | 0.822 |
| f64 | 513x513 | **0.901** | **0.803** | **0.815** |
| c64 | 511x511 | 0.941 | 0.867 | 0.858 |
| c64 | 513x513 | 0.933 | 0.858 | 0.850 |

`svd_thin` (guard), median over C/F layout:

| dtype | shape | 1T | 4T | 8T |
| --- | --- | ---: | ---: | ---: |
| f64 | 511x511 | 0.967 | 0.929 | 0.934 |
| f64 | 513x513 | 0.968 | 0.934 | 0.938 |
| c64 | 511x511 | 0.983 | 0.955 | 0.955 |
| c64 | 513x513 | 0.986 | 0.958 | 0.958 |

f64, milliseconds, 511/513:

| shape | layout | 1T | 4T | 8T |
| --- | --- | --- | --- | --- |
| 511 | F | 17.017 -> 15.332 | 10.120 -> 8.122 | 10.522 -> 8.601 |
| 511 | C | 17.074 -> 15.388 | 10.112 -> 8.151 | 10.611 -> 8.585 |
| 513 | F | 17.299 -> 15.591 | 10.261 -> 8.252 | 10.713 -> 8.736 |
| 513 | C | 17.457 -> 15.724 | 10.327 -> 8.274 | 10.780 -> 8.772 |

The f64 values-only case is 9.9% faster at one thread, 19.6-19.7% at four and
18.3-18.5% at eight. Complex f64 is 6-7% at one thread and 14-15% at four and
eight: its bidiagonalisation is twice the element work, so the serial stages
this change removes are a smaller share of its total.

## One-thread follow-up (`2b30412d`, `52b71343`)

The first pair left f64 at one thread at 9.9%. Two dqds changes closed it:

- `2b30412d` adds `RealField::mul_add_impl` (the unfused default; `f32`/`f64` override it
  with the hardware FMA) and uses it for `delta = delta * t - tau`, which was two serial
  operations on the sweep's dependency chain.
- `52b71343` carries the sweep's smallest factor across iterations instead of rescanning
  the live block before every shift; the cache is dropped when a deflation releases that
  entry. The pre-sweep scan is removed on the sweeps that do not release the minimum.

Paired Docker+MKL run, baseline `20261114T101544Z` (a395150c) against candidate
`20261114T122233Z` (52b71343), same harness `f3cd501`, same grid and protocol as the first
pair. 72 cells; the MKL control is median 0.9993, min 0.979, max 1.088 (the max is the
same `svd_thin f64 F 511 8T` control cell as the first pair).

`svd_values` candidate/baseline, median over C/F:

| dtype | shape | 1T | 4T | 8T |
| --- | --- | ---: | ---: | ---: |
| f64 | 511x511 | **0.884** | **0.774** | **0.809** |
| f64 | 513x513 | **0.882** | **0.776** | **0.807** |
| c64 | 511x511 | 0.922 | 0.848 | 0.867 |
| c64 | 513x513 | 0.919 | 0.846 | 0.861 |

f64 milliseconds, 511/513:

| shape | layout | 1T | 4T | 8T |
| --- | --- | --- | --- | --- |
| 511 | F | 17.017 -> 15.038 | 10.120 -> 7.834 | 10.522 -> 8.515 |
| 511 | C | 17.074 -> 15.097 | 10.112 -> 7.832 | 10.611 -> 8.571 |
| 513 | F | 17.299 -> 15.272 | 10.261 -> 7.971 | 10.713 -> 8.660 |
| 513 | C | 17.457 -> 15.368 | 10.327 -> 8.005 | 10.780 -> 8.685 |

`svd_thin` (guard), median over C/F: f64 511 0.959 / 0.921 / 0.926 and 513
0.961 / 0.923 / 0.930 (1T/4T/8T); c64 511 0.974 / 0.953 / 0.961 and 513
0.978 / 0.957 / 0.961.

The f64 values-only case is now 11.6-11.8% faster at one thread, 22% at four and 19% at
eight. Complex is 7.8-8.1% at one thread and 15% at four and eight: its bidiagonalisation
is twice the element work, so the serial stages these changes remove are a smaller share
of its total and the two memory-bound complex products dominate it.
## QR at 511/513 (internal cap `0f50f4a2`, caller-side width `1b36f3f7`, both reverted)

The QR factorization was not part of the first pair. A width scan on the local host
(`qr_in_place` at 511/513, the campaign's `qr_factor` call) found that the shipped
recommendation (32 columns at 511, 48 at 513) is not the eight-thread optimum:

| shape | width | 1T | 4T | 8T |
| --- | ---: | ---: | ---: | ---: |
| 511 | 16 | 3.31 | 1.54 | **1.29** |
| 511 | 32 (shipped) | 3.33 | 1.49 | 1.42 |
| 511 | 64 | 3.48 | **1.43** | 1.68 |
| 513 | 16 | **3.35** | 1.57 | **1.29** |
| 513 | 48 (shipped) | 3.46 | 1.41 | 1.57 |
| 513 | 64 | 3.43 | **1.37** | 1.69 |

At eight participants the top-level panel is the serial part, so a narrower block
trades more parallel trailing applies for less recurrence work: 16 is 9.4% ahead at
511 and 17.5% at 513 for `qr_factor`. At four participants the wider block wins, and at
one it is flat (511) or 16 is 2.9% ahead (513). The factor plus the thin `Q` formation
(the `qr` family) is dominated by the block apply, so the same move is 2.9% (511) and
8.3% (513) at 8T.

A candidate capped the internal top-level block at 16 when `par.degree() >= 8` and the
matrix is a square of at most 640 rows. Paired local A/B at 8T (base/candidate
alternated, three runs each; `qr_factor`, then factor + thin Q):

| shape | `qr_factor` | factor + Q |
| --- | ---: | ---: |
| 256 | 0.426 -> 0.390 (-8.5%) | 1.583 -> 1.513 (-4.4%) |
| 384 | 0.849 -> 0.772 (-9.1%) | 3.407 -> 3.266 (-4.1%) |
| 511 | 1.430 -> 1.296 (-9.4%) | 6.208 -> 6.029 (-2.9%) |
| 513 | 1.570 -> 1.296 (-17.5%) | 6.660 -> 6.100 (-8.3%) |

The 1T and 4T rows are controls and agree within noise.

**The candidate was rejected by its own numerical check.** A Docker+MKL paired run of
`qr_factor` (baseline `20261114T124251Z`, a395150c; candidate `20261114T124801Z`,
0f50f4a2) passed the `verify` grid at 1/4/8T but the timed 8T grid returned
`qr_factor residual inf exceeds ...` for the colmat row on f64 and c64 at 511, 512 and
513. The baseline run on the same harness and host passed all cells. The internal cap
therefore changes the block width underneath a caller-provided `H` whose layout the
look-ahead/staged-panel path assumes, and that path is not correct at the smaller width.
The commit is reverted by `71a9e84f`; the tree is the SVD tip again.

### Caller-side width (`1b36f3f7`, reverted by `aedbe8f5`)

The safe route is to make the caller pass a 16-wide `H`: `recommended_block_size` returns
16 for a square of 129..640 columns. That path is numerically correct. The Docker+MKL
paired `qr_factor` run (baseline `20261114T124251Z`, a395150c; candidate
`20261114T125936Z`, 1b36f3f7) completes with the MKL control at 0.9996, and its
candidate/baseline ratios are:

| dtype | shape | 1T | 4T | 8T |
| --- | --- | ---: | ---: | ---: |
| f64 | 511 | 1.002 | 1.037 | **0.893** |
| f64 | 512 | 0.986 | 1.038 | **0.908** |
| f64 | 513 | 0.973 | 1.070 | **0.802** |
| c64 | 511 | 0.996 | 1.015 | **0.885** |
| c64 | 512 | 0.994 | 1.009 | **0.870** |
| c64 | 513 | 0.980 | 1.022 | **0.757** |

So the eight-thread target is met and exceeded for `qr_factor` (10.7%/19.8% at 511/513
for f64, 11.5%/24.3% for c64), but the same width also applies at four participants,
where it is 1.5-7% slower, and at one it is flat. That global recommendation is not the
adopted form; it is reverted by `aedbe8f5` and kept as the measurement that motivates the
participant-aware API below.

### Participant-aware recommendation (`9cced7f4`)

The width is the caller's, so the participant count belongs in the recommendation the
caller uses. `recommended_block_size_with_degree` returns 16 for a square of 129..640
rows at eight or more participants and the shape-alone width otherwise (below 129 rows
the shape-alone width is at most 8, so 16 would be wider than the shape calls for and can
exceed `n`). The fbench
caller passes its declared budget (`fbench` commit `2fc0fa2`); the upstream arm keeps the
shape-alone width, so the pair has the same control structure as the SVD runs.

Paired Docker+MKL `qr_factor` run: control baseline `20261114T131848Z` (the API with the
shape-only policy, `cec72348`) against candidate `20261114T132518Z` (9cced7f4), same
harness `2fc0fa2`, 511/512/513, f64/c64, C/F, 1/4/8T. The MKL control is median 0.9932
(the run's overall level; the 1T/4T rows below are read against it):

| dtype | shape | 1T | 4T | 8T |
| --- | --- | ---: | ---: | ---: |
| f64 | 511 | 0.999 | 0.996 | **0.900** |
| f64 | 512 | 0.995 | 0.995 | **0.904** |
| f64 | 513 | 0.995 | 0.999 | **0.806** |
| c64 | 511 | 0.999 | 0.997 | **0.881** |
| c64 | 512 | 0.993 | 0.995 | **0.872** |
| c64 | 513 | 0.996 | 0.995 | **0.761** |

One and four participants are unchanged (0.993-0.999, inside the run's level and the
MKL control), and eight participants is 10.0%/19.4% ahead for f64 and 11.9%/24.0% for
c64 at 511/513. This is the adopted QR form: the 8T target is met without the 4T
regression of the global recommendation.

### c64 at one thread

An instrumented values call (511x511 c64, 1T, last samples of the warm run) splits as:
total 36.75 ms, bidiagonalisation 34.3 (Y product 12.2, X product 11.2, trailing update
9.17, row 0.65, column 0.49, tail 0.58), dqds 2.14, copy 0.12. The complex apply is 3.8x
the f64 apply for 4x its flops, i.e. it runs at the same flop rate, and the two complex
products are 2.7x the f64 ones for 2x the bytes. A two-`k = 16` complex apply was tried
and was slower (9.92 against 9.17 ms). The complex one-thread result is therefore
recorded as an **explicit unmet target**: 7.8-8.1%, with the remaining 2.2-2.4 points in
complex arithmetic, not in the serial stages this work removed.

## Scope decision (2026-11-14)

The user accepted the current scope after the paired evidence: f64 `svd_values` at
10%+ for 511/513 at 1T/4T/8T, QR `qr_factor` at 10%+ for 8T with 1T/4T unchanged by the
participant-aware recommendation, and c64 `svd_values` at 1T 7.8-8.1% recorded as an
explicit unmet target (4T/8T are at 15%). Closing c64 1T would need a dqds algorithm
change or a complex-kernel campaign, which is outside this layout change.

## What remains unverified

- The f64 values-only 10% target is met at every thread count (11.6-11.8% at
  1T, 22% at 4T, 19% at 8T). Complex is 7.8-8.1% at 1T and 15% at 4T/8T; the
  one-thread complex gap is its bidiagonalisation, which is twice the element
  work of the real one and whose two memory-bound products are untouched.
- c64 values-only at one thread is an **explicit unmet target**: 7.8-8.1%
  (0.922/0.919). The stage split above puts the remaining gap in the complex
  products and apply, which are at their flop/bandwidth limits for this layout;
  a two-GEMM complex apply lost. The 4T/8T rows are at 15%, so the family's
  average is past 10%.
- QR at 511/513 is **adopted at eight participants only**. The participant-aware
  recommendation (`9cced7f4`) leaves 1T/4T at the shape-alone width, so those
  rows do not move, and it is 10.0%/19.4% ahead for f64 `qr_factor` at 8T
  (11.9%/24.0% for c64). The global recommendation that would also move 4T
  regresses it by 1.5-7% and is not adopted; the internal cap is invalid by
  contract (the caller's compact-factor layout). The QR 10% target is therefore
  met at 8T only.
- The host cannot establish formal adoption (boost on, shared L3, no 64-core
  scaling). A campaign pair remains for whoever moves the pin.
