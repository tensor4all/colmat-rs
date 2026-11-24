# LU at the 4T/8T boundary cells: a one-panel look-ahead (colmat-rs#239)

Issue: [tensor4all/colmat-rs#239](https://github.com/tensor4all/colmat-rs/issues/239). Baseline record:
[tensor4all/colmat-benchmark#94](https://github.com/tensor4all/colmat-benchmark/pull/94) (colmat
`a5536ad3`, harness `9988f5a`, evo `zen5-cpu`, boost 0 / `performance`, MKL arm from the same
runs). Measured on evo only (Ryzen AI 9 HX 470, 12 cores, two L3 domains); upstream faer was not
measured; the yardstick is MKL 2026.2 through `dgetrf` only.

## Decisions

### Where the time goes

Stage timers around the recursion (removed before the commit), `f64`, column-major, a caller-held
team pinned to one L3 domain, ld = n as the harness lays it out:

- **The narrow levels do not scale.** At n = 512 the leaves plus the levels of at most 64 columns
  took 546 us of 1153 at 8T and 581 us at 1T. The 64-column panels are the sequential part of a
  team call; MKL's 8T/1T ratio is 5.0, the fork's 3.2, while the fork's 1T is 0.73x MKL.
- **The leaf is bound by moving its data, not by its code.** 64 leaves take 212 us at 8T and
  206 us at 1T (about 4.5 GFLOP/s), linear in the rows (22 cycles per row per 8-wide leaf) with a
  small intercept. Its update loop is load, fused multiply-add, store, unrolled by two.
- **The swap stages of the small nodes are four times slower at 8T than at 1T** (n = 32 nodes:
  50 us against 12, width 1 in both): they write `L` columns that the parent's team product has
  just read on every member.
- **The leading dimension is part of the boundary shape.** `Mat` pads its column stride to a
  multiple of 8, so a probe built on `Mat` measures 511 at ld 512 and 513 at ld 520; the harness
  factors a dense ld = n buffer. At ld = n the probe reproduces the record (4T 255/256/257 0.305 /
  0.321 / 0.322 ms, 8T 511/512/513/1024 1.133 / 1.21 / 1.17 / 5.78). Padding ld moves 4T 256 by
  -6 % (the root's `U` solve at a 2 KiB stride, 40 -> 32 us), 8T 513 by -4 % (the 128-column
  nodes' product at ld 513, 88 -> 64 us) and 8T 1024 by -4.5 %: a part of the gap, not all of it.

### The change: one panel of look-ahead per team level

A level whose right child is wider than 64 columns updates that child's first panel (the node of
at most 64 columns on its leftmost chain, or the leaf that chain stops at when a caller raises
`recursion_threshold` past 64) first; then the leader factors the panel alone while the
other members compute the rest of the trailing product (`thread::join_sized(par, 1, ..)`), and the
right child skips the panel. The arithmetic of every column is the same mathematically, but the
trailing product of a level becomes two calls (panel columns, then the rest on `p - 1` members),
which need not round like the single call. What is established
(`lu::tests::par_identity::lookahead_does_not_change_the_factorization`, f64/f32/c64/c32, column-
and row-major, 200-300 and 8T n = 1024, team widths 4 and 8, plus `recursion_threshold = 128`):
the permutation and transposition count are those of the same call without the look-ahead, and
the factor is within `||on - off||_1 <= 30 max(m, n) eps ||off||_1` of it. Exact bit identity held
on x86_64 (evo, ubuntu CI) at every tested case; it does **not** hold on aarch64 (Apple M-series)
at f32/f64 257 x 257 column-major, width 4 (deterministic, colmat-rs#244). The feature saves time;
it does not promise cross-platform bit identity. Separately, the pre-existing team path at width 3
(where the look-ahead cannot run) differs from `Par::Seq` in some runs on aarch64 at f32/f64
200 x 200; that is a property of the team product on that host, not of this change.

Unlike the faer#123 attempt (`2026-11-14-lu-trailing-product-lookahead.md`), the child is not
narrowed to half the team (the leader factors only a panel that gains nothing from the team, and
the child gets the whole team back afterwards), and the product is split only where the overlap
pays. The gate is `P > F / (p (p - 1))`: without look-ahead a level spends `F/p + P` (the far
product `F`, then the panel `P`), with it `max(P, F/(p-1))`. `P` is the panel's multiply-adds at
twice the product's cost (measured: 8.1 against 15.5 G multiply-adds per second at 1T), and the
team must have at least four members. Cost of the split as a function of `p`: one extra region
(the near product) and one `join_sized` per level that looks ahead; the far product keeps its
own 2D split on `p - 1` members, and the panel is sequential, so the overlap shrinks as `p` grows
(the gate turns it off once `F / (p (p - 1))` exceeds the panel).

Scans (median ms of 31-41 calls, probe on the pinned team; the tables are next to the constants):
panel 64 was best at every target cell (16 and 32: 3-8 % slower than no look-ahead; 128 and 256:
mixed); a half-width panel for nodes with a right child of at most 64 columns was 2-3 % slower at
8T; the gate's factor 1 and 2 gave the same cells, 4 lost 4T 1024, and no gate at all made 4T 1024
5.7 % and 8T 1536 3 % slower.

### Tried and dropped

- **A left-looking leaf** (each column brought up to date in one pass, the pending scaling fused
  into it): bitwise the right-looking leaf, half the memory operations, and no faster (1T leaf
  stage 197 -> 209 us, 8T 214 -> 218 us at n = 512).
- **Lazy row swaps** (the `L` columns get their swaps at the first ancestor trailing update that
  reads them, plus one work-balanced cleanup at the root): bitwise unchanged, but at 8T n = 512 the
  pending regions took 131 us against the old swap stages' 120 us, and the cells moved -2 % to +3 %.
  An `L` column must be fully permuted before every product that reads it, so the number of
  passes over each column does not fall; only their place does.
- **Forcing the swap stage or the trailing update of the small levels sequential**: no change or
  slower (the regions were already sequential at those levels).
- **Staging the `U` solve's right hand side from 16 lines per set** (instead of 24): the isolated
  4 KiB, 16-column solve goes 6.08 -> 4.51 us at b = 64, but the 2 KiB, 32-column case (4T 256's
  root) does not move and b = 128 gets slower; the target cells are mostly not at a 1 KiB multiple.

## Verification conclusions and constraints

- `lu::tests::par_identity::lookahead_does_not_change_the_factorization` (f64, f32, c64, c32):
  square, tall and wide shapes whose right children reach the panel through uneven halves (200,
  257, 300 x 211, 211 x 300) and 8T n = 1024, column-major and row-major, team widths 3, 4 and 8
  with every region on the team; look-ahead on against off at the same width: permutation and
  transposition count equal, factor within the ratio bound above, and a per-call count that the
  look-ahead ran at widths 4 and 8 and did not at 3. (It replaced a comparison against the
  sequential run, see above.) The existing identity tests (26 shapes, widths
  1/2/4/8, derived floor) pass unchanged.
- Release `cargo test -p colmat --lib`: 850 passed. `cargo clippy --workspace --all-targets -D
  warnings` clean. `cargo fmt --check` reports only the existing diff in
  `faer/tests/gevd_223_extreme_scale.rs`.
- 1T and the small sizes take the old path by construction (a team of at least four, and a right
  child wider than 64 columns, so n > 128); see the campaign below.
- Not measured: OpenBLAS, the 64-core host (the scaling of the gate above 8 threads is a model, not
  a measurement), C order, and the other dtypes' timing (their bits are tested).

### What remains unverified

- The rest of the gap at 8T 511/513 is the sequential panel itself and the leading-dimension
  aliasing of the products and the `U` solve at ld = 2^k ± 1; neither is addressed here.
- The leader idles at the root level when the far product outlasts the panel (8T 511: panel 75 us,
  far product 148 us). Giving it a slice of the far product would save about 1 % by the probe's
  numbers; not tried.

## Campaign

Candidate `7c557fd9` recorded with the commands of the baseline record, unchanged except the label
(`record_run.py zen5-cpu faer-kernels{,-large} ... --engines faer-fork,lapack-mkl --families lu
--dtypes f64 --reps 5 --aa 6 --prime-ms 250 --no-publish`, harness `9988f5a`, recorder on CPUs
0-3,12-15, other agents confined there), against the colmat-benchmark#94 runs of the base
`a5536ad3`. Runs: `faer-kernels-large/20261119T140517Z-lu239-large-mkl-cand` and
`faer-kernels/20261119T141221Z-lu239-daily-mkl-cand` (development-host, not published). Each
number is the median across 6 process starts of each start's median; primary in-place row;
not tested for significance.

| cell (f64, F) | fork/MKL before | after | fork after/before | MKL after/before |
|---|---|---|---|---|
| 4T 255 | 1.064 | 1.025 | 0.961 | 0.997 |
| 4T 256 | 1.124 | 1.081 | 0.967 | 1.005 |
| 4T 257 | 1.099 | 1.056 | 0.955 | 0.994 |
| 4T 511 | 1.056 | 1.040 | 0.982 | 0.997 |
| 4T 512 | 1.051 | 1.010 | 0.967 | 1.007 |
| 4T 513 | 1.084 | 1.062 | 0.975 | 0.995 |
| 4T 1024 | 0.996 | 0.978 | 0.987 | 1.005 |
| 8T 255 | 1.021 | 1.006 | 0.995 | 1.011 |
| 8T 256 | 1.023 | 1.035 | 1.009 | 0.997 |
| 8T 257 | 1.027 | 1.043 | 1.012 | 0.997 |
| 8T 511 | 1.118 | 1.088 | 0.975 | 1.002 |
| 8T 512 | 1.084 | 1.055 | 0.975 | 1.001 |
| 8T 513 | 1.133 | 1.113 | 0.984 | 1.001 |
| 8T 1024 | 1.096 | 1.098 | 0.999 | 0.997 |
| 8T 1536 | 0.984 | 0.978 | 0.994 | 1.000 |

- **The issue's target is not met.** Every target cell moved toward MKL, but 8T 511/513/1024 are
  still 1.09-1.11x and 4T 256/257 1.06-1.08x. 8T 1024 did not move.
- **8T 257 is 1.2 % slower** (the six starts do not overlap: 0.288-0.292 against 0.291-0.294 ms);
  the gate's model predicts a gain there, so the overlap's fixed costs are what it misses.
- C order moves the same way (4T 255/257 0.95-0.96, 8T 511/513 0.98, 8T 257 1.009).
- 1T is 0.991-1.001 and n <= 128 at 4T/8T 1.000-1.016 with the n = 32 starts overlapping
  (5.24-5.34 against 5.26-5.53 us): those cells never reach the look-ahead (a team of at least
  four and a right child wider than 64 columns). The MKL control row moved 0.992-1.011.
