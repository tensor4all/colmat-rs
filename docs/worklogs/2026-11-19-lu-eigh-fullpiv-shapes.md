# Three benchmark shapes: LU F-order at 2^k, `eigh` at one thread, `full_piv_lu` at 2^k

Issue: [tensor4all/colmat-rs#234](https://github.com/tensor4all/colmat-rs/issues/234). Evidence:
`tensor4all/colmat-benchmark` at the colmat pin `93da62172a96` (the 2026-11-17 MKL, OpenBLAS and
large-tier MKL runs on `zen5-cpu`), and a paired MKL-arm run of `origin/main` (`1b397a01`) against
this branch recorded for this record (see *Campaign pair*). No upstream revision was measured beyond
the `faer-upstream` control row the harness carries.

Host: the development workstation (`zen5-cpu`, AMD Ryzen AI 9 HX 470), measured CPUs 4 / 4-7 / 4-11
(the 8 MiB L3 domain), probes pinned the same way. Development-host numbers: provisional.

## Decisions

### `full_piv_lu`: the dip at 256/512/1024 is the pivot row swap at a 4 KiB leading dimension

- **Cause, from the breakdown.** The published curve (f64, F order, primary rows) drops at exactly
  the powers of two and recovers on both sides: 4T 9.5 / 5.7 / 9.6 GF/s at 255/256/257 and
  14.8 / 7.7 / 14.9 at 511/512/513, 8T 19.4 / 8.8 / 19.5 at 511/512/513; at 1T, where only powers of
  two are declared, 128 -> 256 goes 4.2 -> 3.9 GF/s. The harness factors in place at `ld = n`. The
  probe's `Mat` rounds the leading dimension up to 8 elements, so its 255 and 511 alias too, and they
  are as slow as 256 and 512; padding the leading dimension by one line (n = 512, ld 520) gives the
  513 rate back (1T 20.4 -> 14.2 ms, 4T 11.2 -> 5.3 ms). The algorithmic count does not change and
  there is no rank-revealing fallback on these inputs. (1536 = 3 x 512 has a 12 KiB stride, also a
  4 KiB multiple; above 1024 the matrix also leaves the 8 MiB L3, which this change does not address.) Stage timers (1T, n = 512): the row swap
  was **6.4 of 20.8 ms** at ld 512 and 0.6 ms at ld 520; the column swap, the scaling and the fused
  update did not move. Each step swaps the pivot row across all `n` columns, two lines per column,
  and at a 4 KiB stride those lines share one L1 set and a handful of L2 sets.
- **Chosen: split the row swap.** The pivot column's two entries are swapped at once; the columns past
  the pivot are swapped inside the step's fused rank-one update, which streams those columns anyway;
  the columns of `L` (left of the pivot, read by no later step) take all their swaps in one
  column-wise pass after the loop, split over the team when it pays (`rate::lu_swap`). The transpose
  dual's scaling of the pivot row moves into the sweep with the swap. The factors are the same bits
  (the swaps are exact and the arithmetic is untouched).
- **The swap is made one column ahead of the update that reads it.** Done in the same iteration, the
  column's vector loads read the entry the scalar store has just written and wait for it: n = 32
  went 10.7 -> 17.1 us. One column ahead: 11.9 us. The remaining 1.2 us at 32 and 1.5 us at 64 are
  the `L` pass (0.6 us at 32, measured by leaving it out) and the per-column prologue.
- **Rejected: swap the `L` part eagerly, fuse only the trailing part.** n = 512: 1T 17.2 ms and 4T
  7.6 ms, against 14.3 and 4.5 with the deferred pass: the `L` half of the row is as aliased as the
  other half.

### `eigh` at one thread: the tridiagonal QR's rotations were applied one call at a time

- **The gap.** Published f64 F-order 1T, fork against MKL: 0.112 / 0.095 ms at 32, 0.573 / 0.379 at
  64, 1.339 / 0.872 at 96, 2.572 / 1.810 at 128, 10.59 / 9.84 at 256; the fork wins at 512 (50.4 /
  60.3) and in c64 at every size. 4T/8T use the D&C path and are level with or ahead of MKL.
- **Where the time is.** Below n = 128 a sequential call takes the tridiagonal QR iteration (the
  team path takes D&C with leaves of 16 and is not touched). At f64 n = 128, 1T, 18.5 k rotations:
  the bulge chase without `U` is about 40 ns per rotation on this core, a latency chain of two
  divisions and a square root, and applying each rotation to two whole columns of `U`
  (`JacobiRotation::apply_on_the_right_in_place`, one dispatch, one SIMD context, head and tail per
  rotation) took about as long again. The values-only call is 1.29 ms of the 2.38 ms call.
- **Chosen: apply a sweep's rotations together.** The chase stores the sweep's `(c, s)` (two columns
  of `n` reals from the caller's scratch; a D&C leaf lends the merge's `mus`/`shifts`, unused below a
  leaf) and `apply_rotation_sequence` applies them afterwards: a strip of four SIMD vectors of rows
  stays in registers while the rotations pass over it, a column is loaded and stored once per chunk
  instead of twice per rotation. The per-element operations are the same fused ones as the
  rotation-at-a-time loop, and an identity rotation is still skipped, so `U` and the eigenvalues
  are the same bits (`rotation_sequence_matches_rotation_at_a_time`; the whole `self_adjoint_evd`
  output compared bitwise with `origin/main` at n = 2..300, 1T and 4T).
- **Chunks of 8 rotations, not the whole sweep.** In a D&C leaf `U` is a block of the whole matrix's
  `U`, so a strip's columns can be 4 KiB apart: the whole sweep in one call made n = 512 1T 16 %
  slower (58.0 against 50.4 ms). A chunk keeps the strip's streams within an L1's associativity
  (eight ways or more on current x86 and ARM cores). Scan (f64, 1T, ms): the table is next to
  `ROTATION_CHUNK`; 8 is within 1 % of the best at every size and the only width that does not
  lose at 512.
- **Also chosen: `sqrt(1 + t*t)` for `hypot(t, 1)` in `make_givens`, f32/f64 only.** For `|t| <= 1`
  `hypot` scales both legs by the power of two `sqrt_max_positive`, which changes no rounding, so the
  two are the same bits (checked on 10^7 arguments over 20 decades as well); the direct form takes
  the range selection off the chase's latency chain. Worth about 7 % of the chase.
- **Not done: the chase itself.** It is latency-bound (about 40 ns per rotation here) and MKL's
  `?stedc` avoids it by divide and conquer below n = 128. Changing the 1T entry threshold is a
  separate measurement: the 2026-11-10 record found D&C at 1T -5 % to +9 % against QR, and that
  balance moves with this change.

### LU F order: a 2^k dip from the `U` panel solve's right hand side, left as measured

- **The odd shape is real and it is the leading dimension.** In the published F-order chart the fork
  dips at 256 and 512 and does not recover at 257/513 as MKL does (4T: 36.5 / 33.6 / 34.9 GF/s at
  255/256/257, 59.4 / 57.1 / 57.6 at 511/512/513; 8T 512 -6 %). The harness call path is not it:
  `fbench run` reproduces the published primary cells within 2 % and the probe agrees. Dispatch is
  not it either: at a fixed `n` the dip follows the leading dimension (n = 255 at ld 256 drops to
  34.5, n = 256 at ld 257 recovers to 35.9, n = 512 at ld 520 reaches 61.2 against 57.1). Stage timers
  (n = 512, ld 512 against 520): the `U` panel's triangular solve carries +222 of +265 kcycles at
  1T, +124 of +174 at 4T and +94 of +155 at 8T; the products, the leaves and the swaps barely move.
- **Isolated solve (unit lower `b x b` against `b` columns, 1T):** ld 512 against 520 is 2.6x at
  b = 16 (not staged), +25 % at 32, +19 % at 64, +13 % at 128, +8 % at 256 (staged nodes); the
  triangle's own stride does not matter (padding only the triangle changed nothing), which is the
  2026-11-05 record's finding. What is left at the staged sizes is the staging copy itself.
- **Rejected: a contiguous per-column copy in the staging.** It took the isolated solve from 15.1 to
  14.4 us at b = 64 and 671 to 648 us at 256, and moved the LU at 256/512 by 0.5-1 % at 1T/4T/8T,
  inside the spread of the interleaved repeats. Not carried.
- **Not done: the staging rule at 8T.** At 8T a level's columns are split eight ways, so a chunk of
  the b = 128 level has 16 columns and falls under the 24-lines-per-set floor that was measured with
  `b = k` at one thread. Revisiting that floor per chunk width is a separate, host-dependent scan.

## Verification conclusions and constraints

- Tests: `cargo test -p colmat --lib` passes (842 at the time of the change, plus the two new tests).
  `split_row_swap_reconstructs_every_layout`: `P^T (L U + S) Q^T = A` and `|L| <= 1` for
  column-major, row-major (the transpose dual), generic-stride (the `matmul` fallback) and
  4 KiB-stride views, square, wide and tall, truncated (`max_rank`) and not, f64 and c64, on one
  thread and on a team of 3. `rotation_sequence_matches_rotation_at_a_time`: bitwise equality with
  the rotation-at-a-time loop, sizes across the chunk and SIMD strip boundaries, early deflation,
  `U` inside a 512-row matrix.
- Bitwise: `self_adjoint_evd` (`U`, eigenvalues) and `full_piv_lu` (column-major, row-major, wide)
  match `origin/main` at n = 2..300 at 1T and 4T.
- Campaign pair: see below. The LU family is the control (unchanged code).
- **Unverified.** Only this host (Zen 5/Zen 5c, 12-way 48 KiB L1). The 64-core host and the
  OpenBLAS arm were not run. f32/c32 were not measured (the code paths are shared).

## Campaign pair

`colmat-benchmark` `record_run.py zen5-cpu`, MKL arm (`faer-fork`, `faer-upstream`, `lapack-mkl` in one
binary), `--reps 5 --aa 3 --prime-ms 250`, families `lu,full_piv_lu,eigh`, f64/c64, C and F order;
baseline `origin/main` `1b397a01`, candidate `4849555e`; recorder pinned to CPUs 0-3,12-15. Runs
(not published, `--allow-local-commits`): `faer-kernels/20261118T235559Z-i234-base` against
`20261119T001949Z-i234-cand` (32-512, 1T/4T/8T) and `faer-kernels-large/20261119T004108Z-i234-large-base`
against `*-i234-large-cand` (255/257/511/513/1024, 4T/8T). Per-cell ratios are in the PR.

- **Controls.** The `faer-upstream` and `lapack-mkl` rows, unchanged code: median 1.0014 over 180 F-order
  cells (0.937-1.107; the two outliers are `faer-upstream` single cells), 1.0001 over the C-order ones,
  1.0015 over the 100 large-tier cells (0.983-1.019). The fork's `lu` (unchanged code): 0.980-1.031.
- **`full_piv_lu`** (F order, cand/base): f64 4T 0.765 / 0.521 / 0.429 at 128/256/512, 8T 0.749 / 0.489 /
  0.342, 1024 0.557 (4T) and 0.405 (8T); 255/257/511/513 0.76-0.92 (the separate swap pass also went away
  where it did not alias); 1T 0.707 at 256 and 0.721 at 512. The curve is monotone now: f64 4T
  10.8 / 10.9 / 11.2 GF/s at 255/256/257, 16.7 / 17.8 / 17.6 at 511/512/513, 20.7 at 1024.
  **Regression:** n = 32 is 1.05-1.16 slower (f64 1T 10.7 -> 12.4 us; the unchanged upstream row of the
  same cell moved 1.107 in that pair) and f64 64 1T 1.039: the `L` pass and the per-column prologue.
- **`eigh`** (cand/base): f64 1T 0.944 / 0.754 / 0.782 / 0.801 / 0.909 / 0.955 at 32..512, c64 1T
  0.949 / 0.804 / 0.830 / 0.856 / 0.940 / 0.981; 4T/8T 0.95-1.003 (32 at 0.95-0.96, from `make_givens`
  and the D&C leaves; nothing above 1.003).
