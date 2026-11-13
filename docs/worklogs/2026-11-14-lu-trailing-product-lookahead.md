# LU: the trailing product's look-ahead was measured and dropped (faer#123)

**Outcome: a regression at every thread count, not shipped. What remains is the bit-identity test
that the attempt produced** (`faer/src/linalg/lu/tests/par_identity.rs`), because it guards an
invariant the residual family cannot see.

## What was tried

The partial-pivoting LU recursion ran one region per level: swap and solve the panel's columns, one
product over the whole trailing block, then the recursion into the lower-right block. At `n = 2048`
on 8 participants (`evo2`/`evo2_vpn`, Ryzen AI MAX+ 395, 16 physical / 32 logical, two L3 domains),
the product was 10.3 ms of a 15.5 ms factorization (66%) at 5.1x of 8, the panel solve 4.6 ms (30%,
a serial triangular solve per column chunk), the unblocked leaf 2.0 ms (13%) and the swap stage
0.8 ms; the serial bound implied by the totals is `W_s <= (8/7)(T8 - T1/8) = 5.78 ms`.

The attempt split that product at the child's first read set — the first `level_block_size(n1)`
columns of the trailing block, which is all the child's own panel recursion touches — computed
those in the level's region, and ran the rest in the same region beside the child, whose section
kept half the participants (`thread::join_sized`). The child waited for it at its own trailing
update through a one-shot flag on the level's frame (no allocation), and the level applied the
child's row swaps on the columns left of its span after the join, so the two views were disjoint in
time without a second join per level.

## Measured result — dropped

`fbench run --threads 1,4,8,16 --n 512,1024,2048 --dtype f64 --family lu,solve --rows
faer-fork,lapack-mkl --reps 5 --prime-ms 250 --statistic median`, 3 processes per cell, median of
the process medians, fork `031a94a` against baseline `e1c83482`, MKL unchanged row as the
machine-state control (it moved 0.98–1.05, so the host was stable):

| row | n | 1T | 4T | 8T | 16T |
|---|---|---|---|---|---|
| fork `lu` column-major | 512 | 1.30 | 1.53 | 1.85 | 2.03 |
| fork `lu` column-major | 1024 | 1.25 | 1.64 | 1.98 | 2.16 |
| fork `lu` column-major | 2048 | 1.23 | 1.38 | 1.53 | 1.40 |
| fork `lu` row-major | 2048 | 1.13 | 1.64 | 1.82 | 1.65 |
| fork `solve` row-major | 2048 | 1.14 | 1.69 | 1.90 | 1.72 |
| MKL `lu` | 2048 | 0.98 | 1.00 | 1.00 | 1.00 |

(ratios after/before, > 1 is slower; full grid in `faer-123-results/after/*.csv` with the baseline
in `baseline/summary.csv`.)

Two causes, both visible in the table:

- **The product is split even when the look-ahead cannot open** (1T/4T: `par_degree < 4`), and two
  product calls over a column-split trailing block cost 13–30% at one thread. The split has to be
  free when it buys nothing, and it is not: the far call loses the level's blocking.
- **At 8T the overlap is far worse than the sum of its parts** (1.8x): the child's section keeps
  `p/2` participants, so both the child and the far product run on half the team, and the child's
  own region — which was already the level's most parallel stage — loses more to the narrowing than
  the overlap hides. The earlier step model reached the same conclusion (a wash at `n = 2048`, 8
  participants) and the measurement says it is worse than a wash.

The issue's target (>= 10% at 8T and 32T) was therefore not met by a wide margin, and the plan's own
rule ("if the measured win is below 5%, do not ship it") applies.

## Verification conclusions and constraints

- What stays: `par_identity.rs` factors 26 shapes (9..1024, dense around the 8-wide leaf and at
  sizes whose half does not round back to itself) at widths 1, 2, 4 and 8 for `f64`, `f32`, `c64`
  and `c32` and requires the same permutation and the same factor bits as `ParThreshold::Never`.
  It found two real defects in the attempt (a missing wait for a non-leaf child, and a raw child
  view over all `ncols` columns instead of the level's own `start..end` span, which is the same
  thing only at the top level) and it now guards the unchanged code.
- Limitations: the 32T fork cell is not in the table because `fbench` refuses to label any row at
  32T on this host — MKL 2026.2 clamps itself to the 16 physical cores, so no vendor row exists to
  control the cell. The probe measurement of the attempt at 32T was not completed once the 1–16T
  grid showed the direction.
- What the numbers still say about the next attempt: the product at 5.1x of 8 is the largest single
  loss (about 2.5 ms of the 15.5 ms at `n = 2048`) and no scheduling change in this issue's option
  set touches it — it is the kernel's own occupancy. The serial chain (leaf + panel solve + swap,
  about 7 ms at 8T) is the part a look-ahead can hide, and hiding it requires the child to keep the
  participants it already uses, not half of them.
- Measured revisions: fork `031a94a` (the attempt) and `e1c83482` (baseline), upstream `0539947f`
  (`faer-v0.24.4`); harness `faer-benchmark` `712a5de` (local `colmat` rename, unpublished), MKL
  2026.2.
