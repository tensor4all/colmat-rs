# Independent EVD merge repairs (#84)

## Change and status

Experimental branch `experiment/eigh-parallel-repair`, based on
`cb3822e9f66f281326307ead44abdb2b31e38843`. Serves
[tensor4all/colmat-rs#84](https://github.com/tensor4all/colmat-rs/issues/84).
Only the divide-and-conquer merge's weight repair and repaired eigenvector
column construction are parallelized. Tridiagonal reduction is unchanged.
No profiling counters, diagnostic flags, public APIs, new pools or unsafe code
are included. Each item retains its previous scalar arithmetic order.

One worker owns one weight or a complete output column. Shared inputs are
immutable. Work is O(k²) and O(n k), respectively; no per-worker partial
matrix, serial reduction or additional scratch is introduced. For p workers,
private state is O(p), and the existing O(k) input vectors are read by workers.
The caller-owned team and existing dispatch-cost model select the payable
width, capped by independent items rather than a fixed thread count.

## Evidence and isolation

Independent profiling and paired diagnostic experiments are preserved in the
sibling colmat-benchmark checkout:

- `data/probes/20261114T072415Z-evd-repair-profile-84/`: baseline phase profile
  and calibration.json (root n=512, actual non-deflated k=465).
- `probes/evd-repair-par-84/`: measured source snapshot, exact patch, driver,
  correctness checks and aggregation script.
- `data/probes/20261114T074741Z-evd-repair-par-84/`: raw samples, process guards,
  manifest, paired-process.csv and summary.csv.

The local Xeon E5-2699 v3 (`witch`) has 18 physical cores and one shared L3.
User authorized a rough local scan because evo/broccoli were unavailable.
Turbo and schedutil remained enabled. These are reference measurements,
not the campaign adoption arm. Each cell has three fresh processes; within
one process nine timed calls per block, original/candidates/original, and
per-process paired ratios aggregated by their median. No failed guard was
rerun or replaced. The two 512/8T inputs passed all three idle endpoint guards.

For mixed deterministic f64 input, n=512/8T, solver repair alone took
20.310 ms versus original 27.163 ms; median paired ratio 0.7502 (24.98% less
whole-call time). The modular holdout gave ratio 0.7728 (22.72% less).
Tridiagonal suffix alone gave ratios 0.9593 and 0.9595; combining it with
repair gave 0.7384 and 0.7624. Thus 25% and 26% are both relative to the
original, not successive reductions. The suffix is deliberately excluded.
At 512/16T repair alone gave 0.6883, but only one of three processes passed
the idle guard; this cell does not establish a clean scaling result.

## Experimental cost estimates

Weight and vector estimates are 4.456651636 and 14.712088374 ns per k² and
n k unit, from the independent three-process single-thread f64 profile.
They are host-only estimates, intentionally kept local to this experimental
module rather than presented as calibrated Zen5/M5/global rate-table entries.
Single precision currently uses the same estimate and is not performance
calibrated. Timing on other CPUs, four scalar families, 64-core scaling and
matched OpenBLAS/MKL campaign comparisons remain unverified. Adoption requires
canonical-host calibration and the normal campaign acceptance arm.

Earlier reduction-width scans and the slower blocked reduction are excluded.
Their independent evidence remains under the sibling benchmark data/probes
tridiag-width, tridiag-model, tridiag-grain and tridiag-blocked records.
No proprietary implementation internals were consulted. This patch parallelizes
existing colmat arithmetic.

## Verification

The isolated measured prototype passed 823 library tests (162 existing ignored),
the eigenvector allocation integration test, and 1,152 focused numerical calls
across f32/f64/c32/c64, including 576 repair-only bit comparisons.
The branch adds a private focused test of f32/f64 repairs: sequential versus
2/4/8-worker execution, empty/fully deflated/partially deflated cases, poisoned
output columns, and exact bits. Complex EVD uses this same real repair kernel.
Final branch checks are recorded below after completion.

An additional process-wide allocation check on the complete dense 256/4T EVD
found four allocations after warm-up, both with the candidate and with mode 0
(original arithmetic) of the frozen diagnostic baseline. Evidence:
`/tmp/eigh-parallel-repair-alloc.log` and
`/tmp/eigh-repair-baseline-alloc.log`; the baseline driver is preserved as
`probes/evd-repair-par-84/snapshot/faer/tests/eigh_repair_allocation_baseline.rs`
in the sibling benchmark checkout. This is an inherited dense-path limitation,
not evidence of zero allocations for the whole parallel dense decomposition.
Its source has not been isolated, and it is outside this solver-only change.
The committed allocation regression instead calls the public tridiagonal
solver directly at 4/8T, isolating the changed solver from dense reduction
and back transformation, and counts allocations on every worker.

Final branch verification (release, locked):

- `cargo test --release --locked -p colmat --lib --test evd_svd_vectors_no_alloc`:
  824 library tests passed, 162 existing ignored; one sequential allocation
  integration test passed. Log `/tmp/eigh-parallel-repair-tests-final.log`.
- `cargo test --release --locked -p colmat --test no_step_alloc`: passed,
  including the new 4/8T tridiagonal-solver cases with zero counted allocations.
  Log `/tmp/eigh-parallel-repair-alloc-solver.log`.
- `git diff --check`: passed. No formal performance acceptance is claimed
  for this cleaned branch; the earlier timing numbers are the isolated probe's.
