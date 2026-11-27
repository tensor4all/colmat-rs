# `self_adjoint_evd` values only: DSTERF-style iteration (colmat-rs#287)

Status: **draft proposal** — the eigenvalues change in the last bits (see *Decision needed*).

## Reproduction

Host: AMD Ryzen AI MAX+ 395 (Zen 5), CPU 4 pinned with `taskset`, `Par::Seq`, release,
`-C target-cpu=native`. The issue's host is a Ryzen 9 PRO 8945HS (Zen 4), a different host.
Reference: system OpenBLAS 0.3.26 (Ubuntu), `dsyevr('N','A','L', ABSTOL=-1)`, one thread, raw
Fortran call. Same input as the issue (`A = B B^T + n I`, lower triangle), scratch preallocated
outside the timed region, minimum of 30/10/4 repetitions (n<=64 / 128 / >=256), **median of 15
batches, 3 interleaved rounds**. Colmat base: `475e2842` (origin/main).

| n | colmat main us | OpenBLAS us | main/OpenBLAS | issue ratio (Zen 4, OpenBLAS) |
|---:|---:|---:|---:|---:|
| 32 | 46.4 | 33.0 | 1.41 | 1.44 |
| 64 | 168.2 | 123.6 | 1.36 | 1.41 |
| 128 | 648.3 | 468.4 | 1.38 | 1.43 |
| 256 | 2853.2 | 1972.6 | 1.45 | 1.41 |
| 512 | 14148.4 | 9858.2 | 1.44 | 1.36 |

**It reproduces**, flat in n, as reported. (The PureBLAS.jl reference of the issue was not
available on this host and was not measured.)

## Attribution by stage

`dsyevr('N','A')` on a full-range values-only request is `dsytrd` followed by `dsterf` (the
documented LAPACK driver structure). Timed separately (OpenBLAS) against colmat's public
`tridiag_in_place` and the remainder of `self_adjoint_evd`:

| n | tridiag colmat | `dsytrd` | QL stage colmat (total - tridiag) | `dsterf` | QL colmat/`dsterf` |
|---:|---:|---:|---:|---:|---:|
| 32 | 14.2 | 12.5 | 32.1 | 17.9 | 1.8 |
| 128 | 206.7 | 200.1 | 437.3 | 233.0 | 1.9 |
| 256 | 1145.3 | 1000.1 | 1724.0 | 884.5 | 1.95 |
| 512 | 7383.0 | 6143.1 | 6572.6 | 3311.5 | 2.0 |

Tridiagonalisation is at parity (1.0-1.2x); **the whole gap is the eigenvalue iteration**, about
2x at every n. The vector path is not behind because it needs the rotations anyway.

## Which kind of gap: same family, different formulation

Both are implicit QL/QR iterations with a Wilkinson-type shift; neither uses divide-and-conquer
for values only (`dsyevr` with no vectors takes `dsterf`; colmat's D&C is vector-only and the
`u = None` branch calls `qr_algorithm` directly). So this is **not** a different algorithm class
that cannot be closed. It is a different formulation of the step, and the constant is:

* colmat (`qr_algorithm`, DSTEQR's QR branch): per rotation `make_givens` = a divide, a
  `sqrt(1+t^2)` and a reciprocal, then the diagonal update; the next step depends on it, so the
  chain is latency-bound. Measured 222 742 rotations at n=512 (806 sweeps) -> **29.5 ns per
  rotation** (6572.6 us / 222 742).
* DSTERF (Pal-Walker-Kahan): iterates on the squares of the off-diagonals, forms no rotation, no
  `sqrt` and no `hypot` in the step (`c = p/r`, `s = bb/r`, `p = gamma^2/c`).

Probes (diagnostics, not kept): replacing the per-sweep `hypot` deflation scan by `|a|+|b|`
moved n=128 values 644 -> 613 us (5 %), so the scan is a minor term; counters showed the cost
is the step itself. `perf_event_paranoid = 4` on this host, so no instruction counts; the
attribution is by timing and operation counts.

## Change

`tridiag_evd::sterf`, a values-only PWK iteration (QL and QR branches, 2x2 blocks via the DLAE2
formula, `normalize_range` for the range, 30 n iteration cap, ascending sort), used where
`self_adjoint_evd` and `tridiagonal_self_adjoint_evd` have `u = None`. Vector paths unchanged.
Source formulation: Reference-LAPACK v3.12.1 `SRC/dsterf.f` (BSD, `faer/COPYING.LAPACK.BSD`),
as `qr_algorithm` already cites `dsteqr.f`. The reference was timed through its public API
only; no vendor internals were inspected.

## Result (same pairing, median of 3 interleaved rounds)

| n | main us | candidate us | OpenBLAS us | main/OB | cand/OB | cand/main |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 46.4 | 33.7 | 33.0 | 1.41 | 1.02 | 0.73 |
| 64 | 168.2 | 116.3 | 123.6 | 1.36 | 0.94 | 0.69 |
| 128 | 648.3 | 454.6 | 468.4 | 1.38 | 0.97 | 0.70 |
| 256 | 2853.2 | 2078.3 | 1972.6 | 1.45 | 1.05 | 0.73 |
| 512 | 14148.4 | 11129.3 | 9858.2 | 1.44 | 1.13 | 0.79 |

n=512's remaining 1.13 is tridiagonalisation (7.4-7.7 ms against 6.1 ms), a separate stage.
Raw: `docs/worklogs/evidence/287-sa-evd-values/raw.txt` (drivers beside it).

## Decision needed

`sterf` and `qr_algorithm` are both backward stable but round differently, so values-only
eigenvalues move by a few ulps relative to `||T||`. Nothing was widened: the new test
`sterf_matches_qr_algorithm` bounds the difference by `4 n eps ||T||` on both branches, graded
spectra, split matrices and the `normalize_range` extremes, and every existing test passes
unmodified. Whether eigenvalue-only results may change bits (below the D&C threshold the values-only
eigenvalues were bitwise the `u = Some` ones, since both ran `qr_algorithm`; they no longer are) is the
maintainer's call; hence a draft. Complex scalars share the path (c64 did not show the gap in
the issue; not re-measured here), and f32 was not measured.

## Not done

Campaign (multi-thread, MKL arm, other hosts, Zen 4), the PureBLAS reference, c64/f32 timings.
