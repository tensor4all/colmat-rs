# colmat-rs#275 evidence: f64 square QR, one core

Host: AMD Ryzen AI 9 HX 470 (Zen 5), Linux, rustc 1.96.0, Julia 1.13.1. Issue host: znver3.
Pinned to one core with `taskset -c 5` (the Julia job and the Rust job never shared a core).
Timed region is the factorization only (`copyto!` / `copy_from` of the input is outside it).
Median of 15 batches, each the mean of 200 calls; min of batches in brackets.

Colmat: `qr_in_place`, `Par::Seq`, block size 8 (`recommended_block_size`), main `475e2842`.
PureBLAS: `PureBLAS.geqrf!`, version 0.1.2 from `el-oso/PureBLAS.jl` master (not the issue's pinned
Colmat.jl revision `b376b3ce`; that pins the whole benchmark suite, and I did not run it).
OpenBLAS: Julia's `LAPACK.geqrf!`, `OPENBLAS_NUM_THREADS=1`.

| n | colmat us | PureBLAS us | OpenBLAS us | colmat / PureBLAS | issue kernel ratio (znver3) |
|---:|---:|---:|---:|---:|---:|
| 96 | 78.9 (78.8) | 76.5 (76.1) | 163.3 (161.6) | 1.03 | 66.6 / 38.6 = 1.73 |
| 128 | 162.6 (162.4) | 151.2 (150.7) | 360.6 (359.7) | 1.075 | 115.6 / 74.4 = 1.55 |

The issue's API-level factor ratios (`qr(A)`, 1.12 and 1.16) are not reproduced either at n=96 and are
within noise of the n=128 kernel ratio. PureBLAS is 2x slower on this host than on the issue's
(76.5 vs 38.6 us at n=96), colmat 18 % slower (78.9 vs 66.6); the absolute times are not comparable
across hosts, only ratios.

## Stage attribution (colmat, temporary timers, not committed)

`qr_in_place_small` (taken for both sizes: `small_path` is true, area <= 128 x 128) split at the end
of the reflector sweep:

| n | total us | reflector sweep us | compact-WY block factors `H` us |
|---:|---:|---:|---:|
| 96 | 78.9 | 71.5 (91 %) | 5.3 (6.7 %) |
| 128 | 162.6 | 151.1 (93 %) | 7.9 (4.9 %) |

(Sweep and `H` include two `Instant::now()` pairs per call, a few tens of ns.) `geqrf!` produces no
block factors, so `H` is work colmat does that PureBLAS's number does not: removing it entirely would
move the ratio at n=128 from 1.075 to about 1.02. The sweep is where the remaining time is.

Instruction counts could not be taken: `perf_event_paranoid=4` on this host and no valgrind, and the
setting cannot be changed without root. So "more work" versus "more overhead" is not settled by
counters here; the n=96 to n=128 time ratio (2.06) against the flop ratio (2.37) shows no
size-independent overhead term in either implementation.
