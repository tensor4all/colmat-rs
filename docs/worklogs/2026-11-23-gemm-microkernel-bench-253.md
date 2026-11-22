# Microkernel-level benchmark of the x86 f64 GEMM kernels (#253, E0)

Issue [#253](https://github.com/tensor4all/colmat-rs/issues/253) orders its work as: a
microkernel benchmark first (E0), then four cheap experiments (E1–E4: always pack B, raise `kc`,
prefetch C, prefetch A), and an AOCL port only if the evidence calls for one. This record covers
E0 only. No library code changes.

## Decisions

- **Time one generated kernel, nothing around it.** The probe calls one entry of
  `t4a-private-gemm-x86`'s `F64_SIMD512x4`, `F64_SIMD512x8` and `F64_SIMD256` tables through the
  crate's public `call_microkernel`, with `lhs = null` and `rhs = null`. With null pointers the
  kernel reads the packed panels, so no packing, blocking loop, team or `matmul` dispatch runs in
  the timed region. The kernel list is rebuilt in `build.rs`'s generation order, and the probe
  asserts each table's length. If the generator changes, the probe stops instead of timing a
  different kernel under the old name. The packed layouts come from the generator's prologue: A
  holds `mr` values per depth step, with `lhs_cs = m * simd bytes`, and B holds `n` values per depth
  step, with `rhs_rs = n * 8`.
- **Shapes.** The suite runs every generated tile at four residencies, `l1`, `l2`, `cstream` and
  `ccold` (the issue's E0(a), (b) and (c); `cstream` is the realistic C walk, which the prefetcher
  can follow, and `ccold` defeats the prefetcher). The production tiles (48x4, 24x8, 12x4) also
  run with one row fewer, which exercises the masked load/store path, and with B passed unpacked
  at `ldb` = 4000 and 4096. Unpacked B is what `gemm` hands the kernel whenever `pack_rhs = tall`
  is false, so these rows measure the kernel side of H2/E1.
- **Statistic.** Each case reports the median and IQR of per-sample FLOP/cycle and ns/call, over
  31 samples of about 10 ms each. A 200 ms time-based priming spin runs before every case.
- **Cycles without a counter.** evo runs with `perf_event_paranoid = 4`, and no unprivileged
  perf event opens there. The probe tries a user-mode `PERF_COUNT_HW_CPU_CYCLES` first. When the
  kernel refuses, it falls back to wall time multiplied by a core clock, measured before and after
  every case with a chain of dependent register `add`s, and the header records which source was
  used. One check that the fallback is right: the measured FMA ceiling comes out at 16.00 FLOP/cycle
  (IQR ±0.01). That is an integer number of FMAs per cycle, which a wrong clock would not give.
- **Pins.** `scripts/gemm-microkernel-bench.sh` records the git head and the count of modified
  tracked files, the `rustc` version, the `Cargo.lock` entry of the kernel crate (the binary also
  prints it, read from the lock at build time), and the binary's sha256. It saves a copy of the
  binary, so a later experiment can be paired against that very binary. It also records the host
  settings (governor, boost, frequency cap, `perf_event_paranoid`). The run is pinned to one CPU,
  and the script discards it when that core's SMT sibling was at least 2% busy.
- **Not in this PR.** The `llvm-mca` comparison is left out because no `llvm-mca` is installed on
  evo. The AOCL zen5 reference kernel on the same packed inputs is left out too: it is the
  decision-gate instrument, and gets its own scoped PR once E1–E4 are done.

## Verification conclusions and constraints

- **evo is a 256-bit-datapath Zen5.** The host is a Ryzen AI 9 HX 470 (mobile Zen5, four Zen5 and
  eight Zen5c cores; measured on CPU 2, a Zen5 core), with boost off and a 2.0 GHz cap. The
  measured FMA ceiling is 16.0 FLOP/cycle for both `zmm` and `ymm`, which is 50% of the issue's
  nominal AVX-512 peak of 32. The issue itself says mobile Zen5 "is not a substitute" for
  EPYC Turin. Kernel-quality conclusions for a 512-bit datapath need a Ryzen 9000 or EPYC Turin
  host.
- **The production tiles are at the ceiling of this core.** The table below gives FLOP/cycle as
  the median [Q1, Q3], and in brackets the percentage of the measured FMA ceiling:

  | tile | l1 | l2 (kc 512) | cstream | ccold |
  |---|---|---|---|---|
  | AVX-512 48x4 | 15.72 [15.71, 15.73] (98.2%) | 15.98 [15.96, 15.99] (99.8%) | 15.93 [15.91, 15.94] (99.5%) | 15.51 [15.50, 15.51] (96.9%) |
  | AVX-512 24x8 | 15.84 [15.81, 15.85] (98.9%) | 15.96 [15.95, 15.96] (99.7%) | 15.86 [15.85, 15.88] (99.1%) | 15.49 [15.47, 15.50] (96.7%) |
  | AVX2 12x4 | 15.61 [15.60, 15.62] (97.6%) | 15.80 [15.79, 15.81] (98.8%) | 15.75 [15.74, 15.76] (98.5%) | 14.17 [14.15, 14.18] (88.6%) |

  What this means for the experiments that follow, on this host:
  - E3 (prefetch C) has at most about 3% of kernel-level headroom on the AVX-512 tiles, and about
    11% on the AVX2 12x4 tile. On the AVX2 tile a cold C costs 10% (15.80 → 14.17), because each call is 4x shorter
    (1.56 µs against 6.18 µs at kc 512) and hides less of the miss. That makes the AVX2 tile the
    place to look for H1/H3.
  - E4 (prefetch A) and the tile-shape experiment have at most 0.2–2% of kernel-level headroom
    here.
  - A whole-GEMM gap on evo therefore lies outside the kernel, in packing, blocking or the team
    (E1, E2), as the issue anticipated.
- **Masked rows cost only the idle lanes.** 48x4 with 47 rows takes the same time per call as with
  48 (6180.8 against 6184.8 ns), so its FLOP/cycle drops by the idle lane alone (97.6% of the
  ceiling, against 47/48 = 97.9%).
- **H2 at the kernel level.** B passed unpacked at `ldb` 4096 against 4000 differs by at most 0.2%
  (48x4 at `l2`: 15.95 against 15.98), and the cstream pair is inside each other's IQR. At this
  level, a power-of-two `ldb` costs nothing measurable once B is in cache. H2's TLB and set-conflict
  costs, if they exist, belong to the whole-GEMM run of E1.
- **Edge kernels.** The kernels with `n` = 1..3 and short `m` are limited by loads and broadcasts,
  not FMAs: 4.3 FLOP/cycle at 8x1 and 11.6 at 48x1, both at l2. They serve the edges of a product,
  not its bulk.
- **Repeatability.** A second run of the same saved binary (sha256 `135d89b6…`) differs from the
  first by 0.29% on average over 225 rows, and by at most 1.57% (`avx2/256/8x1/l1`). All 223
  correctness smokes pass in both runs. The SMT sibling was 0.09% and 0.10% busy. Run 1 flagged one
  case `unstable` (`avx2/256/8x4/l1`: core clock 1.956 → 1.989 GHz across the case); run 2 flagged
  none. This is a within-host scan, not a paired comparison, and no candidate is involved.
- **Not verified.** The cycle source is the add-chain fallback, not a hardware counter. It is
  consistent with the 2.0 GHz cap and with the integer FMA ceiling, but it has not been
  cross-checked against `perf` on this host. Nothing here says how the same tiles behave on a
  512-bit datapath.

Evidence: [`evidence/2026-11-23-gemm-microkernel-253/`](evidence/2026-11-23-gemm-microkernel-253/)
(`run1-*`, the kept run, from commit `0b94358c` with the tree clean; `run2-*`, the same binary
re-run). The saved baseline binary for E1–E4 stays on evo next to the run, at
`~/projects/tensor4all/colmat-253-e0-evo/bin/`.
