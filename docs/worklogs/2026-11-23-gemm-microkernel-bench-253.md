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
  `ccold` (the issue's E0(a), (b) and (c); `cstream` walks C down column blocks, which the prefetcher
  can follow, and `ccold` defeats the prefetcher. Whether `cstream` matches the order
  `private_gemm_x86` visits C is not verified: its 1T path can traverse row-major). The production tiles (48x4, 24x8, 12x4) also
  run with one row fewer, which exercises the masked load/store path, and with B passed unpacked
  at `ldb` = 4000 and 4096. Unpacked B is what `gemm` hands the kernel whenever `pack_rhs = tall`
  is false, so these rows measure the kernel side of H2/E1.
- **Statistic.** Each case reports the median and IQR of per-sample FLOP/cycle and ns/call, over
  31 samples of about 10 ms each. A 200 ms time-based priming spin runs before every case.
- **Cycles without a counter.** evo runs with `perf_event_paranoid = 4`, and no unprivileged
  perf event opens there. The probe tries a user-mode `PERF_COUNT_HW_CPU_CYCLES` first. When the
  kernel refuses, it falls back to wall time multiplied by a core clock, measured before and after
  every case with a chain of dependent register `add`s, and the header records which source was
  used. One check that the fallback is right: the measured FMA ceiling comes out at 16.03 (`zmm`)
  and 16.01 (`ymm`) FLOP/cycle, IQR within ±0.02. That is an integer number of FMAs per cycle,
  which a wrong clock would not give. When a counter does open, it is read with its enabled and
  running times, and a sample the event did not count for its whole interval (multiplexed) or that
  counted zero is rejected: the row prints `-` for FLOP/cycle and names the rejection and the
  enabled/running multiplier, and the probe exits 4. A partial count would otherwise inflate
  FLOP/cycle by that multiplier, because the FLOP numerator covers the whole interval (#278 review;
  unit-tested on parsed samples, since evo cannot open the counter).
- **Pins.** A build run of `scripts/gemm-microkernel-bench.sh` writes a build record next to the
  saved binary (`<binary>.build`): the binary's full sha256, the git head and the count of
  modified tracked files at build time, the `rustc` version and the `Cargo.lock` entry of the
  kernel crate (the binary also prints it, read from the lock at build time). A paired rerun with
  `--bin` needs that record with a matching sha256 and copies it, so `meta.txt` reports the
  build's `build_*` lines whichever checkout invokes the rerun, and names that checkout separately
  (`invoker_git_*`). It also records the host settings (governor, boost, frequency cap,
  `perf_event_paranoid`). The run is pinned to one CPU, chosen only by the script's `--cpu` (a
  `--cpu` among the forwarded probe arguments is refused, and a run whose probe reports another
  pinned CPU is discarded), and the script discards it when that core's SMT sibling was at least
  2% busy. `scripts/test-gemm-microkernel-bench.sh` tests these guards with a stub probe.
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
  | AVX-512 48x4 | 15.73 [15.72, 15.74] (98.2%) | 15.96 [15.96, 15.97] (99.6%) | 15.88 [15.86, 15.89] (99.0%) | 15.48 [15.47, 15.50] (96.6%) |
  | AVX-512 24x8 | 15.84 [15.83, 15.85] (98.8%) | 15.97 [15.96, 15.98] (99.6%) | 15.90 [15.90, 15.91] (99.2%) | 15.51 [15.50, 15.51] (96.7%) |
  | AVX2 12x4 | 15.67 [15.66, 15.68] (97.9%) | 15.81 [15.81, 15.82] (98.8%) | 15.69 [15.68, 15.69] (98.0%) | 14.20 [14.19, 14.21] (88.7%) |

  What this means for the experiments that follow, on this host:
  - E3 (prefetch C) has at most about 3% of kernel-level headroom on the AVX-512 tiles, and about
    11% on the AVX2 12x4 tile. On the AVX2 tile a cold C costs 10% (15.81 → 14.20), because each call is 4x shorter
    (1.56 µs against 6.19 µs at kc 512) and hides less of the miss. That makes the AVX2 tile the
    place to look for H1/H3.
  - E4 (prefetch A) and the tile-shape experiment have at most 0.4–2% of kernel-level headroom
    here.
  - A whole-GEMM gap on evo therefore lies outside the kernel, in packing, blocking or the team
    (E1, E2), as the issue anticipated.
- **Masked rows cost only the idle lanes.** 48x4 with 47 rows takes the same time per call as with
  48 (6181.9 against 6185.2 ns), so its FLOP/cycle drops by the idle lane alone (97.5% of the
  ceiling, against 47/48 = 97.9%).
- **H2 at the kernel level: no resolvable effect.** B passed unpacked at `ldb` 4096 against 4000
  differs by at most 0.14% at `l2` and at most 0.58% at `cstream` (run 2's 48x4 pair at `l2` is
  left out: its `ldb` 4000 row is flagged `unstable`, clock 1.61 → 1.99 GHz). Some pairs have
  disjoint IQRs, but the sign is not reproducible between the two runs at `cstream`: 12x4 has 4096
  0.58% faster in run 1 and 0.47% slower in run 2, and 48x4 and 24x8 flip as well. So a power-of-two `ldb` has no
  effect at this level that rises above the run-to-run spread, once B is in cache. H2's TLB and
  set-conflict costs, if they exist, belong to the whole-GEMM run of E1, which needs a paired,
  interleaved design to see effects of this size.
- **Edge kernels.** The kernels with `n` = 1..3 and short `m` are limited by loads and broadcasts,
  not FMAs: 4.2 FLOP/cycle at 8x1 and 11.7 at 48x1, both at l2. They serve the edges of a product,
  not its bulk.
- **Repeatability.** The second run reran the saved binary (sha256 `c9f826ff…`, built at
  `972b43d8` with the tree clean). The core clock dipped to 1.6–1.7 GHz inside a few cases in
  this pair, against the 2.0 GHz cap, with the host otherwise idle. Run 1 flagged four cases
  `unstable` (`avx512/512x4/24x1/l1`, `avx512/512x8/16x5/cstream`, `avx2/256/8x3/cstream`,
  `avx2/256/4x4/l1`) and run 2 two (`avx512/512x4/8x1/l1`, `avx512/512x4/48x4/l2/direct:4000`).
  Over the 219 rows that neither run flagged, the runs differ by 0.39% on average and by at most
  5.1% (`avx512/512x4/48x1/cstream`; the 48x1 rows are the noisiest), and the flagged rows differ
  by up to 10.3%. So the 1% clock check before and after a case does not catch every dip inside
  one. The production-tile cells in the tables are unflagged in run 1 and agree with the
  previous instrument's run (`0b94358c`, sha256 `135d89b6…`) within 0.5%. All 223 correctness
  smokes pass in both runs. The SMT sibling was 0.19% and 0.11% busy. This is a within-host scan,
  not a paired comparison, and no candidate is involved.
- **Not verified.** The cycle source is the add-chain fallback, not a hardware counter. It is
  consistent with the 2.0 GHz cap and with the integer FMA ceiling, but it has not been
  cross-checked against `perf` on this host. Nothing here says how the same tiles behave on a
  512-bit datapath.

Evidence: [`evidence/2026-11-23-gemm-microkernel-253/`](evidence/2026-11-23-gemm-microkernel-253/)
(`run1-*`, the kept run, built at commit `972b43d8` with the tree clean, after the #278 review
fixes; `run2-*`, the same binary rerun with `--bin`). The previous instrument's runs (`0b94358c`)
are in this file's history. The saved baseline binary for E1–E4 stays on evo next to the run,
with its build record, at `~/projects/tensor4all/colmat-253-e0-evo-fix-run1/bin/`.
