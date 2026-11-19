# `eigh` at 4T/8T: team members ran scalar code with a dirty upper vector state

Issue: [tensor4all/colmat-rs#238](https://github.com/tensor4all/colmat-rs/issues/238). Follows
[`2026-11-19-lu-eigh-fullpiv-shapes.md`](2026-11-19-lu-eigh-fullpiv-shapes.md) (#235).

Host: the development workstation (`zen5-cpu`, AMD Ryzen AI 9 HX 470, 12 cores / 24 threads, CPU boost
off, `performance` governor). Measured CPUs 4 / 4-7 / 4-11 (the 8-core L3 domain), observers and
every other process on 0-3,12-15. Development-host numbers, provisional. No upstream revision was
measured beyond the `faer-upstream` control row the harness carries. MKL is timed through its public
LAPACK entry points only (`dsyevd`, and `dsytrd`/`dstedc`/`dormtr` for the stage comparison).

## Decisions

- **The issue's candidates are not the cause.** Stage breakdown of f64 `self_adjoint_evd` (probe
  `faer-benchmark/probes/eigh-stages-238`, the same matrix for both libraries), 4T, n = 512:
  reduction to tridiagonal 5.5 ms against `dsytrd` 8.9, back-transformation 2.6 against `dormtr`
  6.0, tridiagonal divide-and-conquer 15.5 against `dstedc` 5.9. The reflector block size and the
  reduction's panel belong to stages that already win; the loss is the D&C. At 1T n = 256 the call
  was level with MKL after #235 (9.55 against 9.82 ms) and has no team member.
- **Inside the D&C, the merges' secular-equation roots did not scale.** The top merge's roots
  (k = 465) took 7.1 ms at 4T and 7.2 ms at 1T. Per root, the leader's block ran at about 16 us and
  every other member's at about 60 us, on any CPU pair.
- **Cause: the x86 GEMM kernels return with a dirty upper vector state.** `t4a-private-gemm-x86`'s
  kernels are generated assembly that use `ymm`/`zmm` registers and never execute `vzeroupper`. The
  leader goes on in compiled code, which issues `vzeroupper` when it leaves an AVX function; a
  member goes back to `threadteam`'s wait loop, which has no AVX code, and starts its next job
  dirty. There every legacy-SSE scalar instruction (the default x86-64 target's scalar `f64`)
  carries a false dependency on its destination's upper bits. On a member, inside the root job:
  a dependent chain of 10^6 divisions ran as on the leader (13.5 vs 12.5 ms); 1000 independent
  `secular_eq` passes ran 3.7x slower (3.6 vs 0.97 ms), also on `d`, `z` copied by the member; one
  `vzeroupper` before the root removed the difference. `XGETBV(1)` reports `YMM_Hi128`/`ZMM_Hi256`
  in use on 4 of 4 members after a team GEMM. A plain `std::thread` running the sequential solve
  beside the main thread was not slowed. (`perf` is unavailable to the user here:
  `perf_event_paranoid` = 4.)
- **Chosen: every participant clears at the end of its job in a team region.**
  `team::end_of_job` (`vzeroupper` behind an AVX detection) runs after each share in
  `TeamFork::run` (the kernel's stages), `team::run`, `deal`, `join_raw` and `split`: one
  instruction per participant and region. A sequential call executes nothing new; no size, dtype or
  width enters the rule. The root fix belongs to the kernel crate (its assembly should end with
  `vzeroupper`, as compiled AVX code does); that change to `t4a-private-gemm-x86` is not made here.
- **Rejected: clear after every kernel call** (the branch's first commit, `f3f822db`). The same
  `eigh` gain, but it puts an instruction on every sequential product. Its campaign pair showed
  C-order `cholesky` n = 32 1.16-1.24x and c64 `svd_values` 1.05-1.07x slower, 1T included. A probe
  did not reproduce either for that commit, and showed the same 7 % on the chosen commit's
  sequential c64 bidiagonalisation (10.65 -> 11.44 ms, n = 256), where no changed code runs;
  rebuilt with `-C llvm-args=-align-loops=64`, base and the chosen commit give 10.70 and 10.69 ms.
  Those cells follow each binary's loop placement, not the clear. (Probe `faer-benchmark/probes/regress-238`; the
  per-root and per-merge timers are `eigh-stages-238/instrumentation.patch`, not part of the change.)

## Verification conclusions and constraints

- `members_start_jobs_with_clean_upper_vector_state` reads `XGETBV(1)` on every member of a team
  of 4 after three team products and after a `team::run` job that ends in a sequential product; it
  fails with 4 of 4 members dirty when the clear is disabled, and skips on a host without the x86
  kernel or without `XGETBV` ECX = 1.
- The arithmetic is untouched; every result keeps its bits.
- Scaling: the cost is one instruction per participant per region, independent of the problem;
  the effect removed is per member, so it should matter more on wider teams. Not measured: the
  64-core host, the OpenBLAS arm, f32/c32.
- A job that leaves dirty state through a path other than these dispatch points (a direct
  `threadteam` call outside `utils::team`) is not covered; none exists in `faer/src` today.

## Campaign pair

Recorder `record_run.py zen5-cpu`, MKL arm (oneAPI 2026.2, Intel OpenMP), `--reps 5 --aa 3
--prime-ms 250`, f64 and c64, C and F order, measured CPUs 4 / 4-7 / 4-11, recorder and every other
process on 0-3,12-15, boost off. Baseline `origin/main` `0ad25fe1`, candidate `fea0c056` (the
later commits on the branch are docs only). Runs under `colmat-benchmark/data/results/zen5-cpu/`,
recorded with `--no-publish --allow-local-commits` (the harness checkout was on an unpushed publish
commit), so they are acceptance evidence, not a published page:

- daily subset, `linalg-small`/`linalg-mid`/`gemm-mid`, 1/4/8T, families `eigh`, `svd_values`,
  `svd_thin`, `qr_factor`, `lu`, `cholesky`, `gemm`: `faer-kernels/20261119T063148Z-i238-base`
  against `20261119T084601Z-i238v2-cand`;
- boundary tier `linalg-boundary` (255/257/511/513), 4/8T, `eigh`, `svd_values`, `svd_thin`, `lu`,
  `qr_factor`: `faer-kernels-large/20261119T093624Z-i238v2-large-base` against
  `20261119T100425Z-i238v2-large-cand`.

`candidate_compare.py`: the `faer-upstream` control reproduces the baseline (median 1.00005 over 864
daily cells, 0.9996 over 288 boundary cells). Fork, beyond 5 %: daily 84 faster (all `eigh`), 11
slower; boundary 68 faster (`eigh` 4T/8T, four `lu` 4T), none slower. The 11 are `cholesky` cells
of 2.5-12 us whose upstream control moved 0.89-1.10 in the same cells, and three cells whose
medians moved under 0.3 %. Per family, the guards' median candidate/base ratio is 0.986-1.004 at
every thread count.

| `eigh` f64 F | base ms | cand ms | cand/base | fork/MKL before | after |
|---|---|---|---|---|---|
| 1T 256 | 9.550 | 9.566 | 1.002 | 0.977 | 0.977 |
| 4T 255 | 5.824 | 3.502 | 0.601 | 1.331 | 0.803 |
| 4T 256 | 5.626 | 3.341 | 0.594 | 1.214 | 0.722 |
| 4T 257 | 5.744 | 3.536 | 0.616 | 1.294 | 0.796 |
| 4T 511 | 24.591 | 15.795 | 0.642 | 1.294 | 0.831 |
| 4T 512 | 24.708 | 15.562 | 0.630 | 1.167 | 0.733 |
| 4T 513 | 25.025 | 16.131 | 0.645 | 1.306 | 0.843 |
| 8T 256 | 3.933 | 2.448 | 0.622 | 1.027 | 0.636 |
| 8T 512 | 16.362 | 10.337 | 0.632 | 1.135 | 0.718 |

`eigh` at 1T and at n = 32 does not move (0.986-1.008). `svd_values`/`svd_thin` do not move either.
At 1T below n = 256 the f64 call is still 1.10-1.20x MKL's; that is the 1T small-size gap of #234,
not this change.
