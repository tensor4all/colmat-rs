# Block width of the bidiagonalisation, and where the 512 values call actually goes (#179)

Exploration for [tensor4all/faer#179](https://github.com/tensor4all/faer/issues/179) under the
handover in [#198](https://github.com/tensor4all/faer/issues/198), whose first declared change for
`#179` is the blocking width of `faer/src/linalg/svd/bidiag.rs`. Measured on **broccoli**
(AMD EPYC 7702P, host profile `zen2-cpu`), not on evo; a broccoli number cannot close a bar and no
bar is claimed here.

**Outcome: no library change.** `BLOCK_SIZE = 16` is already the minimum of the swept set at both
bar sizes, the apply share moves in the wrong direction for every wider block, and the 1T→8T
scaling of the values call is 1.42–1.49 whatever the width. The blocking width is not this call's
lever; the evidence below says which stage is.

## Decisions

**The sweep is the width of the blocked panel** (`BLOCK_SIZE`), swept 8, 16, 24, 32, 48, 64 at
`n = 512, 256` for `svd_values` and `bidiag_in_place` (f64), 1T and 8T, with `n = 128, 64` as the
guard cells (`BLOCK_FLOOR = 128` keeps them out of the blocked loop, so a move there would mean the
change leaked). All numbers are medians of three processes of five chronological calls after a
250 ms prime, on the host's own 1T cell (CPU 0) and 8T cell (CPUs 0-7, two CCX of the `zen2-cpu`
profile), observer pinned to CPUs 60-63, idle gate before and after every process, governor
`performance`, boost off.

## The sweep

`svd_values` (whole call) and `bidiag_in_place` (the reduction alone), ms, 512×512 and 256×256 f64,
median of three process medians; the guard row is 128×128 at 8T.

| `BLOCK_SIZE` | 512 1T | 512 8T | 1T→8T | 256 1T | 256 8T | 128 8T (guard) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 52.29 | 36.93 | 1.42 | 7.686 | 7.229 | 1.204 |
| **16 (shipped)** | **50.65** | **33.98** | **1.49** | **7.646** | **7.220** | **1.205** |
| 24 | 51.84 | 34.80 | 1.49 | 7.933 | 7.569 | 1.207 |
| 32 | 53.44 | 35.93 | 1.49 | 8.117 | 7.701 | 1.208 |
| 48 | 56.88 | 38.64 | 1.47 | 9.022 | 8.591 | 1.201 |
| 64 | 59.70 | 42.15 | 1.42 | 9.394 | 8.915 | 1.208 |

`bidiag_in_place` alone at 512 f64: 43.24 (1T) → 26.12 (8T), 1.66×.

Two facts the table settles. **16 is an interior optimum**, not a floor of the scan: 8 costs 8.7 %
at 8T because the apply becomes level-2 again (below), and every wider block costs more in the
panel's own recurrences than it saves in the apply — monotonically, by 24 % at 64. And **the
width does not touch the scaling**: 1T→8T is 1.42–1.49 at every width, against `#198`'s bar of
2.5×. `BLOCK_SIZE = 16` is the best value of this parameter on this host and stays.

The 128/64 guard cells do not move (128 8T: 1.201–1.208 across all six widths, inside the session
spread), which is the expected result of `BLOCK_FLOOR = 128` and confirms the change under test is
confined to the blocked loop.

## The apply share, and what it buys

The instrumented build (`probes/hh179-phases/instrumentation.patch`, feature `probe-timers`, one
accounting domain per run; timers removed from the shipped tree) splits the 512 f64 reduction, so
the trailing update's share is taken inside one run rather than between two builds:

| `BLOCK_SIZE` | 8T apply (ms / share) | 1T apply (ms / share) | 8T reduction total |
| ---: | ---: | ---: | ---: |
| 8 | 9.21 / **31.5 %** | 11.82 / 26.2 % | 29.26 |
| 16 | 4.56 / **17.3 %** | 8.38 / 19.2 % | 26.34 |
| 24 | 3.41 / 12.9 % | 7.51 / 16.8 % | 26.51 |
| 32 | 2.84 / 10.2 % | 7.32 / 15.8 % | 28.01 |
| 48 | 2.12 / 6.9 % | 6.74 / 13.7 % | 31.00 |
| 64 | 1.91 / 5.5 % | 5.80 / 11.0 % | 34.86 |

The apply share does move — 31.5 % down to 5.5 % at 8T — and that is the whole of `#198`'s
hypothesis, measured. It is not enough, for a reason the third column shows: the block's own
recurrence work grows faster than the apply shrinks, so the reduction's total is minimised at 16
where the two cross. `#198`'s premise that "the block is too small for the apply to appear" is
therefore not what limits this call on this host: the apply appears (17.3 % at the shipped width),
it is simply not the dominant stage.

## Where the time actually is (512×512 f64, `BLOCK_SIZE = 16`)

Stage split of the reduction, instrumented, 8T and 1T in the same accounting domain:

| stage | 1T ms | 8T ms | 1T→8T | share at 8T |
| --- | ---: | ---: | ---: | ---: |
| column update + left reflector | 0.86 | 0.97 | 0.89 | 3.7 % |
| `Y` product (`A^H u`, per step) | 14.72 | 6.71 | 2.19 | 25.5 % |
| row gather + update + right reflector | 2.14 | 2.97 | 0.72 | 11.3 % |
| `X` product (`A r^H`, per step) | 16.03 | 9.43 | 1.70 | 35.8 % |
| trailing update (apply), per panel | 8.38 | 4.56 | 1.84 | 17.3 % |
| unblocked tail (128 steps) | 0.75 | 0.81 | 0.93 | 3.1 % |
| compact-factor upgrade | 0.89 | 0.90 | 0.99 | 3.4 % |
| **total** | **43.78** | **26.34** | **1.66** | |

Two consequences for the next change on `#179`:

1. **The two per-step products are 61 % of the reduction at 8T and scale only 1.70–2.19×.** They are
   matvecs of the shrinking trailing block (`A[i.., i+1..]^H u` and `A[i+1.., i+1..] r^H`), so they
   are bandwidth-bound from L2/L3, and the parallel route already runs them on the caller's team —
   the gate is not the problem (`par-trace` on this tree shows the panel gate choosing width 8 for
   the first 24 panels of a 512 call). Making them level-3 is the two-stage/tile reduction `#198`
   puts explicitly out of scope here, so it needs its own issue and its own decision.
2. **21.5 % of the reduction at 8T does not scale at all** (row gather/update/right reflector, the
   128-step unblocked tail which `par-trace` shows gated to width 1 at every step, and the
   compact-factor upgrade). The tail is 3 % of the stage, so the floor is not the lever either —
   measured separately below.

The per-step gates being at width 1 in the tail is not a defect: their areas are 16 129 units and
below, and `utils::cost` says a region of that size does not pay for its sections on this host. It
is recorded because `#198` asked for the share of "what does not scale".

## The floor (secondary, measured after the width, same protocol)

`BLOCK_FLOOR` 64 / 96 / 128 at `BLOCK_SIZE = 16`, ms:

| floor | 512 1T | 512 8T | 256 1T | 256 8T |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 51.08 | 33.77 | 7.968 | 7.531 |
| 96 | 51.03 | 33.98 | 7.788 | 7.486 |
| **128 (shipped)** | **50.75** | **34.05** | **7.645** | **7.199** |

At 512 the three floors are inside each other's spread (33.77 against 34.05, ranges overlapping), and
at 256 the shipped 128 is the best by 4.6 % at 8T. This reproduces the previous scan
(`2026-10-23-blocked-bidiag.md`, "the floor does not matter between 64 and 128" at 512, and floor 64
worse at 256). Nothing to change.

## Rejected

- **Widening the block to show the apply disappear.** Measured: 32 halves the apply share to 10.2 %
  and costs 6 % of the reduction, 64 cuts it to 5.5 % and costs 32 %. The trade is not a tuning
  question, it is the panel recurrence's `O(ib)` growth against the apply's kernel efficiency.
- **A width that varies with the remaining size** (narrow early where the recurrences are wide,
  wider near the end). Both ends of the sweep lose for the same reason at both bar sizes, and the
  optimum is 16 at 512 and at 256, so there is no size at which a second value wins. Adding a
  schedule would need a measurement to justify it that this sweep does not have.
- **A public `block_size` in `BidiagParams`.** Nothing outside this file chooses it, and the sweep
  says the current value is the optimum of the range.
- **Judging `#198`'s bars here.** `512×512` f64 8T `svd_values` fork/MKL `<= 1.00` needs the evo
  MKL arm, and the 1T→8T bar would need a code change that moves scaling — this one does not
  (1.42–1.49 at every width). Both stay open and unclaimed.
- **A paired campaign arm on broccoli.** The harness cannot measure current main: `fbench` declares
  `t4a-faer = ">=0.24, <0.26"` (`fbench/Cargo.toml`) and its `Cargo.lock` names 0.24.4, while
  current main is 0.32.0, and `build_for_faer_rev.sh` refuses the patch resolution rather than
  measure a crate it did not link. Moving that requirement and re-baselining is a harness change
  with its own deliverable, and it would leave no comparable page. So the evidence here is the
  probe below, quoted as a diagnostic with its protocol, and the campaign arm remains for whoever
  moves the pin.

## Verification

- **No library code changed.** The only diff is this record and the `PATCHSET.md` row, so no
  arithmetic, threshold, scratch query or schedule moved, and the numerical checks in the suite are
  unaffected by construction (`git diff --stat` over `faer/src` is empty).
- The probe's matrix, work-copy padding and call structure follow `svd_imp`: the probe replicates
  `work_rows`' 1 KiB-stride padding, because an unpadded 512-row `f64` copy aliases L1 sets and
  costs 3.3 ms at 8T on this host.
- The probe's own A/A repeat (the `BLOCK_SIZE = 16` cell measured in two separate sessions) agrees
  to 0.8 % at 512 8T (33.85 against 34.11 ms), which is the spread the width table is read against.

## What remains unverified

- **evo.** No bar is judged: the fork/MKL ratio and the 2.5× scaling bar both need evo, which this
  session did not have. This record closes no bar and claims none.
- **c64 and f32.** The sweep is f64 only. The constant is shared by every dtype and `c64`'s
  `rate::bidiag()` differs (2.1997 against 1.0427 ns per unit), so a c64 optimum could differ in
  principle; the width's mechanism (panel recurrence against apply efficiency) is dtype-independent,
  so it was not measured. Nothing in the sweep is close enough to a crossover for a dtype to move it
  by the 8.7 % that separates 8 from 16.
- **Behaviour beyond 8 threads.** `#198` asks for scaling evidence past evo's 8; the width shows no
  scaling signal at all (1.42–1.49× flat across widths and budgets), so a 16/32/64-thread scan of the
  same parameter would be measuring the same non-lever.

## Probe

`faer-benchmark/probes/hh179-phases` (untracked, house rule): `run_sweep.py` (the declared protocol
and the grid), `sweep.sh` / `shares.sh` / `floor.sh` (one rebuild per point), `instrumentation.patch`
(the stage timers, feature `probe-timers`, off by default). Raw rows: `out/sweep.csv`, `out/*.jsonl`,
`out/shares.txt`. Probe, patch and raw output travel in the PR that links this record and are not
adopted by the fork.
