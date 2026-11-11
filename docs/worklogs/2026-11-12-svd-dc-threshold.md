# `svd_thin` below n = 128: the recursion was never reached

Issue: [tensor4all/faer#404](https://github.com/tensor4all/faer/issues/404) - the remaining
faer-vs-MKL losses. This record covers the largest unowned one and the one change that carries it.

Measured revisions, both on `zen5-cpu` (host `shinaoka-EVO-X1Pro`, Ryzen AI 9 HX 470):
`before` is `4d140a9` (origin/main, 0.32.0), `after` is this branch. Both runs were
recorded by `faer-benchmark`'s `scripts/record_run.py` with the campaign declaration, one
process-set group of three, and published nowhere (`--no-publish`); the harness carries one local
commit that re-resolved `Cargo.lock` from `t4a-faer` 0.28.0 to 0.32.0 so the pair could be built
against main at all.

The candidate run built the branch at `a50d929`; edits after the recording are comments and
`#[cfg(test)]` cases only. That is checked, not asserted: `objcopy` of `.text` and `.rodata` from the
binary the recorder measured and from the binary this commit builds are the same bytes
(`d614e836c749…`), while origin/main's differ (`cab1bb640d84…`), so the numbers below are the
numbers for this tree.

## The loss, and why this one

Ranked against MKL on the revision both arms of the campaign cover (`71b25ffb`, 0.27.1, pages
`…-0539947ffb75-mkl-timing-v2.md`), by the median faer/MKL ratio of a family, with each already
owned family excluded:

| family, dtype, regime | median faer/MKL | worst cell | owner |
|---|---|---|---|
| `eigh` f64 | 1.35 | `128` 8T 2.23 | #391, #354, #180 |
| `svd_thin` f64 `linalg-small` | **1.27** | `96x96` 8T 1.56 | **none** |
| `lu` f64 tall | 1.17 | `512` 8T 2.78 | #123 |
| `solve` f64 mid | 1.13 | `512` 8T 2.88 | #123 |
| `qr_factor` f64 tall | 1.10 | `64x1024` 8T 2.18 | #178 |

`svd_thin` f64 is the only family where **both** vendors beat faer and nothing claims it (1.27
against MKL, 1.28 against OpenBLAS), and it is also the largest absolute loss of the unowned ones
(0.72 ms of a 2.01 ms call at `96x96` 8T). It is confined to `n = 64` and `n = 96`, which is what
located the cause.

`#179` (`svd_values`) is **not** the largest loss at this revision and the change cannot reach it.
Measured here before the change: f64 median 0.989 (`linalg-small`) and 0.899 (`linalg-mid`), c64
0.743 and 0.710 - faer already wins, so the issue's premise is from `0e15729`, not from main. The
values-only route returns through `dqds` before the gate below is read, and both runs put it inside
5% every-where, so this is left as the recorded follow-up rather than folded in.

## Decisions

- **The loss is route selection, not arithmetic.** `SvdParams::recursion_threshold` (default 128)
  is read twice, and it is compared against the *same* value in both roles:
  `compute_bidiag_real_svd` runs `bidiag_svd::qr_algorithm` below it and
  `bidiag_svd::divide_and_conquer` above it, and the same value arrives as that call's
  `qr_fallback_threshold`, the leaf size at which the recursion itself hands back to QR. So below
  n = 128 nothing reached the recursion at all - a 96-square thin SVD ran one QR over the whole
  bidiagonal - and at exactly n = 128 the recursion started, split, and gave both halves straight
  back to QR (`n < qr_fallback_threshold`, 128).
- **The recursion is cheaper at every size where the route differs, so the default moves and
  nothing else.** `qr_algorithm` takes no `Par` (`bidiag_svd.rs:281`), so the band below 128 was also
  the one band with no threaded route at all, which is why faer gained nothing from eight threads
  there while MKL's driver did: at `96x96` the call is 2.03 ms at 1T and 2.01 ms at 8T, while MKL goes
  1.58 to 1.28. The qualifier is not decoration: n = 32 keeps QR at either value (0.161 against
  0.162 in table C), and n = 1024 at 8T is the same 167.0 ms either way, so "every size" would have
  been a claim the probe does not make.
- **40.** The leaf value is flat between 33 and 44 and 48 is already 4-9% worse at n = 96; 40 is the
  middle of that flat region rather than an edge of it, and it is a multiple of 8. It also leaves
  n = 32 - the campaign's guard cell - on QR, where the route does not change.
- **A base case far under 128 is not this machine's quirk.** Reference LAPACK's `DBDSDC` takes
  `SMLSIZ` from `ILAENV(9, 'DBDSDC', ..)` and falls back to `DLASDQ` below it
  (`lapack/SRC/dbdsdc.f:291`, `:346`; `ilaenv.f:677-681` returns 25). faer's own LAPACK-derived SVD
  cases run at `TESTING/svd.in`'s threshold 50 and at 8, i.e. the recursion has been exercised far
  below 128 - only never through the default.
- **One stage, no API change.** The field stays public and its default moves; no signature, no
  caller, no version change. `Auto<T>` already exists to tune a hyperparameter per scalar type, and
  a per-type gate is *not* used here: the only evidence for one is the four `f32` cells below, at one
  size and one thread count, which is not enough to add a second path (see the follow-up).

## Verification conclusions and constraints

- **The pair is valid.** Both runs carry identical measured paths (`4d140a9` vs `a50d929` differ only
  in `faer/src`), the machine, the declaration, the harness build inputs and the toolkit identity;
  the in-run control row (`faer-upstream`, the unchanged crate) reproduces the baseline over all 720
  cells at median ratio 0.99997 with **0** cells outside the baseline's own between-start range.
- **The target improves and nothing near it moves.** Of 720 `faer-fork` cells, 188 are faster than
  5%, 4 are slower than 5%, and the rest are inside the 5% band the campaign sets for a shared host.
  `svd_values` (288 cells, all dtypes) and `qr_factor` (144 cells) are inside 5% **everywhere** -
  they are the guards, and the change does not reach them.
- **Four cells regress, all the same shape, and are accepted on that basis:** `svd_thin f32`, n = 64, **1T only**, 0.500 to 0.533 ms
  (+5.1% to +6.6% per cell, +5.7% as the median of its four); `c32` at the same size and thread
  count is +3.4%. The same cells at 4T and 8T gain 9-10%, and the `f32` family median over all its
  cells is 0.925. The mechanism is in the measurement: `f32`'s QR leaf is cheap enough that the merge
  is not paid back until n is above 64, so its crossover is higher than `f64`'s - which is what a
  per-scalar-type gate would exploit. They are accepted rather than fixed because the family they
  belong to is a net win (median 0.925), the loss is one size at one thread count and not a band, the
  worst one (+6.6%) is inside the +10% p95 that the campaign sets for a shared host, and the
  alternative - a value per scalar type - is unmeasured at 256 and 512, where the same field is also
  the leaf. A reviewer looking for a criterion should read it as: a size/thread-count corner inside a
  winning family is tolerated when the fix for it would change an unmeasured leaf, and is written
  down here so it is not rediscovered as a surprise.
- **No `n = 32` movement.** The guard cell reads 0.994, 0.999, 1.001, 1.005, 1.008 against its own
  baseline across dtypes, orders and thread counts.
- **Independent review: `ollama-cloud/glm-5.3-flash:high` (GLM-5.3-flash), pass with four findings,
  all fixed.** Run against the whole branch diff (`origin/main...HEAD`, both commits) in a separate
  process with read-only tools and none of this session's context, which is the gate the owner moved
  to that model. It confirmed the two roles of the field, the floor being leaf-only, that nothing
  else in the tree assumes 128, and that the new 19..39 leaf band runs leaf code already exercised at
  leaves 3..7 and 63..127 - no correctness defect at a leaf of 40. Its findings, all applied: the
  gate test ran only at `f64` although the band is new for every precision (now instantiated at all
  four, as `tests/lapack.rs` does); the values-only fallthrough was only exercised at n = 127, not at
  the shallowest recursion it now reaches (now also n = 40); "cheaper at every size" was unsupported
  at n = 32 and bounded by equality at n = 1024/8T (now scoped); and the four `f32` cells were listed
  without being accepted against a criterion (basis recorded above). It also asked for the `auto()`
  narrative and the field doc to be trimmed to the mechanism plus a pointer, which they are.
- **One attempt was discarded, not used.** The first baseline recording
  (`20261111T122715Z-acc404-base`) ran the full four-regime grid and hit the two-hour cap with two of
  its three process sets complete (44 CSVs, no `report.md`); the guard log shows one attempt and no
  rejections, so the time was measurement, not retries. It is kept in `faer-benchmark` under that
  name but none of its numbers appear here. The grid was then cut to `linalg-small` + `linalg-mid`,
  which is why tall and wide are missing below.
- **Coverage gap, stated rather than papered over.** This pair re-measured `linalg-small` and
  `linalg-mid` only; the tall and wide SVD cells (`64x1024`, `1024x64`, `256x4096`, …) were **not**
  re-measured. The reason is the same fact that makes the change safe: the route depends on
  `n = min(m, n)` alone (the bidiagonal is n-square whatever `m` is), and every tall/wide SVD cell in
  the campaign has `n` in {64, 256}, both of which are measured here in the square cells with the
  same route. What is not covered is the `m`-dependent apply around the bidiagonal, which this change
  does not touch. A decision probe (not a campaign) confirmed the direction at those shapes; that is
  weaker than a recorded cell and is called a probe, not evidence.
- **What the tests do and do not pin.** `tests/route_at_the_gate.rs` reads the default from
  `Auto` and requires the default route, the QR-forced route (`usize::MAX`) and the recursion-forced
  route (`4`) to agree on the spectrum, the reconstruction and both orthogonality residuals at the
  gate and on either side of it. It therefore does **not** fail if the value changes - both routes
  are correct - so it is not a test of *this* number. It pins what moving the number newly exposes:
  the recursion at the band immediately below the gate, and the fact that the scratch reservation
  and the route choice are the same comparison (`bidiag_real_svd_scratch` and
  `compute_bidiag_real_svd`), so a `<=` that moves in only one of them is caught at `gate - 1`,
  `gate`, `gate + 1`. The subproblems it newly exercises are the leaves of parents of 40 to 79, i.e.
  19 to 39 - a band no existing case reaches, since the LAPACK-derived cases run at threshold 8
  (leaves 3 to 7) or 128 (leaves 63 to 127).
- **The values-only fallthrough got its own case.** `svd_values` returns through `dqds` before the
  gate, so the campaign cannot see this path; but when `dqds` gives up (DCHKBD type 16, #110) the
  call falls through to the rotation path and now meets a gate of 40 instead of 128. `tests/lapack.rs
  ::dchkbd_type_16_values_only_finishes` gains `(127, <default>)` next to its `(128, 8)` and
  `(127, 128)`, which is exactly that route at the new value.

## Evidence

Table A is faer against MKL in each run, ratio = faer/MKL, so below 1 is faer ahead. `before` is
origin/main, `after` is this branch; the MKL column is the same binary in both runs, which is the
point of quoting it: the change closes a loss, it does not move the yardstick.

Table A - `svd_thin` f64 and c64 against MKL, ms:

| dtype | n | T | before | MKL | ratio | after | ratio | gain |
|---|---|---|---|---|---|---|---|---|
| f64 | 32 | 1 | 0.170 | 0.166 | 1.021 | 0.172 | 1.035 | 0.99x |
| f64 | 32 | 8 | 0.174 | 0.176 | 0.990 | 0.174 | 0.991 | 1.00x |
| f64 | 64 | 1 | 0.909 | 0.681 | 1.335 | 0.692 | 1.017 | 1.31x |
| f64 | 64 | 8 | 0.913 | 0.621 | **1.472** | 0.556 | 0.895 | 1.64x |
| f64 | 96 | 1 | 2.040 | 1.580 | 1.291 | 1.585 | 1.003 | 1.29x |
| f64 | 96 | 8 | 2.008 | 1.278 | **1.571** | 1.152 | 0.899 | 1.74x |
| f64 | 128 | 1 | 3.400 | 3.235 | 1.051 | 2.859 | 0.884 | 1.19x |
| f64 | 128 | 8 | 2.454 | 2.401 | 1.022 | 1.977 | 0.824 | 1.24x |
| f64 | 256 | 1 | 16.848 | 17.460 | 0.965 | 14.855 | 0.851 | 1.13x |
| f64 | 256 | 8 | 9.671 | 11.148 | 0.868 | 7.762 | 0.699 | 1.25x |
| f64 | 512 | 1 | 94.339 | 118.002 | 0.799 | 88.587 | 0.754 | 1.06x |
| f64 | 512 | 8 | 34.453 | 48.564 | 0.709 | 32.041 | 0.656 | 1.08x |
| c64 | 64 | 8 | 1.030 | 0.970 | **1.062** | 0.717 | 0.741 | 1.44x |
| c64 | 96 | 8 | 2.419 | 2.324 | 1.041 | 1.493 | 0.641 | 1.62x |
| c64 | 128 | 8 | 2.975 | 3.436 | 0.866 | 2.511 | 0.726 | 1.18x |
| c64 | 512 | 8 | 53.714 | 82.904 | 0.648 | 51.072 | 0.617 | 1.05x |

The family medians against MKL, `linalg-small`: `svd_thin` f64 **1.309 to 0.932** (worst cell
1.57 to 1.04), `svd_thin` c64 0.897 to 0.735; `linalg-mid`: f64 0.853 to 0.747, c64 0.716 to 0.652.
`svd_thin` f64 `linalg-small` was the worst row of the ranking and is now below parity.

Table B - the same change read against itself, `after/before` per dtype, size and thread count,
median over both orders and both the plain and `in_place` variants (24 cells per row):

| dtype | n | 1T | 4T | 8T |
|---|---|---|---|---|
| f64 | 32 | 1.001 | 1.000 | 0.999 |
| f64 | 64 | 0.772 | 0.633 | 0.628 |
| f64 | 96 | 0.773 | 0.574 | 0.572 |
| f64 | 128 | 0.848 | 0.808 | 0.803 |
| f64 | 256 | 0.882 | 0.827 | 0.802 |
| f64 | 512 | 0.940 | 0.952 | 0.931 |
| c64 | 32 | 1.001 | 1.003 | 1.001 |
| c64 | 64 | 0.849 | 0.712 | 0.708 |
| c64 | 96 | 0.817 | 0.622 | 0.610 |
| c64 | 128 | 0.907 | 0.859 | 0.844 |
| c64 | 256 | 0.942 | 0.884 | 0.855 |
| c64 | 512 | 0.973 | 0.974 | 0.954 |
| f32 | 32 | 0.997 | 0.998 | 0.996 |
| f32 | 64 | **1.057** | 0.899 | 0.905 |
| f32 | 96 | 0.808 | 0.616 | 0.611 |
| f32 | 128 | 0.991 | 0.926 | 0.925 |
| f32 | 256 | 0.917 | 0.849 | 0.821 |
| f32 | 512 | 0.944 | 0.961 | 0.942 |
| c32 | 32 | 0.999 | 0.992 | 0.999 |
| c32 | 64 | 1.034 | 0.887 | 0.884 |
| c32 | 96 | 0.834 | 0.652 | 0.648 |
| c32 | 128 | 0.997 | 0.941 | 0.932 |
| c32 | 256 | 0.955 | 0.889 | 0.863 |
| c32 | 512 | 0.971 | 0.977 | 0.957 |

Table C - the value, from a decision probe (a standalone driver over the fork at `4d140a9`, the
campaign's pinned cores and idle guard, 15 starts, min). This chose 40; it is not acceptance
evidence and the tables above are:

| n | 1T 128 | 1T 40 | 8T 128 | 8T 40 |
|---|---|---|---|---|
| 32 | 0.161 | 0.162 | 0.227 | 0.223 |
| 64 | 0.874 | 0.689 | 0.913 | 0.612 |
| 96 | 2.107 | 1.554 | 2.080 | 1.198 |
| 128 | 3.367 | 2.855 | 2.522 | 2.041 |
| 256 | 16.72 | 14.73 | 9.615 | 7.757 |
| 512 | 94.62 | 88.86 | 34.62 | 32.14 |
| 1024 | 637.7 | 632.4 | 167.0 | 167.0 |

48 is 4-9% worse than 40 at n = 96, and 32 puts the guard cell onto the recursion for ~10% at 1T,
which is why the value sits inside 33-44 instead of at its edge.

## Follow-ups, recorded not done

- **`#179` `svd_values`.** At or below parity at this revision (see above); reopen against a
  measurement, not the old premise. Not touched by this change.
- **A per-scalar-type gate.** `f32`/`c32` at n = 64 want a higher gate than `f64`/`c64`; the four
  cells above are the whole evidence for it, and `Auto<T>` is the mechanism if it is ever justified.
  Its own measurement would have to cover the leaf-size effect at 256 and 512, since the value also
  sets the leaf, not only the entry.
- **Tall and wide SVD cells** are unmeasured in this pair (see the coverage gap).
- **`lu`/`solve` at 2.8x** remain the largest losses overall and are owned by `#123`; `eigh` is
  `#391`'s.

## Reproduction

From the repository root. The evidence tarball holds both runs exactly as the recorder wrote them
(one CSV per process set, its `.guard`, and the `run.yaml` with the flags):

```sh
mkdir -p /tmp/svd-dc && cd /tmp/svd-dc
tar xzf <repo>/docs/worklogs/evidence/2026-11-12-svd-dc-threshold.tar.gz
```

Then, for any cell, the per-start median is `total_ms` of that `run<N>-<regime>-<T>t.csv`, and the
aggregation the tables use is:

```sh
python3 - <<'PY'
import csv, glob, statistics as st
RUNS = {'before': '20261111T142652Z-acc404-base', 'after': '20261111T152001Z-acc404-cand'}

def key(r):
    return (r['regime'], r['family'], r['dtype'], r['order'], r['m'], r['n'],
            r['threads'], r['note'])

def load(run):
    out = {}
    for p in sorted(glob.glob(f'{run}/run*-*.csv')):
        start = p.split('/')[-1].split('-')[0]          # one process set
        for r in csv.DictReader(open(p)):
            if r['status'] == 'ok':
                out.setdefault(start, {}).setdefault(key(r), {})[r['row']] = float(r['total_ms'])
    return out

CACHE = {run: load(run) for run in RUNS.values()}

def med(label, engine, cell):
    """median over process starts of that start's median, as the campaign reports a cell"""
    values = [s[cell][engine] for s in CACHE[RUNS[label]].values() if engine in s.get(cell, {})]
    return st.median(values)

def cell(dtype, n, t, variant='', order='f'):
    regime = 'linalg-small' if n < 128 else 'linalg-mid'
    return (regime, 'svd_thin', dtype, order, str(n), str(n), str(t), variant)

# A: faer against MKL in each run
rows = ([(d, n, t) for d in ('f64',) for n in (32, 64, 96, 128, 256, 512) for t in (1, 8)]
        + [('c64', n, 8) for n in (64, 96, 128, 512)])
for dtype, n, t in rows:
    c = cell(dtype, n, t)
    b, m, a = (med('before', 'faer-fork', c), med('before', 'lapack-mkl', c),
               med('after', 'faer-fork', c))
    print(f'A {dtype} {n} {t} {b:.3f} {m:.3f} {b / m:.3f} {a:.3f} {a / med("after", "lapack-mkl", c):.3f} {b / a:.2f}x')

# B: the same run read against itself, median over both orders and both variants
for dtype in ('f64', 'c64', 'f32', 'c32'):
    for n in (32, 64, 96, 128, 256, 512):
        cols = [st.median([med('after', 'faer-fork', cell(dtype, n, t, v, o))
                           / med('before', 'faer-fork', cell(dtype, n, t, v, o))
                           for v in ('', 'in_place') for o in ('f', 'c')]) for t in (1, 4, 8)]
        print(f'B {dtype} {n} ' + ' '.join(f'{x:.3f}' for x in cols))
PY
```

The comparison the campaign runs itself, which also checks the control row and refuses the pair if
the declaration, the machine or the measured paths differ:

```sh
scripts/candidate_compare.py --baseline data/results/zen5-cpu/faer-kernels/20261111T142652Z-acc404-base \
  --candidate data/results/zen5-cpu/faer-kernels/20261111T152001Z-acc404-cand \
  --checkout <worktree with a50d929>
```
