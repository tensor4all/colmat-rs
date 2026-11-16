# colmat#191 follow-up — the 1024 owning-LU gap

Worktree: `/home/shinaoka/work/colmat-191lu/own1024`, branch `perf/191-lu-owning-1024`, based on
`origin/main` `a5de9b36` (the merge of #210). The LU source in this worktree is byte-identical to the
measured revision `41086210` (`solvers.rs`, `linalg/mod.rs` and `linalg/lu/` diff empty against
`~/work/colmat-191lu/own`).

State inherited from #210 (`bench-results/evo-lu-mkl-20261116T120000Z-owning-after/REPORT.md`):

| cell | owning / in-place | owning / MKL |
|---|---|---|
| 256/512, f64+c64, F+C, 4T/8T | 1.00–1.07 | 0.85–1.12 |
| 1024 f64 F 8T | — | 1.06 |
| 1024 c64 F 8T | 1.31 | 1.31 |

The one-allocation change removed the owning route's second large allocation and with it the whole
`512` gap; `1024 c64` did not move (24.41 → 24.85 ms at 8T). This task attacks the 1024 sizes only.

## 1. What is already known, and what is not

Known from the repository's own records (both on the *previous* host, evo2, not this one):

- The owning route's overhead is `copy + split`, and the *split's* cost is bytes provisioned, not
  work done: `docs/worklogs/2026-11-12-lu-mkl-evo2.md` table C/E measured `owning − owning_no_split`
  = 0.822 ms (f64) / 1.668 ms (c64) at n = 512, 8T with 992 / 2016 minor faults per call, exactly
  4 KiB per fault, and `copy_upper_slices_only` + `zero_upper_slices_only` = 0.043 ms warm. Four
  rewrites of the split loop (contiguous column spans, `MatIndex`, triangular helpers, 8 hand-spawned
  threads) all landed inside the noise of the same fault count; hand-spawned threads were *slower*,
  so on that host the fault cost did not parallelise.
- The dropped look-ahead experiment (`docs/worklogs/2026-11-14-lu-trailing-product-lookahead.md`) is
  not to be re-tried: narrowing the child to half the team cost more than the overlap hid.
- `#210` established on **this** host that removing one of two simultaneous large allocations is
  worth 1.4–1.6× on the owning route at 512; the mechanism it recorded is the *behaviour* (two live
  blocks re-fault, one does not), not glibc's decision path.

Not known, and the reason this plan starts with a profile:

- whether `1024 c64` `in-place` is at parity with MKL (i.e. whether the 1.31 is entirely owning
  overhead) — the #210 report reports both ratios as 1.31, which is only consistent if it is;
- whether the residual is *bytes moved* (copy n² + split read n² + split write 2n²), *fresh-page
  provisioning* (2n² elements, 32 MiB at `1024 c64`), panel width, or trailing-update width;
- this host's allocator behaviour. THP is `madvise` here and `AnonHugePages` is 0, so a fresh
  anonymous mapping can only fault 4 KiB at a time. **Hypothesis, not a measurement** (the
  2026-11-12 evo2 record states the same rule for its own allocator story: it traced no glibc
  decision path): glibc munmaps *every* mmap'd chunk on free, and its dynamic `M_MMAP_THRESHOLD`
  cannot exceed `DEFAULT_MMAP_THRESHOLD_MAX` = 32 MiB, so a request whose chunk exceeds 32 MiB is
  mmapped-and-munmapped on **every** call, while a 16 MiB chunk would be served from the arena after
  its first call. `1024 c64` needs exactly 33,554,432 B (2·1024²·16; `padded_col_stride` adds
  nothing at 1024) and `1024 f64` / `512 c64` need 16,777,216 B — a 2× cliff at exactly the size
  this task is about, and a candidate explanation for why `1024 c64` alone did not improve.

  §2.2 has its own step for this: the hypothesis is **not used for any decision** until a
  standalone alloc/write/free experiment on this host confirms or refutes it (fault counts at
  16 MiB vs 32 MiB, plus one `MALLOC_MMAP_THRESHOLD_`/`MALLOC_TOP_PAD_` diagnostic run for pricing
  only — never a measurement repair, per `harness/docs/reliable-measurement.md:122`).
- the harness's vendor row is not the same call. `fbench/src/lapack.rs:1093-1113` times MKL's `lu`
  "owning" row as `?getrf` on a **prepared, reused** buffer with the L/U split only in the untimed
  check, while the faer rows' owning op allocates and splits inside the clock;
  `harness/docs/reliable-measurement.md:107-112` says so explicitly ("the owning faer routes … are
  kept as the call a user makes … and are not the comparison with the vendor row"). So the issue's
  `owning / MKL` figure charges colmat for materialising L and U and MKL for nothing. The report
  must give the like-for-like number (`in_place / MKL`) next to it; the owning cost is still a real
  user cost and is what this task is about.

## 2. Phase 0 — the profile (no library file is edited for it)

Host `shinaoka-EVO-X1Pro` (Zen 5, 24 logical CPUs), measured CPUs `4`/`4-7`/`4-11` (the
`4-11,16-23` L3 domain), observer pinned to `0-3,12-15`, one measurement at a time through
`~/work/colmat-191/harness/benchmarks/scripts/pinned.sh` with its idle guard.

### 2.1 `fbench` grid (the grid the task asks for)

```
FB=<binary built for the measured fork rev>, rows faer-fork,faer-upstream,lapack-mkl
for T in 1 4 8; for s in 0 1 2:
  PINNED_CSV=… pinned.sh $(cpus $T) -- $FB run --threads $T --n 768,1024,1536 \
    --dtype f64,c64 --family lu,solve --rows $ROWS --regime linalg-probe-1024 \
    --reps 5 --prime-ms 250 --rotate-rows $s --statistic median --csv …
```

`--regime linalg-probe-1024` is deliberately not a declared regime of either suite (the suites declare
`1024,1536` at 4T/8T only, and nothing at 768): this is a diagnostic grid, not a campaign page, and
the label keeps it out of the recorder.

Both orders are in every cell: `fbench` declares `order` as a case axis and runs `c` and `f` for each
shape, and the CSV's `order` column carries them (the #210 grids show both). This matters for the
attribution because a C-order input takes `copy_columns_into`'s tiled branch
(`faer/src/linalg/mod.rs:381`), so the copy's cost is order-dependent and the two orders are
reported apart, never pooled.

Run twice:

- **grid A** with the harness unmodified. Its `faer-fork` owning row is the stale composite
  (`fbench/src/backend.rs:1906 partial_lu_of`: `to_owned` + `lu_in_place` + a serial element-by-element
  split), which is not the library route (#210 §4);
- **grid B** with a local, uncommitted harness patch that makes the fork `lu`/`solve` owning row hold
  a `PartialPivLu<T>` in its state and reconstruct from `L()`/`U()`/`P()` — the shape the upstream row
  already has (`backend.rs:1376`). The patch, its sha256 and the built `fbench` sha256 are recorded in
  the evidence directory. It is not proposed as a PR (`tensor4all/colmat-benchmark` is a different
  repository).

Grid B is what "fbench owning vs in-place vs MKL" means for this issue; grid A is kept because it
shows the harness defect has not moved.

### 2.2 Probe: attribution and the provisioning floor

`~/work/colmat-191lu/probe` (`lu191probe`, links the checkout by path) extended with a
`/proc/self/stat` field-10 minor-fault delta per call, and run at `n = 768, 1024, 1536`, `f64`/`c64`,
`C`/`F`, `T = 1,4,8`:

| route | what it measures |
|---|---|
| `in_place` | `lu_in_place` into a hot preallocated buffer — the floor. **It allocates the same `vec![0usize; m]` ×2 and the same `MemBuffer::new(lu_in_place_scratch)` inside the clock as `new_imp` does**, so `lib − in_place` is not biased by hoisted per-call allocations (review finding 6; the #210 probe hoisted them and said so) |
| `lib` | `MatRef::partial_piv_lu` — today's owning route |
| `lib_stage=copy` | the owning route with the factorization and split skipped (diagnostic build) |
| `lib_stage=nosplit` | … with the split skipped |
| `alloc1_double` | one fresh `Mat::with_capacity(2n², 1)`, every element written — the provisioning floor at 32 MiB |
| `alloc1_full` | the same at 16 MiB (the `1024 f64` / `512 c64` case) |
| `alloc_write_free_sep` | the same two sizes in a standalone loop: allocate, write all, drop, 50× — confirms or refutes the §1 allocator hypothesis before anything is built on it |

The stage-selector build is a **throwaway worktree** (`/tmp/…/diag1024`, an env var read once in
`new_imp`), never committed, used only to attribute `lib − in_place`; the before/after numbers come
from the real branch. Minor faults are also read for `in_place` and `lib` in the real build (no
library edit needed).

**The in-place side needs its own attribution** (review finding 4): if `in_place / MKL` is a visible
share of the 1.31, then "panel width" and "trailing-update width" are confounded inside the kernel,
and a C5 sweep would only resolve them together. The same diagnostic worktree therefore carries two
small scans at `1024 c64` (and `8T`, plus `4T` if they disagree), each on `lu_in_place` alone:

- `recursion_threshold` ∈ {8, 16} — the leaf/panel width;
- the top-level level block size ∈ {512 (today), 256} — the trailing update's width, via the
  recursion's own `half` rule.

If neither moves the in-place cell by 5 %, §2.3 records "the gap is not panel or trailing-update
width at this size" instead of a C5 sweep.

### 2.3 What each outcome selects

| profile outcome | selected change |
|---|---|
| `lib_stage=copy` and `alloc1_double` account for ≥50 % of `lib − in_place` at 1024 c64 | provisioning is the term; the change must overlap or remove touches, not reorder bytes → **C2** first, then **C4** only if the bytes that remain are still bandwidth-bound |
| the split's cold re-read is a visible share (`nosplit → lib` grows with n while the write traffic does not) | **C1** |
| `lib − in_place` is small at 1024 c64 too, and `in_place / MKL` is the whole 1.31 | the kernel, not the owning route: the leaf/panel and trailing-update scans of §2.2 name which of the two, and only then **C5** — otherwise the report says the size has no width term |
| no candidate clears 5 % of the residual | report the structural limit with the measurement (`in_place` is the API that does not pay it), ship at most C1-C3 if they are clean wins, and say so |

### 2.4 What the profile measured (filled in before implementation)

Every number below is from `bench-results/evo-lu-mkl-20261117T-owning-1024/`; the record with the
tables, the mechanism and the decisions is `docs/worklogs/2026-11-17-lu-owning-1024.md`.

1. **The owning route's overhead is the factor buffer's first touch.** `lib − in_place` is 0.36-0.79 ms
   at 4T/8T wherever the block has fewer than 100 faults per call (`768 c64` 18.9 MB, `1024 f64`
   16 MiB) and 3.4-15.6 ms wherever it has 8193 or more (`1024 c64` = 32 MiB, `1536 f64` = 37.7 MB,
   `1536 c64` = 72 MiB). Stage decomposition at `1024 c64 F 8T`: allocate 0.93 ms (1 fault), copy
   +2.61 ms (4097 faulting pages), factor +18.56 ms (no new pages), split +2.65 ms (4096 faulting
   pages). The factorization is 75 % of the call and is not the difference.
2. **The mechanism is glibc's 32 MiB `mmap` ceiling, confirmed as the reviewer's finding 5 asked.**
   `alloc`+write+free of the same bytes: 16 MiB 0.70 ms / 614 faults, 32 MiB 12.70 ms / 8193,
   64 MiB 25.50 ms / 16385. Two live 16 MiB blocks (32 MiB total) cost the same 12.49 ms, so the
   one-allocation representation is not the cause. Neither `MALLOC_MMAP_THRESHOLD_` nor
   `MALLOC_TOP_PAD_` moves the real route (24.75 → 24.44 / 24.48 ms), and `MADV_HUGEPAGE` measured no
   effect at all (8193 faults either way — a `malloc` block is not 2 MiB-aligned).
3. **The prize is 4.4-4.8 ms at 8T**: `owning_reuse` (same split into resident buffers) is 20.38 ms
   against 24.76 at `1024 c64 F 8T` and 18.74 against 23.53 at `1536 f64 F 8T` — 80-90 % of the whole
   owning overhead. Removing it would put `1024 c64` owning at ≈0.96× MKL.
4. **No width term (§2.2's scans, review finding 4).** Leaf `recursion_threshold` 4/8/**8**/16/32 at
   `1024 c64 8T`: 19.19/19.02/19.53/21.55 ms — today's 8 is the optimum. Level block 512/**256**/128:
   18.98/18.71/19.49 at 8T and 102.66/104.19/105.78 at 1T — 1.4 % at 8T against -1.5 % at 1T, inside
   the 3 % bar. So C5 is not selected and the `1024 f64` `in_place`/MKL 1.10 is not a blocking term.
5. **The issue's yardstick was wrong**, which changes the target: on the harness row that calls the
   API (grid B), `1024 c64` owning/MKL is **1.14** and `in_place`/MKL **0.86**; the quoted 1.31 is the
   *owning/in-place* ratio of the #210 record, and that record has no 1024 MKL cell at all. Reported
   in the worklog §1 and §5 rather than silently re-based.

**Selection (§2.3).** The last row applies: the term is provisioning that the one-allocation owning
route cannot avoid, so the report states the floor with its measurement and **no candidate ships**.
C1's ceiling is ≈0.2 ms (the split's 2.65 ms is 4096 faults plus 40 MiB, and the cold re-read it
would remove is 16 MiB of that); C2's is ≤0.8 ms and needs a size-conditioned allocation path that
would regress every arena-served size; C3 and C4 fail their measured preconditions; C5 is refuted by
scan 4. This is the plan's own anticipated exit, so the reviewer is informed of the outcome in the PR
body rather than being asked to re-review a decision the plan already covers.

## 3. The bounded candidate set

One change, chosen by §2.3; each is small, keeps the one-allocation representation, the strides of
`L()`/`U()`, and the bits. **At most one of C1/C2/C4 ships in this PR** (the task asks for a bounded
set of changes, and each one is measured on the full grid before the next starts); C3 and C5 are
conditional by the rules below. Before any of them is implemented, the §2 numbers are appended to this
file and, if the selection is not one of the five, the reviewer is told before the code is written.

- **C1 — split fused into the panel loop.** Today `split_lu_in_place` re-reads the packed factor cold
  after the factorization. Instead, after each panel's trailing update the panel's own columns are
  extracted in the *panel*'s column range (the same values, in the same order), so the read comes from
  L2 instead of DRAM. **One design only** (review finding 3): the destination is threaded through
  `lu_in_place_recursion`/`lu_in_place_recursion_imp` as a parameter the recursion carries down, so
the owning and plain in-place routes stay one kernel path with one blocking (R-performance-07); there
  is no second entry point and no copy of the recursion body. Bit-identity holds because the written
  values are identical.
- **C2 — the square block's provisioning moved into the copy pass.** The strict lower triangle of the
  square block is a constant zero; writing it during the copy pass (where the team is already touching
  fresh pages) removes n²/2 writes from the post-factorization split, which then faults nothing.
  Same bytes, but the page faults stop being on the critical path after the factorization.
- **C3 — parallel width and chunking of the two fills.** `copy_tasks` splits `n` columns into
  `min(degree, n/16)` contiguous chunks. **Not a standing candidate** (review finding 7): the evo2
  record measured that the fault cost did not parallelise (8 hand-spawned threads were *slower*) and
  that four loop rewrites landed inside the noise. Attempted only if §2.2's profile shows a
  thread-placement effect on the faults.
- **C4 — non-temporal stores for the split's and the copy's destination writes** (they are not read
  again inside the call, and 2/3 of the split's writes are zeros): removes the read-for-ownership
  traffic, ~48 MiB at `1024 c64`. **Last in order, after C2** (review finding 8), and only if the
  profile says the residual term is bandwidth rather than faults: under a fault-dominated term NT
  stores change nothing. Only through the crate's existing `pulp` dispatch.
- **C5 — panel/leaf or trailing-update width at 1024 c64**, only if §2.2's two scans show one of them
  moving the in-place cell: a scan of `recursion_threshold` and of the level block size at 1024, kept
  only with a win; it changes the blocking and therefore the bits, so it needs `par_identity` and the
  pinned numerical tests, and the worklog records why the factors legitimately differ.

Not in scope: the public API (a caller-provided owning buffer is a different design and is what
`lu_in_place` already is), the two-allocation variant, allocator environment variables or `mallopt`
(a library cannot reach into its caller's allocator, and the harness's `docs/reliable-measurement.md`
forbids repairing a measurement that way), and `MADV_HUGEPAGE`/`MAP_POPULATE` (needs `libc`, breaks the
`no_std` build and is a platform policy, not a library decision).

## 4. Acceptance bars

1. **Bit-identity**: `partial_piv_lu_new_matches_the_serial_split` and
   `parallel_path_is_bit_identical_with_one_column_per_member` pass unchanged (C5 excepted: it changes
   the blocking deliberately and must then pass `par_identity`).
2. **No regression**: no cell of the §2.1 grid worse than today's by more than 3 %, which is this
   host's measured between-set disagreement for this grid (the #210 report's drift table: worst 2.5 %,
   256 c64 F 4T 1593.3 → 1568.3), and every before/after pair is taken in one session with rotated
   rows, so a 2 %-class claim is only made for a paired difference above that drift. The issue's
   "no >2 % 1T regression" is read as "no 1T regression beyond this host's noise": 2-3 % cells are
   reported as inside the noise, never dropped (review finding 2; R-performance-04 rejects a bar the
   machine cannot deliver).
3. **The target**: `1024 c64` owning/MKL at 8T improves from 1.31 to ≤ 1.20, and `1024 f64` does not
   get worse. If the profile shows the term is provisioning that the one-allocation owning route
   cannot avoid, the report states the floor with its measurement instead of claiming the target.
4. **Scaling**: report `1T → 8T` per cell before and after, next to MKL's own curve.
5. Every number from `pinned.sh` with the idle guard accepted, one measurement at a time, observer
   pinned; 3 rotated sets, medians of the sets.

## 5. Tests and commands

```
cargo fmt --all -- --check
cargo test -p colmat --lib --features std,rayon,threadteam linalg::solvers::tests
cargo test -p colmat --lib --features std,rayon,threadteam linalg::lu
cargo test -p colmat --lib --features std,rayon,threadteam solvers
cargo test -p colmat --lib --features std,rayon,threadteam
cargo clippy -p colmat --all-targets --features std,rayon,threadteam -- -D warnings
```

## 6. Edge cases, compatibility, deliberate non-changes

| case | expectation |
|---|---|
| `(0,0)`, `(0,n)`, `(m,0)` | unchanged: `total = 0`, 0-row views |
| tall/wide, padded strides | unchanged (`pad(m)`/`pad(size)`); C1–C4 keep the same writes |
| `Clone`, `Debug`, `ShapeCore` | unchanged |
| the conjugated (`!C::IS_CANONICAL`) input | unchanged serial `to_owned` |
| `FullPivLu`, `lu_in_place`'s public behaviour | unchanged |
| `no_std` build | untouched: no `libc`, no new dependency, no allocator call |

Deliberately not changed: `MIN_CHUNK_COLS`, the permutation construction, `lu_in_place`'s blocking
(except under C5), the `L()`/`U()` contract, `colmat-benchmark`.

## 7. Risks and rejected alternatives

1. **Risk: the win is this host's allocator.** It is, if the profile names provisioning. Mitigation:
   the report gives the fault counts and `alloc1_double`/`alloc1_full` next to the wall time, so a
   reader can see which part is library work and which is page provisioning, and the worklog says the
   win is allocator-conditional. A change that only reorders bytes must not be presented as a
   factorisation win.
2. **Risk: C1's fusion breaks the bit-identity or the `par` decomposition.** Mitigation: same values,
   same column order; the fused path is a `pub(crate)` entry point used only by `new_imp`, and the
   bit-identity test plus `par_identity` cover it.
3. **Risk: C4 is dtype-specific and can regress small n.** Mitigation: gated on the measured grid;
   keep only if every tested cell is neutral or better, and it is dropped otherwise.
4. **Rejected: the dropped look-ahead** (child on half the team): measured 1.4–2.2× slower.
5. **Rejected: rewrite the split's loop shape.** Four rewrites were measured at the same fault count;
   the loop is not the lever.
6. **Rejected: an arena or a caller-provided owned buffer.** The first is not the library's memory;
   the second is a public API change and is already available as `lu_in_place`.
7. **Rejected: fix the harness's owning row here.** Different repository; measured as grid B and
   reported.

## 8. Deliverables

- one PR from `perf/191-lu-owning-1024` with `## Verification`, no merge, review line in the body;
- `docs/worklogs/2026-11-17-lu-owning-1024.md` (the decisions, the profile, what won and what lost);
  this PR ships no source change (§2.4 selection), so `PATCHSET.md` is untouched;
- evidence beside the artifact: `bench-results/evo-lu-mkl-20261117T*-owning-1024/` (grids A and B,
  probe output, fault counts, the harness patch).

## 9. Review record

| reviewer model | scope | verdict |
|---|---|---|
| `ollama-cloud/glm-5.3-flash:high` (pi `glm-brainstormer` agent, read-only tools) | this plan, before any code | **CHANGES** |

Findings as received, and what this plan does with each (the reviewer's numbering):

- **1 (IMPORTANT, §2.1: the C/F axis is not in the grid command)** — accepted: §2.1 now states that
  `fbench` runs both orders for every shape, that the CSV's `order` column carries them, and that the
  two orders are reported apart because C-order takes the tiled copy branch
  (`faer/src/linalg/mod.rs:381`).
- **2 (IMPORTANT, §4 bar 2: a 2 % bar is tighter than this host's 2.5 % drift)** — accepted: bar 2 is
  rebased on the measured between-set drift (3 %), with paired same-session comparison and an explicit
  rule that 2-3 % cells are reported as inside the noise rather than dropped.
- **3 (IMPORTANT, §3/C1: a second entry point would duplicate the recursion)** — accepted: C1 is one
  design now, a destination parameter threaded through the recursion, citing R-performance-07; no
  copied recursion body.
- **4 (IMPORTANT, §2.2/§2.3: no attribution inside the in-place route)** — accepted: §2.2 adds the
  leaf-width and level-block-size scans in the diagnostic build, and §2.3 selects C5 only when one of
  them moves the cell; otherwise the report states that this size has no width term.
- **5 (MINOR, §1: the glibc mechanism is asserted as fact)** — accepted: §1 labels it a hypothesis,
  points at the 2026-11-12 record's own "hypothesis, not a measurement", and §2.2 adds the standalone
  alloc/write/free fault experiment as the step that confirms or refutes it before anything depends on
  it.
- **6 (MINOR, §2.2: the probe's `in_place` route hoists the per-call perm vectors and MemBuffer)** —
  accepted: the probe's `in_place` route now allocates exactly what `new_imp` allocates inside the
  clock.
- **7 (MINOR, §3/C3: which-thread-faults is contrary to the evidence)** — accepted: C3 is demoted from
  a candidate to a fallback attempted only if the profile shows a thread-placement effect.
- **8 (MINOR, §3/C4: NT stores are the last resort under a fault-dominated term)** — accepted: C4 is
  ordered after C2 and gated on the residual being bandwidth-bound.

Not changed, per the reviewer: §2.3's outcome table and its "below 5 % means report the structural
limit" exit, §1's separation of hypothesis from measurement, §7's rejected alternatives, and §8's
deliverable shape. The reviewer also judged the grid-B harness patch legitimate (it makes the row
measure its declared subject, keeps grid A, and is not proposed as a PR), and considered the
`1.31 → ≤ 1.20` bar defensible because bar 3 lets the report state the floor instead.
