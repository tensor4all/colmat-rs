# `svd_thin` below n = 128: the recursion was never reached

Issue: [tensor4all/faer#404](https://github.com/tensor4all/faer/issues/404) - the remaining
faer-vs-MKL losses. This record covers the largest unowned one and the one change that carries it.

Measured revisions, both on `zen5-cpu` (host `shinaoka-EVO-X1Pro`, Ryzen AI 9 HX 470):
`before` is `4d140a9` (origin/main, 0.32.0), `after` is this branch. Both runs were
recorded by `faer-benchmark`'s `scripts/record_run.py` with the campaign declaration, one
process-set group of three, and published nowhere (`--no-publish`); the harness carries one local
commit that re-resolved `Cargo.lock` from `t4a-faer` 0.28.0 to 0.32.0 so the pair could be built
against main at all.

The candidate run built the branch at `a50d929`, a state that the review round then rewrote; that
commit is **not** retained on any published ref, so nothing below can be re-derived by checking out
what was measured. One statement replaces it, and only the reproducible half of it is claimed: every
edit after the recording is a doc comment inside a `#[cfg(test)]` module - the branch is those
commits, and
`git diff 6e7e80f..HEAD -- faer/src` touches only comment lines inside `#[cfg(test)]` modules (the
DCHKBD case's doc comment, plus two `//` lines above its case list that a later round deleted; the
gate test itself arrived in the recorded perf commit), so no line compiled into the library changed
after the numbers were taken. The command prints that diff; it is the check, not an assertion.

### Not in this repository, and not evidence here

Three things the record leans on are defined outside it. They are named so nobody has to guess where
a number came from, and they are harness conventions or published pages - not measurements committed
with this PR, and not reproducible from it:

- **The campaign guard's 5% floor** - `candidate.py:221` in the separate `campaign-toolkit`
  repository. It accepts a candidate inside `[low * 0.95, high * 1.05]` of the baseline's three
  starts. It is the *definition* of the check the control row is quoted against; the reproducible
  side of that comparison (0 of 720 under the floor, 296 of 720 against the raw range) is computed
  from the committed tarball by the script below. The toolkit is not vendored in faer.
- **The "+10% p95 for a shared host" acceptance convention** - the campaign's rule for reading a
  single regressed cell on a host it does not own, used for the four `f32` cells. A convention, not a
  measurement; the reproducible half of that argument is the cell values (0.500 -> 0.533 ms), which
  are in table B.
- **The `71b25ffb` ranking medians (1.27 MKL, 1.28 OpenBLAS)** - read from the published
  `faer-benchmark` pages for that revision and their raw CSVs, which live in the harness repository,
  not in faer. They chose the target. They are not evidence for any number this PR reports, and this
  PR's own pair covers `linalg-small` and `linalg-mid` only.

An earlier round of this record also compared `objcopy` hashes of `.text` + `.rodata` between the
measured build and the head. That claim is dropped: the measured tree is on no published ref, so
neither half of the comparison can be reproduced from this repository, and a claim a reader cannot
check does not belong in the evidence.

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

This table is read from the harness's published pages, not measured here: none of it is
reproducible from this repository or from the tarball (see "Not in this repository, and not evidence
here"). It chose the target; the numbers this PR reports come from its own pair.

`svd_thin` f64 is the only family where **both** vendors beat faer and nothing claims it (1.27
against MKL, 1.28 against OpenBLAS - published harness pages, not this PR's evidence; see "Not in
this repository" above), and it is also the largest absolute loss of the unowned ones
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
  there while MKL's driver did: at `96x96` the call is 2.04 ms at 1T and 2.08 ms at 8T, while MKL goes
  1.58 to 1.28. The qualifier is not decoration: n = 32 keeps QR at either value (0.159 at 128
  against 0.160 at 40 in table C, and the recursion there is 11% worse), and n = 1024 at 8T is
  167.043 against 167.040 ms, so "every size" would have been a claim the probe does not make.
- **40.** At n = 96 the four sampled thresholds 33, 36, 40 and 44 all land within 0.19% at 1T
  (1.552-1.555 ms) and 0.75% at 8T (1.195-1.204 ms), while 48 is already 4.6% (1T) and 9.9% (8T)
  worse. 40 is interior to that band rather than at either end - 0.0% from 33 at 1T and 0.2% at 8T,
  0.2% and 0.7% from 44, with the band's midpoint near 38 - and it is a multiple of 8. It also
  leaves n = 32 - the campaign's guard cell - on QR, where the route does not change.
- **A base case far under 128 is not a choice unique to this decision.** Two supports, one
  checkable here and one not. Checkable: faer's own LAPACK-derived SVD cases already run the
  recursion at thresholds 8 and 128 (`faer/src/linalg/svd/tests/lapack.rs`, its `threshold`
  argument), i.e. the recursion has been exercised far below 128 - only never through the default,
  which is the whole defect. Not checkable here: reference LAPACK's `DBDSDC` takes `SMLSIZ` from
  `ILAENV(9, 'DBDSDC', ..)` (25 in the reference distribution) and falls back to `DLASDQ` below it.
  Those Fortran sources are not vendored in this repository; the citation is LAPACK 3.12.0 as
  distributed in OpenMathLib/OpenBLAS commit `791b02a1…`, the same copy the header of
  `tests/lapack.rs` names, so a reader can fetch exactly what was read. Nothing in the change rests
  on it: the value is bracketed by table C, which is committed.
- **One stage, no API change.** The field stays public and its default moves; no signature, no
  caller, no version change. `Auto<T>` already exists to tune a hyperparameter per scalar type, and
  a per-type gate is *not* used here: the only evidence for one is the four `f32` cells below, at one
  size and one thread count, which is not enough to add a second path (see the follow-up).

## Verification conclusions and constraints

- **The pair is valid.** Both runs carry identical measured paths (`4d140a9` vs `a50d929` differ only
  in `faer/src`), the machine, the declaration, the harness build inputs and the toolkit identity.
  The control row is `faer-upstream`, the untouched upstream crate - the *same* binary in both runs,
  which is what makes it a control: it measures how much the host moved between the two sessions,
  not how faer 0.24.4 compares to this fork. Recomputed from the evidence tarball's `faer-upstream`
  rows: 720 cells, median candidate/baseline ratio **0.999972**.
  Its "0 outside" needs the definition that produced it, because the raw range says something else.
  `candidate_compare.py` asks `low * 0.95 <= candidate <= high * 1.05` of the baseline's three
  starts (`candidate.py:221` in the separate campaign toolkit - harness-only, not vendored here),
  and under that floored band 0 of
  720 cells are outside. Against the **unfloored** min/max range, **296** of 720 are, i.e. session
  drift on this host is larger than the within-session spread of three starts. That is the reason
  the campaign's band has the floor, and it is quoted here instead of the bare "0 outside", which
  does not reproduce without it.
- **The target improves and nothing near it moves.** Of 720 `faer-fork` cells, 188 are faster than
  5%, 4 are slower, and 528 are inside the band, counting the per-cell median ratio - the statistic
  printed by the script below, and the only one quoted here. `svd_values` (288 cells, all dtypes)
  spans 0.979 to 1.028 and `qr_factor` (144 cells) 0.986
  to 1.016, so both are inside 5% **everywhere** - they are the guards, and the change does not
  reach them.
- **Four cells regress, all the same shape, and are accepted on that basis:** `svd_thin f32`, n = 64, **1T only**, 0.500 to 0.533 ms
  (+5.1% to +6.6% per cell, +5.7% as the median of its four); `c32` at the same size and thread
  count is +3.4%. The same cells at 4T and 8T gain 9-10%, and the `f32` family median over all its
  cells is 0.925. The mechanism is in the measurement: `f32`'s QR leaf is cheap enough that the merge
  is not paid back until n is above 64, so its crossover is higher than `f64`'s - which is what a
  per-scalar-type gate would exploit. They are accepted rather than fixed because the family they
  belong to is a net win (median 0.925), the loss is one size at one thread count and not a band, the
  worst one (+6.6%) is inside the campaign's +10% p95 convention for a shared host (harness-only,
  see "Not in this repository"), and the
  alternative - a value per scalar type - is unmeasured at 256 and 512, where the same field is also
  the leaf. A reviewer looking for a criterion should read it as: a size/thread-count corner inside a
  winning family is tolerated when the fix for it would change an unmeasured leaf, and is written
  down here so it is not rediscovered as a surprise.
- **No `n = 32` movement.** Over all 48 `svd_thin` cells at n = 32 (four dtypes, both orders, both
  variants, three thread counts) the candidate/baseline ratio spans **0.978 to 1.018** with median
  0.998 - f64 0.990-1.013, c64 0.995-1.010, f32 0.989-1.018, c32 0.978-1.003. Nothing there is a
  route change, so the band is the host's, not the change's.
- **Independent review: GLM-5.3-flash (`ollama-cloud/glm-5.3-flash`), seven rounds, every finding
  either fixed or shown not to hold.** Round 1 returned the whole branch diff (`origin/main...HEAD`)
  in a separate process with read-only tools and none of this session's context, which is the gate
  the owner moved to that model, and found four issues; rounds 2-7 are recorded below in order, each
  against the head it was run on. It confirmed the two roles of the field, the floor being leaf-only, that no other
  place in `faer/src` keys off `SvdParams::recursion_threshold`, and that the new 19..39 leaf band
  runs leaf code already exercised at
  leaves 3..7 and 63..127 - no correctness defect at a leaf of 40. Its findings, all applied: the
  gate test ran only at `f64` although the band is new for every precision (now instantiated at all
  four, as `tests/lapack.rs` does); the values-only fallthrough was only exercised at n = 127, not at
  the shallowest recursion it now reaches (now also n = 40); "cheaper at every size" was unsupported
  at n = 32 and bounded by equality at n = 1024/8T (now scoped); and the four `f32` cells were listed
  without being accepted against a criterion (basis recorded above). It also asked for the `auto()`
  narrative and the field doc to be trimmed to the mechanism plus a pointer, which they are.
- **Second round, the standard PR review at the first head (GLM-5.3-flash, `NEEDS_CHANGES 0`,
  `UNVERIFIED 4`), all four answered here.** (1) The measured state `a50d929` is on no published ref
  after the review rewrite, so the `.text`/`.rodata` identity is now split into the half a reader can
  rebuild (this branch's hash) and the half that cannot be re-derived, and is labelled an
  observation; the durable statement is the diff itself: every post-recording edit is a comment or a
  `#[cfg(test)]` case. (2) The control row's "0 outside" was quoted without the 5% floor the
  comparator applies (a 5% floor, defined in the separate toolkit - see "Not in this repository");
  the raw range puts 296 of 720 outside, and both numbers
  are now in the record along with the point of the row - it is the same upstream binary in both
  runs, so it measures the host, not 0.24.4 against this fork. (3) Table B's caption said 24 cells
  per row; it is 4 per entry and 12 per row. (4) The guard band and the `svd_thin f64` family median
  were quoted from an earlier statistic; recomputed from the tarball they are 0.978-1.018 (median
  0.998) over all 48 n = 32 cells and 1.306 -> 0.932 (worst 1.571 -> 1.035). Every number in the
  text is now printed by the reproduction script below, so a reader checks the script rather than
  this prose.
- **Third round (6 `UNVERIFIED`, all six closed by changing the artefact, not the wording).**
  (1) `PATCHSET.md` still carried the pre-recomputation medians and paired the worst *before* cell
  with its own after value, calling it the worst cell; the row now prints 1.306 -> 0.932,
  0.854 -> 0.746, names `96x96` 8T as the worst-before cell with its 1.571 -> 0.899, and states that
  the worst remaining cell after the change is 1.035 at `32x32` 1T. (2) The probe behind 40 was not
  in the tarball, so the value could not be re-derived; `probe/` is now committed with the driver,
  the four CSVs and each run's guard, table C is regenerated from those CSVs (the earlier table had
  been pasted from an earlier pass and disagreed with them by up to 3%), and the aggregation script
  prints table C and the sweep. (3) The body's test count did not match the command as printed; both
  are now given with the base revision on the same host (814 / 810, +4) and the reviewer's host
  (806 / 802, +4). (4) The `dbdsdc.f` / `ilaenv.f` citation is now marked as not vendored here, with
  the LAPACK copy it comes from named by the same commit `tests/lapack.rs` cites, and with the
  in-repo half (the crate's own cases at thresholds 8 and 128) given as the part a reader can check;
  nothing in the change depends on it. (5) "Code-quality is green" is replaced by what the workflow
  actually says (`continue-on-error: true`, so it cannot distinguish clean from tolerated) and the
  claim that the crate fails clippy on `main` is dropped rather than re-asserted. (6) Hosted CI is
  reported as it stood (four green, three pending) instead of as green.
- **Fourth round (4 `UNVERIFIED`, all four closed).** (1) "Nothing else in the tree assumes 128" was
  false as written: `SelfAdjointEvdParams::auto()` has the identical gate-equals-leaf pair at 128
  (`evd/mod.rs:91`, `:494`, `:507`, floor `tridiag_evd.rs:389`), now scoped to
  `SvdParams::recursion_threshold` in the claim and recorded as a follow-up for `#391`.
  (2) "40 is the middle of the flat region" overstated the committed sweep, whose midpoint is about
  38; the text now says 40 is interior to the band and gives the distances it can support (0.0% from
  33 at 1T, 0.2% at 8T; 0.2% and 0.7% from 44), and the perf
  commit's message is noted as carrying the looser wording. (3) The `objcopy` `.text`/`.rodata`
  comparison is withdrawn outright - neither half is reproducible from this repository - leaving
  `git diff 6e7e80f..HEAD -- faer/src` empty as the only claim tying the numbers to the tree.
  (4) The comparator's "189 faster / 5 slower" came from a tool that is not in the evidence; only
  the per-cell statistic the committed script prints (188 / 4 / 528) is quoted now.
- **Fifth round (2 `UNVERIFIED`, both closed).** (1) "0.8% from 33" was the width of the 40..44 gap at
  8T, not the distance from 33, whose committed value is 0.0% (1T) and 0.2% (8T); the sentence now
  gives the four sampled thresholds' spread (0.19% at 1T, 0.75% at 8T) and the distances it can
  support, and the round-4 entry that repeated the wrong figure is corrected too. (2) The campaign
  toolkit's 5% floor, the +10% p95 convention and the `71b25ffb` ranking medians are defined outside
  this repository; a "Not in this repository, and not evidence here" section now names each of them
  with what they are used for and what the reproducible half is, the ranking table carries its own
  pointer, and the script's label no longer cites a file that is not here.
- **Sixth round (1 `NEEDS_CHANGES`, 2 `UNVERIFIED`, all three closed).** (1) The `(40, <default>)`
  case was claimed as values-only fall-through coverage in all four precisions; probed directly on
  that fixture's magnitudes, `dqds` finishes at n = 40 for `f64` and `c64` and only fails for
  `f32`/`c32`, so at n = 40 the gate is reached by the single-precision values-only modes and, in
  the double-precision ones, by the `U`/`V` modes. The test's doc comment and this record now say
  which is which, and keep the genuine part: at n = 127 `dqds` gives up at every precision.
  (2) "leaves of 19..39" was called off by one and changed to 20..39 - **that change was itself the
  error** (see the seventh round: the split drops the boundary entry, so the second child is
  `n - n/2 - 1`). (3) "the idle guard accepted on the first
  attempt of every cell" was too broad: all 36 counted `run*` process sets were first-attempt
  acceptances, but one of the candidate run's 6 `verify` passes
  (`verify-linalg-mid-4t.guard`) was rejected on attempt 1 and re-run. Both documents now state it
  per process set, and the source diff that the record points at is described as comment-only rather
  than empty.
- **Seventh round, the author's own final gate (GLM-5.3-flash, `ollama-cloud/glm-5.3-flash`, full
  branch diff at `6880c28ea`, `NEEDS_CHANGES 1`, `UNVERIFIED 4`).** The one real finding: the sixth
  round's leaf "correction" was itself wrong. The split drops the boundary entry
  (`rem = n - k - 1`), so a parent of 40 gives children of 20 and 19, not 20 and 20, and 19 is
  reachable at the gate itself; the band is 19..39, as first written. Restored in all three passages
  and in the PR body, with the arithmetic spelled out so it is not "corrected" again. Its delete list
  asked for one line: the two-line inner comment above the `(n, threshold)` list, which by then also
  contradicted the corrected doc comment - deleted. Its four `UNVERIFIED` items are its own
  environment (no shell: it could not extract the tarball, run the reproduction or compute a diff),
  not defects; every one of them was run here against the same artefacts, and the tightened wording
  above removes the permissive reading it flagged.
- **The idle guard, per process set.** All 36 counted `run*` process sets (18 per run: three starts x
  two regimes x three thread counts) were accepted on the guard's **first** attempt, with no
  rejection in either run's `.guard` files. The `verify*` passes are separate from those: 6 per run,
  and one of them - `verify-linalg-mid-4t.guard` in the candidate run - was rejected on attempt 1
  ("busy after run") and accepted on attempt 2. So the protocol held for every counted cell, and the
  candidate run was re-run once on a check whose output is not part of any table.
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
  `gate`, `gate + 1`. The subproblems it newly exercises are the leaves of parents of 40 to 79. The
  split is `k = n / 2` and `rem = n - k - 1` (`bidiag_svd.rs:1227-1228`): the boundary entry
  `subdiag[k]` becomes the merge's coupling `alpha`, so the two children are of sizes `n/2` and
  `n - n/2 - 1`, and a parent of 40 gives 20 and 19. The band is therefore 19..39 - leaf 19 coming
  from the gate itself - and no existing case reaches it, since the LAPACK-derived cases run at
  threshold 8 (leaves 3..7) or 128 (leaves 63..127).
- **The values-only fallthrough got its own case, and what it covers is narrower than it looks.**
  `svd_values` returns through `dqds` before the gate, so the campaign cannot see this path; but when
  `dqds` gives up (DCHKBD type 16, #110) the call falls through to the rotation path and now meets a
  gate of 40 instead of 128. `tests/lapack.rs::dchkbd_type_16_values_only_finishes` gains
  `(127, <default>)` next to its `(128, 8)` and `(127, 128)`. At n = 127 `dqds` gives up at every
  precision, so that case is values-only coverage of the gate at the new value. The `(40, <default>)`
  case is not the same evidence: probed directly on the magnitudes of that fixture, `dqds` finishes
  at n = 40 for `f64` and `c64` and fails for `f32`/`c32`, so only the two single-precision
  values-only modes fall through to the gate there - for `f64`/`c64` the n = 40 case covers the split
  through the `U`/`V` modes instead. Written down because it is the kind of gap that reads as
  covered.

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

The family medians against MKL are the median of the 18 per-cell median ratios of a family - the
cells that carry a vendor row, since the `in_place` variants are faer-only (the same statistic as the
table, not a ratio of pooled times, which would read 1.370): `linalg-small`
`svd_thin` f64 **1.306 to 0.932**, worst cell 1.571 to 1.035; c64 0.896 to 0.736; `linalg-mid` f64
0.854 to 0.746, c64 0.715 to 0.651.
`svd_thin` f64 `linalg-small` was the worst row of the ranking and is now below parity.

Table B - the same change read against itself, `after/before` per dtype, size and thread count,
median over both orders and both the plain and `in_place` variants (4 cells per entry, 12 per row):

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

Table C - the value, from a decision probe. It is not acceptance evidence; the acceptance evidence
is the campaign pair above. The probe is committed with the runs: `probe/` inside the evidence
tarball holds the driver (`src/main.rs`), its `Cargo.toml`, the four CSVs and the idle guard's own
output for each, and `probe/README.md` says how each was run. Cells are milliseconds, the minimum
over every row committed for that (n, threads, threshold) - the later probe passes re-ran cells from
the earlier ones, so a cell can have up to three invocations. Raw values are in the CSVs.

| n | 1T 128 | 1T 40 | 8T 128 | 8T 40 |
|---|---|---|---|---|
| 32 | 0.159 | 0.160 | 0.227 | 0.223 |
| 64 | 0.874 | 0.679 | 0.913 | 0.611 |
| 96 | 2.042 | 1.552 | 2.080 | 1.195 |
| 128 | 3.357 | 2.850 | 2.522 | 2.041 |
| 256 | 16.72 | 14.73 | 9.61 | 7.76 |
| 512 | 93.88 | 87.88 | 34.62 | 32.14 |
| 1024 | 637.7 | 632.4 | 167.043 | 167.040 |

At n = 96, 128 -> 40 is 0.76x at 1T and 0.57x at 8T, and the sweep around the chosen value reads
(threshold: ms, n = 96): 33: 1.552, 36: 1.554, 40: 1.552, 44: 1.555, 48: 1.624, 56: 1.777,
128: 2.042 at 1T and 33: 1.198, 36: 1.197, 40: 1.195, 44: 1.204, 48: 1.313, 56: 1.433, 128: 2.080
at 8T - the four sampled thresholds 33-44 span 0.19% at 1T and 0.75% at 8T, then 48 is +4.6% (1T)
and +9.9% (8T) and 56 is +14.5% and +19.9%. At n = 32 the recursion costs 11% at 1T (0.177 at 32 against 0.159 at 33-128), which is why the
value stays above the guard cell; at n = 1024 8T the two are equal to 0.002%. The value sits inside the flat
region rather than at its edge.

The perf commit's message quotes an earlier probe pass (0.689, 0.612, 2.107, 1.554, 1.198, ...).
Those differ from the committed CSVs by up to 3% because the pass ran fewer invocations, and the
message also calls 40 "the middle of the flat region" when the committed sweep puts that middle at
38. This table and its reproduction block below are the committed data and supersede both: what the
data supports is that 40 is interior to the flat band, not that it is its centre.

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
- **The same gate-equals-leaf shape in `evd`, found by review and not touched here.**
  `SelfAdjointEvdParams::auto()` sets `recursion_threshold: 128` (`evd/mod.rs:91`), reads it as the
  QR entry gate (`evd/mod.rs:494`, `n < params.recursion_threshold` -> `tridiag_evd::qr_algorithm`)
  and passes the same value as the recursion's leaf (`evd/mod.rs:507`, floored at 4 in
  `evd/tridiag_evd.rs:389`). So below 128 nothing reaches `tridiag_evd`'s divide-and-conquer either,
  and `qr_algorithm` takes no `Par` (`tridiag_evd.rs:386` onward), i.e. the same defect with the same
  consequence for the threaded route. Unmeasured here and out of scope: `eigh` is `#391`'s lane. It
  is written down because the defect is mechanical and will otherwise be rediscovered.

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

# B: the same run read against itself, 4 cells per entry (2 orders x 2 variants), 12 per row
for dtype in ('f64', 'c64', 'f32', 'c32'):
    for n in (32, 64, 96, 128, 256, 512):
        cols = [st.median([med('after', 'faer-fork', cell(dtype, n, t, v, o))
                           / med('before', 'faer-fork', cell(dtype, n, t, v, o))
                           for v in ('', 'in_place') for o in ('f', 'c')]) for t in (1, 4, 8)]
        print(f'B {dtype} {n} ' + ' '.join(f'{x:.3f}' for x in cols))

def per_cell(label, engine, c):
    base = 'before' if label == 'after' else 'after'
    return med(label, engine, c) / med(base, engine, c)

# FAM: family medians against MKL and their worst cell, before -> after. Only the
# 18 cells that carry a vendor row count: the `in_place` variants are faer-only, so
# they cannot enter a ratio against MKL.
def fam_ratios(label, regime, dtype):
    return [med(label, 'faer-fork', c) / med(label, 'lapack-mkl', c)
            for c in CACHE[RUNS[label]]['run0']
            if c[0] == regime and c[1] == 'svd_thin' and c[2] == dtype
            and 'lapack-mkl' in CACHE[RUNS[label]]['run0'][c]]
for regime, dtype in (('linalg-small', 'f64'), ('linalg-small', 'c64'),
                      ('linalg-mid', 'f64'), ('linalg-mid', 'c64')):
    b, a = fam_ratios('before', regime, dtype), fam_ratios('after', regime, dtype)
    print(f'FAM {regime} {dtype} n={len(b)} median {st.median(b):.3f} -> {st.median(a):.3f} '
          f'worst {max(b):.3f} -> {max(a):.3f}')

# GUARD: n = 32, all dtypes, orders, variants and thread counts
g = [per_cell('after', 'faer-fork', c) for c in CACHE[RUNS['before']]['run0']
     if c[1] == 'svd_thin' and c[4] == '32']
print(f'GUARD n=32 {len(g)} cells range {min(g):.3f}..{max(g):.3f} median {st.median(g):.3f}')

# COUNTS: the 5% band, and the guards' span
for fam in ('svd_thin', 'svd_values', 'qr_factor'):
    r = [per_cell('after', 'faer-fork', c) for c in CACHE[RUNS['before']]['run0'] if c[1] == fam]
    print(f'COUNTS {fam} n={len(r)} faster {sum(1 for x in r if x < 0.95)} '
          f'slower {sum(1 for x in r if x > 1.05)} inside {sum(1 for x in r if 0.95 <= x <= 1.05)} '
          f'range {min(r):.3f}..{max(r):.3f}')

# CONTROL: the untouched upstream crate in both runs, and the band the campaign checks against
ratios, raw_outside = [], 0
for c in CACHE[RUNS['before']]['run0']:
    base = [s[c]['faer-upstream'] for s in CACHE[RUNS['before']].values() if 'faer-upstream' in s.get(c, {})]
    cand = [s[c]['faer-upstream'] for s in CACHE[RUNS['after']].values() if 'faer-upstream' in s.get(c, {})]
    if len(base) == 3 and len(cand) == 3:
        ratios.append(st.median(cand) / st.median(base))
        if not (min(base) <= st.median(cand) <= max(base)):
            raw_outside += 1
print(f'CONTROL n={len(ratios)} median {st.median(ratios):.6f} '
      f'outside unfloored range {raw_outside} (the harness guard floors the band at 5%)')

# PROBE: table C, from the committed probe CSVs (min over every row of a cell)
import csv as _csv, glob as _glob
cell = {}
for path in sorted(_glob.glob('probe/probe*.csv')):
    for r in _csv.DictReader(open(path)):
        k = (int(r['n']), int(r['threads']), int(r['threshold']))
        cell[k] = min(cell.get(k, float('inf')), float(r['ms']))
print('PROBE table C (ms, min over committed rows)')
for n in (32, 64, 96, 128, 256, 512, 1024):
    row = [cell.get((n, t, thr)) for t in (1, 8) for thr in (128, 40)]
    print('PROBE n={:<5}'.format(n) + ' '.join(f'{v:.3f}' for v in row))
print('PROBE sweep n=96 1T', ' '.join(f'{thr}:{cell[(96,1,thr)]:.3f}'
                                     for thr in (33, 36, 40, 44, 48, 56, 128)))
print('PROBE sweep n=96 8T', ' '.join(f'{thr}:{cell[(96,8,thr)]:.3f}'
                                     for thr in (33, 36, 40, 44, 48, 56, 128)))
print('PROBE n=32 1T thr32 vs thr40', f'{cell[(32,1,32)]:.3f} vs {cell[(32,1,40)]:.3f}')
PY
```

The comparison the campaign runs itself, which also checks the control row and refuses the pair if
the declaration, the machine or the measured paths differ:

```sh
scripts/candidate_compare.py --baseline data/results/zen5-cpu/faer-kernels/20261111T142652Z-acc404-base \
  --candidate data/results/zen5-cpu/faer-kernels/20261111T152001Z-acc404-cand \
  --checkout <worktree with a50d929>
```

## The gates, as run

```sh
cargo test -p t4a-faer --lib --release
```

- This host (x86_64): **814 passed, 0 failed, 162 ignored**. The base revision `4d140a9` on the same
  host: **810 passed, 0 failed, 162 ignored**, so the +4 is exactly the four `route_at_the_gate`
  instantiations. The absolute count is not portable - a reviewer running the same command on
  aarch64 reported **806** against a base of **802** - because some tests are target-gated; the
  delta is what this change is responsible for.
- `cargo build -p t4a-faer --no-default-features --features std`, and pure `--no-default-features`:
  both build.
- `clippy` is **not** a gate for this crate: `.github/workflows/code-quality.yml` runs it with
  `continue-on-error: true`, so a green `code-quality` run cannot distinguish "clean" from
  "tolerated errors". Nothing in this change was validated by clippy, and no claim is made about
  its state before or after.
- Hosted CI: at the time of writing `body`, `msrv-1.89.0` (ubuntu, windows) and `nostd-nightly`
  (ubuntu, windows) had passed; `build`, `code-quality` and `testing-stable-ubuntu-latest` were
  still **pending**. No claim of a green CI run is made here - the acceptance evidence is the
  campaign pair and the local gates above.

