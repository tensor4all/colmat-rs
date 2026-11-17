# dqds reports a squared-factor range failure instead of an inaccurate success (#209)

## Decisions

- **The defect.** `dqds::singular_values` normalizes the bidiagonal by its largest entry and
  squares the result. A normalized diagonal entry below `sqrt(min_positive)` then squares to a
  subnormal, which keeps fewer mantissa bits than `eps`: the factor's relative accuracy is gone
  before the first sweep, and no sweep can give it back. `B = [[1e-22, 0.25], [0, 1]]` (f32) and
  `[[1e-161, 0.25], [0, 1]]` (f64) returned `Ok` with 2.09 % and 0.13 % relative error in the
  smaller singular value, and a factor that underflows to zero returns it 100 % wrong (f32
  `D = [1, 1e-25]`). `compute_bidiag_real_svd` already reports this class of limit for a block that
  fails mid-sweep — singular values that span more than their squares (#110) — and recovery there
  means the same thing: tell the caller and keep the bidiagonal.
- **The fix.** The diagonal squaring loop tests the product it is about to store and returns
  `SvdError::NoConvergence` when it is below `min_positive` — the stored factor is then subnormal,
  i.e. its relative precision is not guaranteed. The test is on the stored value itself, not on a
  square-root threshold, so it has no boundary of its own, and it is deliberately conservative: a
  factor that is a power of two squares exactly (`f32 D = [1, 2^-70]`, whose answer was exact) and
  is rejected with the rest, because the kernel does not spend a square root per entry to find out.
  `d` has not been written at that point (and `e` never is), so the documented "on error both are
  unchanged, finish on another path" contract holds without new code, and the caller recovers
  exactly as it does after a failed sweep.
- **Why not the accurate arm.** A pre-squaring scale can widen the squared range: the largest entry
  has to reach `sqrt(min_positive) / min|d_i|` to make the small factor normal, and the shift and
  the deflation multiply *two* squared factors (`cur_a[i] * cur_b[i]`), so the products stay inside
  the range only while that scale is at most `max_positive^(1/4)`. That leaves a condition number
  of about 4.0e28 at f32 and 7.8e230 at f64 — both reported cases are inside it, so accuracy is not
  impossible here; it is not taken because the maintainer's acceptance on the issue is the typed
  error ("returning `Ok` with 20x+ epsilon relative error is worse than a clean failure, because
  #206 exists so the LASQ1 adapter can fall back to the preserved reference routine"), because it
  replaces the scale of every values-only SVD call and so needs its own overflow and validation
  campaign, and because #110's record already measured that LAPACK's `DLASQ1` scaling (largest entry
  to `sqrt(eps/safmin)`) does not rescue a graded block. Off-diagonals are deliberately not checked:
  this guard is about the diagonal initialization, and a subnormal squared coupling is absorbed by
  the split below while its neighbours are in range (a whole block at the bottom of the range is a
  separate, pre-existing limit, not one this check covers).
- **Why the check is in the kernel, not in the public wrapper.** #207 (closed) validated the
  squared outputs in the public `bidiagonal_dqds` wrapper. The same defect is reachable from the
  values-only SVD and from any future caller, and the wrapper's contract already promises
  `NoConvergence` for "a squared-factor range limitation", so the kernel is where the promise can be
  kept once; the wrapper and its tests are unchanged and inherit it.

## Verification conclusions and constraints

- The public-API regression is `faer/tests/svd_squared_range.rs`: values-only `svd` of the MWE in
  both precisions must be `Ok`, nonnegative and descending, with each value inside `20 eps` of
  `hypot(1, 0.25)` and `small / hypot(1, 0.25)`; the `1e-10` / `1e-100` / `1e-30` / `1e-300`
  controls are held to the same bars, and a third test pins a negligible value in either position,
  either sign, and an exact zero diagonal. It fails at the first commit (`1e-22`, i.e. 258171 eps)
  and at `origin/main` (2.09 %), and passes here.
- The kernel regression is `dqds::tests::a_subnormal_squared_factor_is_not_a_success`: both precisions of
  the reported inputs return `NoConvergence` with the diagonal bit-identical, and the `1e-10` /
  `1e-100` controls stay `Ok` inside `20 eps` of `small / hypot(1, 0.25)`. It fails on the previous
  kernel (`Ok` where `Err` is required). `dqds::tests::the_squared_range_check_is_a_ratio_and_spares_off_diagonals`
  pins that the rejection is a ratio and not a magnitude (a matrix scaled to `1e-30` / `1e-300`
  answers like the unit-scale one) and that a subnormal squared off-diagonal stays `Ok`.
- **What the SVD route answers.** `compute_bidiag_real_svd` keeps its documented fallback
  (#110), so a dense `svd(..., ComputeSvdVectors::No)` for the same two matrices returns `Ok` on
  the rotation path instead of the kernel's error. That path used to answer *worse* than the dqds
  value it replaced, so the reorder below is part of the same change:
  - LAPACK's `dgesdd(jobz='N')` returns the smaller singular value exactly for both matrices (f32
    within 1 eps, f64 to the last bit), because below n = 25 it reaches `DLAS2` for the final block.
  - faer's `qr_algorithm` tested `|diag[i]| <= eps` and deflated *before* its exact two-by-two
    branch, so a 2x2 block with a negligible diagonal lost the factor
    `diag[j] / hypot(diag[j], e)` — 3.08 % here (`1e-22` instead of `9.7014e-23`).
- **The reorder (revision of the first commit).** `qr_algorithm` now evaluates the exact 2x2 branch
  first and runs the negligible-diagonal deflation only for blocks longer than two
  (`faer/src/linalg/svd/bidiag_svd.rs`). That is not a special case for this issue: the exact branch
  *is* the right algorithm for a 2x2 block. The deflation releases `diag[i]` unchanged and turns the
  pair into `diag[i]` and `hypot(diag[j], e)`, so the small value misses the rotation's cosine
  factor `diag[j] / hypot(diag[j], e)` — an O(1) factor (`1/1.0307764` here), not a small
  correction. The branch also updates both `U` and `V`, where the deflation rotates only `V`. Order and sign are unaffected because `qr_algorithm` ends with a
  sign-flip pass and a selection sort that swaps the matching `U`/`V` columns. Blocks longer than
  two keep the deflation untouched.
  Values-only `svd` of `[[small, 0.25], [0, 1]]`, relative error of the smaller value, against
  `small / hypot(1, 0.25)`:

  | input | `origin/main` | first commit (kernel guard only) | this change |
  | --- | --- | --- | --- |
  | f32, `small = 1e-22` | 2.0890e-2 | 3.0776e-2 | 5.9605e-8 (0.5 eps) |
  | f64, `small = 1e-161` | 1.3033e-3 | 3.0776e-2 | 0 (last bit) |
  | f32, `1e-10` control | 0 | 0 | 0 |
  | f64, `1e-100` control | 0 | 0 | 0 |

  The guard is unchanged and still fires for the kernel: the values-only SVD recovers on the
  rotation path, where the answer is now as accurate as the dqds value was at base and, for f64,
  exact.
- **The remaining limit, measured.** A block longer than two keeps the deflation, so the same factor
  is still lost once per negligible interior diagonal. `B = [[1, 0.25, 0], [0, 1e-161, 0.25], [0, 0,
  1]]` (and its f32 `1e-22` copy) through the values-only `svd`: `origin/main` returns the smaller
  value as **0** (its square underflows: the #209 class, 100 % wrong), this change returns
  `9.701425e-162` (`9.7014247e-23`) where `s / (1 + 0.25^2) = 9.411765e-162` is exact — 3.08 %,
  the same dropped cosine as the 2x2 case. The three returned values then multiply to
  `1.0308e-161` instead of `det = 1e-161`. Strictly better than base and still not relatively
  accurate: a longer block needs the bulge propagated rather than the diagonal released, which is
  LAPACK's structure too (`DBDSQR` reaches `DLAS2` only for the final 2x2). Left to its own change;
  the 2x2 boundary is what this PR fixes.
- **Scope of the behaviour change.** The kernel now also rejects a zero diagonal, which its contract
  already excludes and both callers already check (`faer/src/linalg/svd/mod.rs:378`, and the public
  wrapper on #206). What changes route is a conservative superset of the inaccurate inputs: any
  input with a squared diagonal factor below `min_positive`, including a power-of-two factor whose
  own answer was exact. Singleton input is unaffected: it normalizes to `±1`, never a subnormal.
- Not measured: no timing. The added work is one comparison per diagonal entry on a path that is
  already a multiply per entry.
