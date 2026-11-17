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
  kept once; the wrapper and its tests are unchanged and inherit it. Off-diagonals are deliberately
  not checked: a subnormal squared coupling cannot limit any singular value's relative accuracy
  (the pre-shift split deflates it), and rejecting it would turn exact answers into failures.

## Verification conclusions and constraints

- The regression is `dqds::tests::a_subnormal_squared_factor_is_not_a_success`: both precisions of
  the reported inputs return `NoConvergence` with the diagonal bit-identical, and the `1e-10` /
  `1e-100` controls stay `Ok` inside `20 eps` of `small / hypot(1, 0.25)`. It fails on the previous
  kernel (`Ok` where `Err` is required). `dqds::tests::the_squared_range_check_is_a_ratio_and_spares_off_diagonals`
  pins that the rejection is a ratio and not a magnitude (a matrix scaled to `1e-30` / `1e-300`
  answers like the unit-scale one) and that a subnormal squared off-diagonal stays `Ok`.
- **What the SVD route now answers.** `compute_bidiag_real_svd` keeps its documented fallback
  (#110), so a dense `svd(..., ComputeSvdVectors::No)` for the same two matrices still returns `Ok`
  and now takes the rotation path: the smaller value is `1e-22` / `1e-161` instead of
  `9.904e-23` / `9.689e-162`, i.e. 3.08 % at both precisions (f32 was 2.09 %, f64 0.13 %). That is
  inside the SVD's own validation contract for these magnitudes: the returned values pass the
  existing normwise and mode-agreement criteria, whose denominator floor (`sqrt(MIN_POSITIVE) *
  largest.max(1)` with an `EPSILON * largest` term, DCHKBD test 19,
  `faer/src/linalg/svd/tests/lapack.rs:198`) is a tolerance on agreement between output modes, not a
  relative-accuracy guarantee for the spectrum; both reported values are below that floor. Two
  things are worth knowing and are **not** changed here:
  - LAPACK's `dgesdd(jobz='N')` returns the smaller singular value exactly for both matrices (f32
    within 1 eps, f64 to the last bit), because below n = 25 it runs the rotation path and handles
    the final 2x2 with `DLAS2` before deflating a negligible diagonal.
  - faer's `qr_algorithm` tests `|diag[i]| <= eps` and deflates *before* its exact two-by-two branch
    (`faer/src/linalg/svd/bidiag_svd.rs:350-370`), so a trailing 2x2 with a negligible diagonal
    loses the factor `diag[j] / hypot(diag[j], e)` — 3.08 % for this shape, the bound the rotation
    path now contributes.
    Reordering that path is a change to a different function, takes every SVD with vectors through
    a different branch, and is left to its own change.
- **Scope of the behaviour change.** The kernel now also rejects a zero diagonal, which its contract
  already excludes and both callers already check (`faer/src/linalg/svd/mod.rs:378`, and the public
  wrapper on #206). What changes route is a conservative superset of the inaccurate inputs: any
  input with a squared diagonal factor below `min_positive`, including a power-of-two factor whose
  own answer was exact. Singleton input is unaffected: it normalizes to `±1`, never a subnormal.
- Not measured: no timing. The added work is one comparison per diagonal entry on a path that is
  already a multiply per entry, and only previously-inaccurate inputs change route.
