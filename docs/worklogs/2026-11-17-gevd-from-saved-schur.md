# Eigenvectors of a saved generalized Schur pencil (tensor4all/colmat-rs#211)

## Decisions

- **The gap.** `gevd_real` / `gevd_cplx` own the whole reduction, so a caller
  that already holds a pencil's generalized Schur factors `(S, T)` and its `Q`,
  `Z` could only get the eigenvectors by redoing the QZ. The private conversions
  `qz_to_gevd_real` / `qz_to_gevd_cplx` already do that work from the saved
  factors; this change publishes them as `gevd_from_schur_real` /
  `gevd_from_schur_cplx` (with `gevd_from_schur_scratch`), the analogue of
  `eigenvectors_from_schur_real` / `_cplx` for the ordinary eigenproblem.
- **Signature.** `(s, t)` are read-only `MatRef`s, plus optional `q`, `z`,
  `u_left`, `u_right`, `par`, `stack` and `Spec<GevdFromSchurParams, T>`. The
  params type is the hidden-field pattern the other entry points use, so the
  signature keeps room for a tuning knob without a breaking change.
- **No new numerical path.** The conversion stage back-transforms *in place*: it
  reads its output buffer's columns as the columns of `Q` (resp. `Z`) and
  overwrites them block by block. The wrapper therefore copies `q` into
  `u_left` (resp. `z` into `u_right`) before the call, or fills the identity when
  the pencil's own vectors are wanted. A zeroed temporary plus a separate
  matmul would produce NaN, because the first column read back would be the
  zeroed one; only the in-place form matches the private code. Both
  `qz_to_gevd_*` stay private and unchanged, so `gevd_real` / `gevd_cplx`
  behaviour is untouched.
- **Rejected: fixing the `(alpha, beta)` reporting.** The two routines do not
  exchange the homogeneous values; a caller has `schur_eigenvalues_real` or the
  QZ's own outputs for those. Keeping the pair out of the contract avoids a
  second source of truth.
- **Rejected: prescribing `q` / `z` as identity matrices.** `None` already means
  "the saved pencil's own vectors", and the identity fill is done internally.
- **Preconditions, documented rather than checked.** The factors must be in the
  form the QZ leaves them (real: `S` upper quasi-triangular, `T` upper
  triangular; complex: both upper triangular, `T`'s diagonal real). Checking
  that costs a full pass and the private code would reject merely
  ill-conditioned blocks, not malformed ones. The complex diagonal condition is
  a real limitation inherited from LAPACK `ztgevc`: the private complex path
  scales its shift by `Re(t_jj)`, so a complex `t_jj` yields the vector of the
  pencil whose `t_jj` is replaced by its real part. `qz_cplx` standardizes the
  diagonal to real, so the intended input always satisfies it.
- **Non-finite input.** Any non-finite entry in `s`, `t`, `q` or `z` is rejected
  with `GevdError::NoConvergence` and the requested outputs are filled with NaN,
  including when a non-finite `q` / `z` is passed but its side is not requested;
  the alternative is a silent NaN column in one output and not the other.
  Requested outputs that come out non-finite also return `NoConvergence`, as
  `gevd_real` / `gevd_cplx` do.
- **`0/0` blocks and infinite eigenvalues** keep the LAPACK behaviour
  (`dtgevc.f` 3.12.0, lines 551-558 / 876-883): a block whose `S` and `T`
  diagonals are both below the safe minimum writes the unit vector `e_j`, without
  the back-transformation. Documented, not changed.

## Verification conclusions and constraints

- `tests/public_gevd_from_schur.rs` covers the issue's MWE as a test (f32, f64)
  and as a doctest, the residuals of both vector sides at
  `10 * n * eps` (real and complex, including an infinite `t_jj == 0` block and
  a nonregular `0/0` block), the back-transformation against the pencil before
  the reduction (roots via the public `generalized_hessenberg` +
  `hessenberg_to_qz`: f32/f64/c32/c64, with and without `q`/`z`, `Par::Seq` and a
  2-thread team within the family tolerance), offset/row-major strided inputs,
  NaN-prefilled output buffers (bitwise identical to zeroed ones), exact scratch
  reuse, the rejection states, and `n = 0` / `n = 1`.
- `tests/gevd_no_alloc.rs` extends the existing allocation gate: a warm
  `gevd_from_schur_*` call with `q`, `z` and both vector sides allocates nothing
  at `n = 8` and `n = 100`, real and complex.
- The real left vectors satisfy the transpose relation `y^T A = lambda y^T B`
  (pinned by `linalg::gevd::tests::left_vectors_of_schur_pencil_f64`), the
  complex ones the adjoint relation.
- **Review.** Plan gate with `reviewer-glm` (`ollama-cloud/glm-5.3-flash`,
  thinking high): round 1 `VERDICT: CHANGES` (one blocker: the conversion stage
  back-transforms in place, so the wrapper must copy `q`/`z` into the output
  buffers instead of using a zeroed temporary; three important and three minor
  findings), round 2 `VERDICT: APPROVE`; all findings addressed.
- **Known limitation, not fixed here.** The vectors are computed from the
  factors as given, and the conversion stage's coefficient pair can underflow
  when a block's `s_jj` / `t_jj` is many orders below the largest entry of the
  factors: `S = [[1, 1], [0, 2e-200]]`, `T = diag(1, 1e-200)`, right side only,
  returns `e1` for the second column, the vector of `S v = 0` rather than of
  `(S - 2T) v = 0`, with a relative residual of 0.33 and `Ok`. The cause is not
  the factors' scaling: unlike LAPACK `?tgevc`, the stage does not normalize the
  coefficient pair with a reciprocal, so the pair carries `temp^2` and
  underflows. The defect is in the shared private stage, so `gevd_real` /
  `gevd_cplx` inherit it; fixing it changes that stage for both, which this
  change deliberately does not do (the issue requires no change to the existing
  entry points). Tracked in colmat-rs#223, which also corrects the "range"
  rustdoc in `gevd_from_schur_real`.
- **Not verified against LAPACK.** No `?tgevc` reference values were compared:
  the fork has no LAPACK oracle for a pencil in generalized Schur form, and the
  conversion is the pre-existing code that `gevd_real` / `gevd_cplx` already
  exercise. The residual bounds above are the evidence; a LAPACK comparison
  remains open, as for the other `gevd` entry points.
- `clippy -p colmat --all-targets` is clean; `cargo fmt --all --check` has no
  diff outside `faer/src/linalg/evd/tridiag_evd.rs`, which is already
  unformatted on `main` and untouched here (the CI fmt job is
  `continue-on-error`).
