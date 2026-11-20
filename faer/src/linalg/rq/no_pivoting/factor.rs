//! the packed $RQ$ factorization: $R$ in the upper right of the input, the
//! householder vectors of $Q$ along the rows to its left, and the block
//! coefficients apart.
//!
//! The output conventions (the factor shapes and the optional positive diagonal
//! gauge) follow MatrixFactorizations.jl `src/rq.jl` (MIT, Copyright (c) 2019
//! JuliaLinearAlgebra) and LAPACK's `?gerqf`. The factorization itself is faer's
//! own blocked $QR$ kernel, through the $LQ$ entry, applied to the reversed view;
//! no arithmetic is copied from either project.
use crate::assert;
use crate::internal_prelude::*;
/// information about the resulting $RQ$ factorization.
#[derive(Copy, Clone, Debug)]
pub struct RqInfo {
	/// estimated rank of the matrix: the $LQ$ estimate for the reversed view, an
	/// upper bound on the true rank (without pivoting, a dependent row usually
	/// leaves a rounding-level residual rather than an exact zero).
	pub rank: usize,
}
/// the recommended block size to use for an $RQ$ decomposition of a matrix with
/// the given shape.
#[inline]
pub fn recommended_block_size<T: ComplexField>(
	nrows: usize,
	ncols: usize,
) -> usize {
	linalg::lq::no_pivoting::factor::recommended_block_size::<T>(nrows, ncols)
}
/// $RQ$ factorization tuning parameters.
///
/// An $RQ$ factorization is the $LQ$ factorization of the reversed view, so these
/// are the $LQ$ (that is, $QR$) tuning parameters, applied to that view.
pub use linalg::lq::no_pivoting::factor::LqParams as RqParams;
/// applies the $LQ$ factorization of the view with reversed rows and columns in
/// place, which is this matrix's $RQ$ factorization.
///
/// `Q_coeff` holds the block coefficients, one block per `Q_coeff.nrows()` steps,
/// as in [`lq::no_pivoting::factor::lq_in_place`](crate::linalg::lq::no_pivoting::factor::lq_in_place),
/// in the order of the reversed view.
#[track_caller]
pub fn rq_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	Q_coeff: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<RqParams, T>,
) -> RqInfo {
	let info = linalg::lq::no_pivoting::factor::lq_in_place(
		A.reverse_rows_and_cols_mut(),
		Q_coeff,
		par,
		stack,
		params,
	);
	RqInfo { rank: info.rank }
}
/// computes the layout of required workspace for performing an $RQ$
/// decomposition with no pivoting
#[inline]
pub fn rq_in_place_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
	params: Spec<RqParams, T>,
) -> StackReq {
	linalg::lq::no_pivoting::factor::lq_in_place_scratch::<T>(
		nrows, ncols, block_size, par, params,
	)
}
/// copies the triangular factor $R$ out of the packed factorization in `packed`,
/// leaving the householder vectors intact.
///
/// `out` is either the compact $m \times \min(m, n)$ factor or the full
/// $m \times n$ one, whose first $n - \min(m, n)$ columns are zero. Every entry
/// of `out` is written.
///
/// with `positive`, the diagonal that ends in the last row and column is made
/// real and nonnegative by scaling its column; [`unitary`](crate::linalg::rq::no_pivoting::unitary::unitary)
/// applies the matching phase to $Q$, so that the product is unchanged.
#[track_caller]
pub fn copy_r<T: ComplexField>(
	out: MatMut<'_, T>,
	packed: MatRef<'_, T>,
	positive: bool,
) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	assert!(all(out.nrows() == m, out.ncols() == k || out.ncols() == n));
	linalg::lq::no_pivoting::factor::copy_l(
		out.reverse_rows_and_cols_mut(),
		packed.reverse_rows_and_cols(),
		positive,
	);
}
