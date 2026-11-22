//! the packed $QL$ factorization: $L$ in the lower right of the input, the
//! householder vectors of $Q$ above it, and the block coefficients apart.
//!
//! The output conventions (the factor shapes and the optional positive diagonal
//! gauge) follow MatrixFactorizations.jl `src/ql.jl` (MIT, Copyright (c) 2019
//! JuliaLinearAlgebra) and LAPACK's `?geqlf`. The factorization itself is faer's
//! own blocked $QR$ kernel applied to the reversed view; no arithmetic is copied
//! from either project.
use crate::assert;
use crate::internal_prelude::*;
/// information about the resulting $QL$ factorization.
#[derive(Copy, Clone, Debug)]
pub struct QlInfo {
	/// estimated rank of the matrix: the $QR$ kernel's estimate for the reversed
	/// view, the number of columns whose residual after the previous reflectors is
	/// not exactly zero. without pivoting, a dependent column usually leaves a
	/// rounding-level residual and is counted, so this is not rank-revealing.
	pub rank: usize,
}
/// the recommended block size to use for a $QL$ decomposition of a matrix with
/// the given shape.
#[inline]
pub fn recommended_block_size<T: ComplexField>(
	nrows: usize,
	ncols: usize,
) -> usize {
	linalg::qr::no_pivoting::factor::recommended_block_size::<T>(nrows, ncols)
}
/// $QL$ factorization tuning parameters.
///
/// A $QL$ factorization is the $QR$ factorization of the reversed view, so these
/// are the $QR$ tuning parameters, applied to that view.
pub use linalg::qr::no_pivoting::factor::QrParams as QlParams;
/// applies the $QR$ factorization of the view with reversed rows and columns in
/// place, which is this matrix's $QL$ factorization.
///
/// `Q_coeff` holds the block coefficients, one block per `Q_coeff.nrows()` steps,
/// as in [`qr::no_pivoting::factor::qr_in_place`](crate::linalg::qr::no_pivoting::factor::qr_in_place),
/// in the order of the reversed view.
#[track_caller]
pub fn ql_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	Q_coeff: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<QlParams, T>,
) -> QlInfo {
	let info = linalg::qr::no_pivoting::factor::qr_in_place(
		A.reverse_rows_and_cols_mut(),
		Q_coeff,
		par,
		stack,
		params,
	);
	QlInfo { rank: info.rank }
}
/// computes the layout of required workspace for performing a $QL$
/// decomposition with no pivoting
#[inline]
pub fn ql_in_place_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
	params: Spec<QlParams, T>,
) -> StackReq {
	linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
		nrows, ncols, block_size, par, params,
	)
}
/// copies the triangular factor $L$ out of the packed factorization in `packed`,
/// leaving the householder vectors intact.
///
/// `out` is either the compact $\min(m, n) \times n$ factor or the full
/// $m \times n$ one, whose first $m - \min(m, n)$ rows are zero. Every entry of
/// `out` is written.
///
/// with `positive`, the diagonal that ends in the last row and column is made
/// real and nonnegative by scaling its row; [`unitary`](crate::linalg::ql::no_pivoting::unitary::unitary)
/// applies the matching phase to $Q$, so that the product is unchanged.
#[track_caller]
pub fn copy_l<T: ComplexField>(
	out: MatMut<'_, T>,
	packed: MatRef<'_, T>,
	positive: bool,
) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	assert!(all(out.ncols() == n, out.nrows() == k || out.nrows() == m));
	linalg::qr::no_pivoting::factor::copy_r(
		out.reverse_rows_and_cols_mut(),
		packed.reverse_rows_and_cols(),
		positive,
	);
}
