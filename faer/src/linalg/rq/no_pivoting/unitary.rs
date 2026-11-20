//! the unitary factor $Q$ of a packed $RQ$ factorization.
//!
//! $Q$ is the $LQ$ factor of the reversed view read back through the reversal,
//! so it is the $LQ$ routine applied to reversed views.
use crate::assert;
use crate::internal_prelude::*;
/// the workspace required by [`unitary`] for an `out` of shape
/// `out_nrows x ncols`.
pub fn unitary_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	out_nrows: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	linalg::lq::no_pivoting::unitary::unitary_scratch::<T>(
		nrows, ncols, out_nrows, block_size, par,
	)
}
/// builds the unitary factor $Q$ of the packed factorization in `packed`.
///
/// `out` is $\min(m, n) \times n$ for the thin $Q$ (the last $\min(m, n)$ rows
/// of the full one) or $n \times n$ for the full $Q$. With `positive`, the rows
/// take the phase that
/// [`copy_r`](crate::linalg::rq::no_pivoting::factor::copy_r) removes from the
/// columns of $R$, so that the product $RQ$ is unchanged.
#[track_caller]
pub fn unitary<T: ComplexField>(
	out: MatMut<'_, T>,
	packed: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	positive: bool,
	par: Par,
	stack: &mut MemStack,
) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	assert!(all(out.ncols() == n, out.nrows() == k || out.nrows() == n));
	linalg::lq::no_pivoting::unitary::unitary(
		out.reverse_rows_and_cols_mut(),
		packed.reverse_rows_and_cols(),
		Q_coeff,
		positive,
		par,
		stack,
	);
}
