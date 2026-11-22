//! the unitary factor $Q$ of a packed $QL$ factorization.
//!
//! $Q$ is the $QR$ factor of the reversed view read back through the reversal,
//! so it is the $QR$ routine applied to reversed views.
use crate::assert;
use crate::internal_prelude::*;
/// the workspace required by [`unitary`] for an `out` of shape
/// `nrows x out_ncols`.
pub fn unitary_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	out_ncols: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	linalg::qr::no_pivoting::unitary::unitary_scratch::<T>(
		nrows, ncols, out_ncols, block_size, par,
	)
}
/// builds the unitary factor $Q$ of the packed factorization in `packed`.
///
/// `out` is $m \times \min(m, n)$ for the thin $Q$ (the last $\min(m, n)$
/// columns of the full one) or $m \times m$ for the full $Q$. With `positive`,
/// the columns take the phase that
/// [`copy_l`](crate::linalg::ql::no_pivoting::factor::copy_l) removes from the
/// rows of $L$, so that the product $QL$ is unchanged.
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
	assert!(all(out.nrows() == m, out.ncols() == k || out.ncols() == m));
	linalg::qr::no_pivoting::unitary::unitary(
		out.reverse_rows_and_cols_mut(),
		packed.reverse_rows_and_cols(),
		Q_coeff,
		positive,
		par,
		stack,
	);
}
