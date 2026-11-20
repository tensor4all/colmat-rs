//! rebuilding $A = RQ$ from a packed $RQ$ factorization.
use crate::assert;
use crate::internal_prelude::*;
pub fn reconstruct_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	linalg::lq::no_pivoting::reconstruct::reconstruct_scratch::<T>(
		nrows, ncols, block_size, par,
	)
}
#[track_caller]
pub fn reconstruct<T: ComplexField>(
	out: MatMut<'_, T>,
	packed: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	assert!(all(
		out.nrows() == packed.nrows(),
		out.ncols() == packed.ncols()
	));
	// $P_m A P_n$ is the $LQ$ product of the reversed factors.
	linalg::lq::no_pivoting::reconstruct::reconstruct(
		out.reverse_rows_and_cols_mut(),
		packed.reverse_rows_and_cols(),
		Q_coeff,
		par,
		stack,
	);
}
