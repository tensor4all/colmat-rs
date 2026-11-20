//! rebuilding $A = QL$ from a packed $QL$ factorization.
use crate::assert;
use crate::internal_prelude::*;
pub fn reconstruct_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	linalg::qr::no_pivoting::reconstruct::reconstruct_scratch::<T>(
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
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	assert!(all(out.nrows() == m, out.ncols() == n));
	// $P_m A P_n$ is the $QR$ product of the reversed basis and the reversed $L$.
	let reversed = packed.reverse_rows_and_cols();
	linalg::qr::no_pivoting::reconstruct::reconstruct(
		out.reverse_rows_and_cols_mut(),
		reversed.get(.., ..k),
		Q_coeff,
		reversed.get(..k, ..),
		par,
		stack,
	);
}
