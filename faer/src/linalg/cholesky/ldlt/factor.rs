use crate::assert;
use crate::internal_prelude::*;
use crate::linalg::matmul::internal::*;
use linalg::matmul::triangular::BlockStructure;
use pulp::Simd;
#[allow(clippy::too_many_arguments)] // faer kernel shape, kept flat to match upstream
#[allow(clippy::neg_cmp_op_on_partial_ord)]
// a negated ordered comparison is the unordered-aware test the code wants; swapping it changes NaN behaviour
#[inline(always)]
fn simd_cholesky_row_batch<'N, T: ComplexField, S: Simd>(
	simd: T::SimdCtx<S>,
	A: MatMut<'_, T, Dim<'N>, Dim<'N>, ContiguousFwd>,
	D: RowMut<'_, T, Dim<'N>>,
	start: IdxInc<'N>,
	is_llt: bool,
	regularize: bool,
	eps: T::Real,
	delta: T::Real,
	signs: Option<&Array<'N, i8>>,
) -> Result<usize, usize> {
	let mut A = A;
	let mut D = D;
	let n = A.ncols();
	with_dim!(TAIL, *n - *start);
	let simd = SimdCtx::<T, S>::new_force_mask(simd, TAIL);
	let (idx_head, indices, idx_tail) = simd.indices();
	assert!(idx_head.is_none());
	let Some(idx_tail) = idx_tail else { panic!() };
	let mut count = 0usize;
	for j in n.indices() {
		with_dim!(LEFT, *j);
		let (A_0, Aj) = A.rb_mut().split_at_col_mut(j.into());
		let A_0 = A_0.as_col_shape(LEFT);
		let A10 = A_0.subrows(start, TAIL);
		let mut Aj = Aj.col_mut(0).subrows_mut(start, TAIL);
		{
			let D = D.rb().subcols(IdxInc::ZERO, LEFT);
			let mut Aj = Aj.rb_mut();
			let mut iter = indices.clone();
			let i0 = iter.next();
			let i1 = iter.next();
			let i2 = iter.next();
			match (i0, i1, i2) {
				(None, None, None) => {
					let mut Aij = simd.read(Aj.rb(), idx_tail);
					for k in LEFT.indices() {
						let Ak = A10.col(k);
						let D = D[k].real();
						let D = if is_llt { one() } else { D };
						let Ajk = simd.splat(A_0[(j, k)].conj().mul_real(-D));
						let Aik = simd.read(Ak, idx_tail);
						Aij = simd.mul_add(Ajk, Aik, Aij);
					}
					simd.write(Aj.rb_mut(), idx_tail, Aij);
				},
				(Some(i0), None, None) => {
					let mut A0j = simd.read(Aj.rb(), i0);
					let mut Aij = simd.read(Aj.rb(), idx_tail);
					for k in LEFT.indices() {
						let Ak = A10.col(k);
						let D = D[k].real();
						let D = if is_llt { one() } else { D };
						let Ajk = simd.splat(A_0[(j, k)].conj().mul_real(-D));
						let A0k = simd.read(Ak, i0);
						let Aik = simd.read(Ak, idx_tail);
						A0j = simd.mul_add(Ajk, A0k, A0j);
						Aij = simd.mul_add(Ajk, Aik, Aij);
					}
					simd.write(Aj.rb_mut(), i0, A0j);
					simd.write(Aj.rb_mut(), idx_tail, Aij);
				},
				(Some(i0), Some(i1), None) => {
					let mut A0j = simd.read(Aj.rb(), i0);
					let mut A1j = simd.read(Aj.rb(), i1);
					let mut Aij = simd.read(Aj.rb(), idx_tail);
					for k in LEFT.indices() {
						let Ak = A10.col(k);
						let D = D[k].real();
						let D = if is_llt { one() } else { D };
						let Ajk = simd.splat(A_0[(j, k)].conj().mul_real(-D));
						let A0k = simd.read(Ak, i0);
						let A1k = simd.read(Ak, i1);
						let Aik = simd.read(Ak, idx_tail);
						A0j = simd.mul_add(Ajk, A0k, A0j);
						A1j = simd.mul_add(Ajk, A1k, A1j);
						Aij = simd.mul_add(Ajk, Aik, Aij);
					}
					simd.write(Aj.rb_mut(), i0, A0j);
					simd.write(Aj.rb_mut(), i1, A1j);
					simd.write(Aj.rb_mut(), idx_tail, Aij);
				},
				(Some(i0), Some(i1), Some(i2)) => {
					let mut A0j = simd.read(Aj.rb(), i0);
					let mut A1j = simd.read(Aj.rb(), i1);
					let mut A2j = simd.read(Aj.rb(), i2);
					let mut Aij = simd.read(Aj.rb(), idx_tail);
					for k in LEFT.indices() {
						let Ak = A10.col(k);
						let D = D[k].real();
						let D = if is_llt { one() } else { D };
						let Ajk = simd.splat(A_0[(j, k)].conj().mul_real(-D));
						let A0k = simd.read(Ak, i0);
						let A1k = simd.read(Ak, i1);
						let A2k = simd.read(Ak, i2);
						let Aik = simd.read(Ak, idx_tail);
						A0j = simd.mul_add(Ajk, A0k, A0j);
						A1j = simd.mul_add(Ajk, A1k, A1j);
						A2j = simd.mul_add(Ajk, A2k, A2j);
						Aij = simd.mul_add(Ajk, Aik, Aij);
					}
					simd.write(Aj.rb_mut(), i0, A0j);
					simd.write(Aj.rb_mut(), i1, A1j);
					simd.write(Aj.rb_mut(), i2, A2j);
					simd.write(Aj.rb_mut(), idx_tail, Aij);
				},
				_ => {
					unreachable!();
				},
			}
		}
		let D = D.rb_mut().at_mut(j);
		if *j >= *start {
			let j_row = TAIL.idx(*j - *start);
			let mut diag = Aj[j_row].real();
			if regularize {
				let sign = if is_llt {
					1
				} else {
					if let Some(signs) = signs { signs[j] } else { 0 }
				};
				let small_or_negative = diag <= eps;
				let minus_small_or_positive = diag >= -&eps;
				if sign == 1 && small_or_negative {
					diag = delta.copy();
					count += 1;
				} else if sign == -1i8 && minus_small_or_positive {
					diag = -&delta;
				} else {
					if small_or_negative && minus_small_or_positive {
						if diag < zero() {
							diag = -&delta;
						} else {
							diag = delta.copy();
						}
					}
				}
			}
			let diag = if is_llt {
				if !(diag > zero()) {
					*D = diag.to_cplx();
					return Err(*j);
				}
				diag.sqrt()
			} else {
				diag.copy()
			};
			*D = diag.to_cplx();
			if diag == zero() || !diag.is_finite() {
				return Err(*j);
			}
		}
		let diag = D.real();
		{
			let mut Aj = Aj.rb_mut();
			let inv = simd.splat_real(diag.recip());
			for i in indices.clone() {
				let mut Aij = simd.read(Aj.rb(), i);
				Aij = simd.mul_real(Aij, inv);
				simd.write(Aj.rb_mut(), i, Aij);
			}
			{
				let mut Aij = simd.read(Aj.rb(), idx_tail);
				Aij = simd.mul_real(Aij, inv);
				simd.write(Aj.rb_mut(), idx_tail, Aij);
			}
		}
	}
	Ok(count)
}
#[allow(clippy::too_many_arguments)] // faer kernel shape, kept flat to match upstream
#[inline(always)]
fn simd_cholesky_matrix<T: ComplexField, S: Simd>(
	simd: T::SimdCtx<S>,
	A: MatMut<'_, T, usize, usize, ContiguousFwd>,
	D: RowMut<'_, T, usize>,
	is_llt: bool,
	regularize: bool,
	eps: T::Real,
	delta: T::Real,
	signs: Option<&[i8]>,
) -> Result<usize, usize> {
	let N = A.ncols();
	let block_size =
		4 * (core::mem::size_of::<T::SimdVec<S>>() / core::mem::size_of::<T>());
	let mut A = A;
	let mut D = D;
	let mut count = 0;
	let mut j = 0;
	while j < N {
		let block_size = Ord::min(block_size, N - j);
		let j_next = j + block_size;
		with_dim!(HEAD, j_next);
		let A = A.rb_mut().submatrix_mut(0, 0, HEAD, HEAD);
		let D = D.rb_mut().subcols_mut(0, HEAD);
		let signs = signs.map(|signs| Array::from_ref(&signs[..*HEAD], HEAD));
		count += simd_cholesky_row_batch(
			simd,
			A,
			D,
			HEAD.idx_inc(j),
			is_llt,
			regularize,
			eps.clone(),
			delta.clone(),
			signs,
		)?;
		j += block_size;
	}
	Ok(count)
}
fn simd_cholesky<T: ComplexField>(
	A: MatMut<'_, T>,
	D: RowMut<'_, T>,
	is_llt: bool,
	regularize: bool,
	eps: T::Real,
	delta: T::Real,
	signs: Option<&[i8]>,
) -> Result<usize, usize> {
	struct Impl<'a, T: ComplexField> {
		A: MatMut<'a, T, usize, usize, ContiguousFwd>,
		D: RowMut<'a, T>,
		is_llt: bool,
		regularize: bool,
		eps: T::Real,
		delta: T::Real,
		signs: Option<&'a [i8]>,
	}
	impl<'a, T: ComplexField> pulp::WithSimd for Impl<'a, T> {
		type Output = Result<usize, usize>;

		#[inline(always)]
		fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
			let Self {
				A,
				D,
				is_llt,
				regularize,
				eps,
				delta,
				signs,
			} = self;
			let simd = T::simd_ctx(simd);
			if A.nrows() > 0 {
				simd_cholesky_matrix(
					simd, A, D, is_llt, regularize, eps, delta, signs,
				)
			} else {
				Ok(0)
			}
		}
	}
	let mut A = A;
	let mut D = D;
	if const { T::SIMD_CAPABILITIES.is_simd() } {
		if let Some(A) = A.rb_mut().try_as_col_major_mut() {
			dispatch!(
				Impl {
					A,
					D,
					is_llt,
					regularize,
					eps,
					delta,
					signs,
				},
				Impl,
				T
			)
		} else {
			// A leaf that is not column-major contiguous (a row-major input) runs the SIMD kernel on
			// a column-major copy of its lower triangle instead of the scalar fallback (#192).
			#[cfg(feature = "std")]
			if leaf_stage::applies(A.rb()) {
				if let Some(r) = leaf_stage::with_staged(A.rb_mut(), |S| {
					simd_cholesky(
						S,
						D.rb_mut(),
						is_llt,
						regularize,
						eps.clone(),
						delta.clone(),
						signs,
					)
				}) {
					return r;
				}
			}
			cholesky_fallback(
				A,
				D,
				is_llt,
				regularize,
				eps.clone(),
				delta.clone(),
				signs,
			)
		}
	} else {
		cholesky_fallback(
			A,
			D,
			is_llt,
			regularize,
			eps.clone(),
			delta.clone(),
			signs,
		)
	}
}
/// The diagonal leaf of a matrix that is not column-major contiguous -- a row-major input, whose
/// leaves are row-major blocks -- used to run the scalar [`cholesky_fallback`], a dependent chain
/// of generic operations per entry: at n = 512, `f64`, the row-major factorization spent 0.71 of
/// 2.08 ms (1T) and 0.71 of 1.17 ms (8T) in its eight 64-wide leaves against 0.11 ms for the same
/// leaves column-major, and the leaves are the serial part of a team call (evo2, Zen 5). The leaf is
/// now copied into a column-major buffer (lower triangle only: the kernels neither read nor write
/// the strict upper one), factored there by the SIMD leaf that column-major input uses, and copied
/// back. So a row-major leaf is computed exactly as the column-major leaf of the same values; its
/// bits differ from the scalar fallback's by rounding (a different summation order). The rest of a
/// row-major factorization (its solves and products) keeps its own layout's bits. The buffer
/// belongs to the thread, grows once to at most `MAX_DIM x MAX_DIM` and is reused, so a
/// steady-state call allocates nothing (#111).
#[cfg(feature = "std")]
mod leaf_stage {
	use crate::internal_prelude::*;
	use core::cell::RefCell;
	std::thread_local! {
		static BUF: RefCell<alloc::vec::Vec<u64>> = const { RefCell::new(alloc::vec::Vec::new()) };
	}
	/// the widest leaf staged: the default recursion threshold of the LLT/LDLT recursion; a caller
	/// that sets a wider one keeps the fallback above it
	pub(super) const MAX_DIM: usize = 64;
	/// staged nodes, for the tests that assert the path ran
	#[cfg(test)]
	pub(super) static STAGED: core::sync::atomic::AtomicUsize =
		core::sync::atomic::AtomicUsize::new(0);
	/// a native scalar the buffer of `u64` words can hold, a leaf of `1..=MAX_DIM` rows
	#[inline]
	pub(super) fn applies<T: ComplexField>(A: MatRef<'_, T>) -> bool {
		(const {
			T::IS_NATIVE_F32
				|| T::IS_NATIVE_F64
				|| T::IS_NATIVE_C32
				|| T::IS_NATIVE_C64
		}) && core::mem::align_of::<T>() <= core::mem::align_of::<u64>()
			&& A.nrows() >= 1
			&& A.nrows() <= MAX_DIM
	}
	/// Runs `f` on a column-major copy of the lower triangle of `A` and copies the lower triangle
	/// back. The copy's column stride is `n`, or `n` plus 64 bytes when `n` elements are a multiple
	/// of 1 KiB (the aliasing stride, as in `triangular_solve::stage`). Returns `None` without
	/// running `f` when this thread's buffer is already in use.
	pub(super) fn with_staged<T: ComplexField, R>(
		A: MatMut<'_, T>,
		f: impl FnOnce(MatMut<'_, T>) -> R,
	) -> Option<R> {
		let mut A = A;
		let n = A.nrows();
		let size = core::mem::size_of::<T>();
		let ld = if (n * size).is_multiple_of(1024) {
			n + 64usize.div_ceil(size)
		} else {
			n
		};
		let words = (ld * n * size).div_ceil(8);
		BUF.with(|buf| {
			let Ok(mut buf) = buf.try_borrow_mut() else {
				return None;
			};
			if buf.len() < words {
				buf.resize(words, 0);
			}
			// SAFETY: `T` is a native scalar (plain data, alignment at most 8, checked by `applies`),
			// the buffer holds `ld * n` of them and is borrowed exclusively until the end of this
			// closure; the view is `n x n` with column stride `ld >= n`.
			let mut S = unsafe {
				MatMut::from_raw_parts_mut(
					buf.as_mut_ptr() as *mut T,
					n,
					n,
					1isize,
					ld as isize,
				)
			};
			for j in 0..n {
				for i in j..n {
					S[(i, j)] = A[(i, j)].copy();
				}
			}
			#[cfg(test)]
			STAGED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
			let r = f(S.rb_mut());
			for j in 0..n {
				for i in j..n {
					A[(i, j)] = S[(i, j)].copy();
				}
			}
			Some(r)
		})
	}
}
#[allow(clippy::neg_cmp_op_on_partial_ord)] // a negated ordered comparison is the unordered-aware test the code wants; swapping it changes NaN behaviour
fn cholesky_fallback<T: ComplexField>(
	A: MatMut<'_, T>,
	D: RowMut<'_, T>,
	is_llt: bool,
	regularize: bool,
	eps: T::Real,
	delta: T::Real,
	signs: Option<&[i8]>,
) -> Result<usize, usize> {
	let n = A.nrows();
	let mut count = 0;
	let mut A = A;
	let mut D = D;
	for j in 0..n {
		for i in j..n {
			let mut sum = zero::<T>();
			for k in 0..j {
				let D = D[k].real();
				let D = if is_llt { one() } else { D };
				sum += A[(j, k)].conj() * A[(i, k)].mul_real(D);
			}
			A[(i, j)] = &A[(i, j)] - sum;
		}
		let D = D.rb_mut().at_mut(j);
		let mut diag = A[(j, j)].real();
		if regularize {
			let sign = if is_llt {
				1
			} else {
				if let Some(signs) = signs { signs[j] } else { 0 }
			};
			let small_or_negative = diag <= eps;
			let minus_small_or_positive = diag >= -&eps;
			if sign == 1 && small_or_negative {
				diag = delta.copy();
				count += 1;
			} else if sign == -1i8 && minus_small_or_positive {
				diag = -&delta;
			} else {
				if small_or_negative && minus_small_or_positive {
					if diag < zero() {
						diag = -&delta;
					} else {
						diag = delta.copy();
					}
				}
			}
		}
		let diag = if is_llt {
			if !(diag > zero()) {
				*D = diag.to_cplx();
				return Err(j);
			}
			diag.sqrt()
		} else {
			diag.copy()
		};
		*D = diag.to_cplx();
		if diag == zero() || !diag.is_finite() {
			return Err(j);
		}
		let inv = &diag.recip();
		for i in j..n {
			A[(i, j)] = A[(i, j)].mul_real(inv);
		}
	}
	Ok(count)
}
#[allow(clippy::too_many_arguments)] // faer kernel shape, kept flat to match upstream
pub(crate) fn cholesky_recursion_right_looking<T: ComplexField>(
	A: MatMut<'_, T>,
	D: RowMut<'_, T>,
	recursion_threshold: usize,
	block_size: usize,
	is_llt: bool,
	regularize: bool,
	eps: &T::Real,
	delta: &T::Real,
	signs: Option<&[i8]>,
	mut par: Par,
) -> Result<usize, usize> {
	let n = A.ncols();
	if n <= recursion_threshold {
		simd_cholesky(
			A,
			D,
			is_llt,
			regularize,
			eps.clone(),
			delta.clone(),
			signs,
		)
	} else {
		let mut count = 0;
		let block_size = Ord::min(n.next_power_of_two() / 2, block_size);
		let mut A = A;
		let mut D = D;
		let mut j = 0;
		while j < n {
			let block_size = Ord::min(block_size, n - j);
			let (mut A00, A01, mut A10, mut A11) = A
				.rb_mut()
				.get_mut(j.., j..)
				.split_at_mut(block_size, block_size);
			let mut D0 = D.rb_mut().subcols_mut(j, block_size);
			let mut L10xD0 = A01.transpose_mut();
			let signs = signs.map(|signs| &signs[j..][..block_size]);
			match cholesky_recursion_right_looking(
				A00.rb_mut(),
				D0.rb_mut(),
				recursion_threshold,
				block_size,
				is_llt,
				regularize,
				eps,
				delta,
				signs,
				par.reborrow(),
			) {
				Ok(local_count) => count += local_count,
				Err(fail_idx) => return Err(j + fail_idx),
			}
			let A00 = A00.rb();
			if is_llt {
				linalg::triangular_solve::solve_lower_triangular_in_place(
					A00.conjugate(),
					A10.rb_mut().transpose_mut(),
					par.reborrow(),
				)
			} else {
				linalg::triangular_solve::solve_unit_lower_triangular_in_place(
					A00.conjugate(),
					A10.rb_mut().transpose_mut(),
					par.reborrow(),
				)
			}
			let mut A10 = A10.rb_mut();
			if is_llt {
				linalg::matmul::triangular::matmul(
					A11.rb_mut(),
					BlockStructure::TriangularLower,
					Accum::Add,
					A10.rb(),
					BlockStructure::Rectangular,
					A10.rb().adjoint(),
					BlockStructure::Rectangular,
					-one::<T>(),
					par.reborrow(),
				);
			} else {
				if has_spicy_matmul::<T>() {
					for k in 0..block_size {
						let d = &D0[k].real().recip();
						for i in j + block_size..n {
							let i = i - (j + block_size);
							A10[(i, k)] = A10[(i, k)].mul_real(d);
						}
					}
					spicy_matmul::<usize, T>(
						A11.rb_mut(),
						BlockStructure::TriangularLower,
						None,
						None,
						Accum::Add,
						A10.rb(),
						Conj::No,
						A10.rb().transpose(),
						Conj::Yes,
						Some(D0.rb().transpose().as_diagonal()),
						-one::<T>(),
						par.reborrow(),
						MemStack::new(&mut []),
					);
				} else {
					for k in 0..block_size {
						let d = &D0[k].real().recip();
						for i in j + block_size..n {
							let i = i - (j + block_size);
							let a = A10[(i, k)].copy();
							A10[(i, k)] = A10[(i, k)].mul_real(d);
							L10xD0[(i, k)] = a;
						}
					}
					linalg::matmul::triangular::matmul(
						A11.rb_mut(),
						BlockStructure::TriangularLower,
						Accum::Add,
						A10,
						BlockStructure::Rectangular,
						L10xD0.adjoint(),
						BlockStructure::Rectangular,
						-one::<T>(),
						par.reborrow(),
					);
				}
			};
			j += block_size;
		}
		Ok(count)
	}
}
#[allow(clippy::too_many_arguments)] // faer kernel shape, kept flat to match upstream
pub(crate) fn cholesky_block_left_looking<T: ComplexField>(
	A: MatMut<'_, T>,
	D: RowMut<'_, T>,
	right_looking_threshold: usize,
	recursion_threshold: usize,
	block_size: usize,
	is_llt: bool,
	regularize: bool,
	eps: &T::Real,
	delta: &T::Real,
	signs: Option<&[i8]>,
	mut par: Par,
) -> Result<usize, usize> {
	let n = A.nrows();
	let cholesky =
		|A: MatMut<'_, T>, D: RowMut<'_, T>, signs: Option<&[i8]>, par: Par| {
			cholesky_recursion_right_looking(
				A,
				D,
				recursion_threshold,
				block_size,
				is_llt,
				regularize,
				eps,
				delta,
				signs,
				par,
			)
		};
	// The left-looking driver below is switched off on purpose, not by an oversight: it arrived
	// with this file's blocked driver in sarah-quinones/faer-rs#253 (`009d916`), already written
	// as `true || ..`, and upstream has shipped it that way since. Measured on this fork's M5 Max
	// (f64, 2048, median of seven calls of each, interleaved; `true ||` against
	// `n < right_looking_threshold`) the left-looking path is never the faster one: LLT 8T
	// 14.4 -> 17.1 ms and 1T 67.2 -> 71.1 ms, LDLT 8T 24.3 -> 26.3 ms, LDLT 1T unchanged within
	// noise; both reconstruct to the same relative residual (1.6e-15). Nothing in the suite covers
	// it either: the only test that reaches a factorization of `n >= block_size` compares a team
	// against `Par::Seq` through this same driver, so no test would notice a broken left-looking
	// branch. Keep the right-looking recursion until someone has both a reason and numbers.
	// The dead branch is the point, so the constant condition is allowed here rather than removed.
	#[allow(clippy::overly_complex_bool_expr)]
	// both operands are kept as written: the constant records an upstream construction that is under review
	// both operands are kept as written: the constant records an upstream construction that is under review
	if true || n < right_looking_threshold {
		cholesky(A, D, signs, par)
	} else {
		let mut A = A;
		let mut D = D;
		let mut count = 0;
		let mut j = 0;
		while j < n {
			let bj = Ord::min(n - j, right_looking_threshold);
			let (_, A01, AL, mut AR) = A.rb_mut().split_at_mut(j, j);
			let AL = AL.rb();
			let mut AL0xD0 = A01.get_mut(.., ..bj).transpose_mut();
			let (AL0, AL1) = AL.split_at_row(bj);
			let (mut AR0, mut AR1) =
				AR.rb_mut().get_mut(.., ..bj).split_at_row_mut(bj);
			let (D0, D1) = D.rb_mut().split_at_col_mut(j);
			let D0 = D0.rb();
			let mut D1 = D1.get_mut(..bj);
			if is_llt {
				linalg::matmul::triangular::matmul(
					AR0.rb_mut(),
					BlockStructure::TriangularLower,
					Accum::Add,
					AL0.rb(),
					BlockStructure::Rectangular,
					AL0.rb().adjoint(),
					BlockStructure::Rectangular,
					-one::<T>(),
					par.reborrow(),
				);
				linalg::matmul::matmul(
					AR1.rb_mut(),
					Accum::Add,
					AL1.rb(),
					AL0.rb().adjoint(),
					-one::<T>(),
					par.reborrow(),
				);
			} else {
				if has_spicy_matmul::<T>() {
					spicy_matmul::<usize, T>(
						AR0.rb_mut(),
						BlockStructure::TriangularLower,
						None,
						None,
						Accum::Add,
						AL0.rb(),
						Conj::No,
						AL0.rb().transpose(),
						Conj::Yes,
						Some(D0.rb().transpose().as_diagonal()),
						-one::<T>(),
						par.reborrow(),
						MemStack::new(&mut []),
					);
					spicy_matmul::<usize, T>(
						AR1.rb_mut(),
						BlockStructure::Rectangular,
						None,
						None,
						Accum::Add,
						AL1.rb(),
						Conj::No,
						AL0.rb().transpose(),
						Conj::Yes,
						Some(D0.rb().transpose().as_diagonal()),
						-one::<T>(),
						par.reborrow(),
						MemStack::new(&mut []),
					);
				} else {
					for i in 0..j {
						let d = &D0[i].real();
						for k in 0..bj {
							AL0xD0[(k, i)] = AL0[(k, i)].mul_real(d);
						}
					}
					let AL0xD0 = AL0xD0.rb();
					linalg::matmul::triangular::matmul(
						AR0.rb_mut(),
						BlockStructure::TriangularLower,
						Accum::Add,
						AL0,
						BlockStructure::Rectangular,
						AL0xD0.adjoint(),
						BlockStructure::Rectangular,
						-one::<T>(),
						par.reborrow(),
					);
					linalg::matmul::matmul(
						AR1.rb_mut(),
						Accum::Add,
						AL1,
						AL0xD0.adjoint(),
						-one::<T>(),
						par.reborrow(),
					);
				}
			};
			match cholesky(
				AR0.rb_mut(),
				D1.rb_mut(),
				signs.map(|signs| &signs[j..][..bj]),
				par.reborrow(),
			) {
				Ok(local_count) => count += local_count,
				Err(fail_idx) => return Err(j + fail_idx),
			}
			if is_llt {
				linalg::triangular_solve::solve_lower_triangular_in_place(
					AR0.conjugate(),
					AR1.rb_mut().transpose_mut(),
					par.reborrow(),
				);
			} else {
				linalg::triangular_solve::solve_unit_lower_triangular_in_place(
					AR0.conjugate(),
					AR1.rb_mut().transpose_mut(),
					par.reborrow(),
				);
				for k in 0..bj {
					let d = &D1[k].real().recip();
					for i in j + bj..n {
						let i = i - (j + bj);
						AR1[(i, k)] = AR1[(i, k)].mul_real(d);
					}
				}
			}
			j += bj;
		}
		Ok(count)
	}
}
/// dynamic $LDL^\top$ regularization.
/// values below `epsilon` in absolute value, or with the wrong sign are set to
/// `delta` with their corrected sign.
#[derive(Copy, Clone, Debug)]
pub struct LdltRegularization<'a, T> {
	/// expected signs for the diagonal at each step of the decomposition.
	pub dynamic_regularization_signs: Option<&'a [i8]>,
	/// regularized value.
	pub dynamic_regularization_delta: T,
	/// regularization threshold.
	pub dynamic_regularization_epsilon: T,
}
/// info about the result of the $LDL^\top$ factorization.
#[derive(Copy, Clone, Debug)]
pub struct LdltInfo {
	/// number of pivots whose value or sign had to be corrected.
	pub dynamic_regularization_count: usize,
}
/// error in the $LDL^\top$ factorization.
#[derive(Copy, Clone, Debug)]
pub enum LdltError {
	ZeroPivot { index: usize },
}
impl core::fmt::Display for LdltError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}
impl core::error::Error for LdltError {}
impl<T: RealField> Default for LdltRegularization<'_, T> {
	fn default() -> Self {
		Self {
			dynamic_regularization_signs: None,
			dynamic_regularization_delta: zero(),
			dynamic_regularization_epsilon: zero(),
		}
	}
}
#[derive(Copy, Clone, Debug)]
pub struct LdltParams {
	pub recursion_threshold: usize,
	pub block_size: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
impl<T: ComplexField> Auto<T> for LdltParams {
	#[inline]
	fn auto() -> Self {
		Self {
			recursion_threshold: 64,
			block_size: 128,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
#[inline]
pub fn cholesky_in_place_scratch<T: ComplexField>(
	dim: usize,
	par: ParShape,
	params: Spec<LdltParams, T>,
) -> StackReq {
	_ = par;
	_ = params;
	temp_mat_scratch::<T>(dim, 1)
}
pub fn cholesky_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	regularization: LdltRegularization<'_, T::Real>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<LdltParams, T>,
) -> Result<LdltInfo, LdltError> {
	let params = params.config;
	let n = A.nrows();
	let mut D = unsafe { temp_mat_uninit(n, 1, stack).0 };
	let D = D.as_mat_mut();
	let mut D = D.col_mut(0).transpose_mut();
	let mut A = A;
	let ret = match cholesky_block_left_looking(
		A.rb_mut(),
		D.rb_mut(),
		params.block_size,
		params.recursion_threshold,
		params.block_size,
		false,
		regularization.dynamic_regularization_delta > zero()
			&& regularization.dynamic_regularization_epsilon > zero(),
		&regularization.dynamic_regularization_epsilon,
		&regularization.dynamic_regularization_delta,
		regularization.dynamic_regularization_signs,
		par,
	) {
		Ok(count) => Ok(LdltInfo {
			dynamic_regularization_count: count,
		}),
		Err(index) => Err(LdltError::ZeroPivot { index }),
	};
	let init = if let Err(LdltError::ZeroPivot { index }) = ret {
		index + 1
	} else {
		n
	};
	for i in 0..init {
		A[(i, i)] = D[i].copy();
	}
	ret
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, Row, assert, c64};
	#[test]
	fn test_simd_cholesky() {
		let rng = &mut StdRng::seed_from_u64(0);
		type T = c64;
		for n in 0..=64 {
			for f in [cholesky_fallback::<T>, simd_cholesky::<T>] {
				for llt in [true, false] {
					let approx_eq = CwiseMat(ApproxEq {
						abs_tol: 1e-12,
						rel_tol: 1e-12,
					});
					let A = CwiseMatDistribution {
						nrows: n,
						ncols: n,
						dist: ComplexDistribution::new(
							StandardNormal,
							StandardNormal,
						),
					}
					.rand::<Mat<c64>>(rng);
					let A = &A * A.adjoint();
					let A = A.as_ref().as_shape(n, n);
					let mut L = A.cloned();
					let mut L = L.as_mut();
					let mut D = Row::zeros(n);
					let mut D = D.as_mut();
					f(L.rb_mut(), D.rb_mut(), llt, false, 0.0, 0.0, None)
						.unwrap();
					for j in 0..n {
						for i in 0..j {
							L[(i, j)] = c64::ZERO;
						}
					}
					let L = L.rb().as_dyn_stride();
					if llt {
						assert!(L * L.adjoint() ~ A);
					} else {
						assert!(L * D.as_diagonal() * L.adjoint() ~ A);
					};
				}
			}
		}
	}
	#[test]
	fn test_cholesky() {
		let rng = &mut StdRng::seed_from_u64(0);
		for n in [2, 4, 8, 31, 127, 240] {
			for llt in [false, true] {
				let approx_eq = CwiseMat(ApproxEq {
					abs_tol: 1e-12,
					rel_tol: 1e-12,
				});
				let A = CwiseMatDistribution {
					nrows: n,
					ncols: n,
					dist: ComplexDistribution::new(
						StandardNormal,
						StandardNormal,
					),
				}
				.rand::<Mat<c64>>(rng);
				let A = &A * A.adjoint();
				let A = A.as_ref();
				let mut L = A.cloned();
				let mut L = L.as_mut();
				let mut D = Row::zeros(n);
				let mut D = D.as_mut();
				cholesky_recursion_right_looking(
					L.rb_mut(),
					D.rb_mut(),
					32,
					32,
					llt,
					false,
					&0.0,
					&0.0,
					None,
					Par::Seq,
				)
				.unwrap();
				for j in 0..n {
					for i in 0..j {
						L[(i, j)] = c64::ZERO;
					}
				}
				let L = L.rb().as_dyn_stride();
				if llt {
					assert!(L * L.adjoint() ~ A);
				} else {
					assert!(L * D.as_diagonal() * L.adjoint() ~ A);
				};
			}
		}
	}
}

#[cfg(test)]
mod llt_layout_tests {
	use super::*;
	use crate::linalg::cholesky::llt::factor::{
		LltParams, cholesky_in_place, cholesky_in_place_scratch,
	};
	use crate::{Mat, assert, c32, c64};
	use dyn_stack::MemBuffer;

	/// A Hermitian positive definite `B B^H + n I`, from a fixed seed.
	fn hpd<T: ComplexField>(n: usize, from: impl Fn(f64, f64) -> T) -> Mat<T> {
		let mut seed = 0x9e3779b97f4a7c15u64 ^ n as u64;
		let mut next = || {
			seed = seed
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
		};
		let B = Mat::<T>::from_fn(n, n, |_, _| from(next(), next()));
		let mut A = Mat::<T>::zeros(n, n);
		crate::linalg::matmul::matmul(
			A.as_mut(),
			Accum::Replace,
			B.as_ref(),
			B.adjoint(),
			one::<T>(),
			Par::Seq,
		);
		for i in 0..n {
			A[(i, i)] = A[(i, i)].copy() + from(n as f64, 0.0);
		}
		A
	}

	/// The lower triangle of `A` factored by the default LLT, column-major or row-major, on `par`,
	/// as raw bits in column order.
	fn llt_bits<T: ComplexField>(
		A: &Mat<T>,
		row_major: bool,
		par: Par,
		bits: &impl Fn(&T) -> (u64, u64),
	) -> alloc::vec::Vec<(u64, u64)> {
		let n = A.nrows();
		let mut storage = if row_major {
			A.transpose().to_owned()
		} else {
			A.clone()
		};
		let L = if row_major {
			storage.as_mut().transpose_mut()
		} else {
			storage.as_mut()
		};
		let mut L = L;
		let mut mem = MemBuffer::new(cholesky_in_place_scratch::<T>(
			n,
			par.shape(),
			default(),
		));
		cholesky_in_place(
			L.rb_mut(),
			default(),
			par,
			MemStack::new(&mut mem),
			default(),
		)
		.unwrap();
		(0..n)
			.flat_map(|j| (j..n).map(move |i| (i, j)))
			.map(|(i, j)| bits(&L[(i, j)]))
			.collect()
	}

	/// The `f64`/`c64` LLT blocks are 64 wide with 32-wide leaves (#192); `f32`/`c32` keep the
	/// LDLT defaults.
	#[test]
	fn llt_default_blocking() {
		let f64p = <LltParams as Auto<f64>>::auto();
		let c64p = <LltParams as Auto<c64>>::auto();
		let f32p = <LltParams as Auto<f32>>::auto();
		let c32p = <LltParams as Auto<c32>>::auto();
		let ldlt = <LdltParams as Auto<f32>>::auto();
		assert!((f64p.recursion_threshold, f64p.block_size) == (32, 64));
		assert!((c64p.recursion_threshold, c64p.block_size) == (32, 64));
		assert!(
			(f32p.recursion_threshold, f32p.block_size)
				== (ldlt.recursion_threshold, ldlt.block_size)
		);
		assert!(
			(c32p.recursion_threshold, c32p.block_size)
				== (ldlt.recursion_threshold, ldlt.block_size)
		);
	}

	/// The default LLT gives the same bits on every team width, column-major and row-major, at
	/// sizes around the 32-wide leaf and the 64-wide block.
	#[test]
	fn llt_team_matches_sequential_bits() {
		fn run<T: ComplexField>(
			from: impl Fn(f64, f64) -> T + Copy,
			bits: impl Fn(&T) -> (u64, u64) + Copy,
		) {
			for n in [31usize, 32, 33, 63, 64, 65, 100, 257] {
				let A = hpd::<T>(n, from);
				for row_major in [false, true] {
					let want = llt_bits(&A, row_major, Par::Seq, &bits);
					for width in [2usize, 3, 8] {
						crate::team::with_team(width, |par| {
							let got = llt_bits(&A, row_major, par, &bits);
							assert!(
								got == want,
								"n={n} row_major={row_major} width={width}"
							);
						});
					}
				}
			}
		}
		run::<f64>(|re, _| re, |x| (x.to_bits(), 0));
		run::<c64>(c64::new, |x| (x.re.to_bits(), x.im.to_bits()));
	}

	/// A row-major leaf runs the column-major SIMD leaf on a staged copy (#192): a matrix that is a
	/// single leaf (at most the 32-wide default) factors to the column-major bits exactly. Larger
	/// matrices are not compared: the recursion's triangular solves and products on row-major
	/// operands are not the column-major ones bit for bit (a one-column right-hand side takes a
	/// different product path per layout), so only the leaves are layout independent.
	#[test]
	fn row_major_llt_runs_the_simd_leaf() {
		let before =
			leaf_stage::STAGED.load(core::sync::atomic::Ordering::Relaxed);
		for n in [1usize, 2, 17, 31, 32] {
			let A = hpd::<f64>(n, |re, _| re);
			let bits = |x: &f64| (x.to_bits(), 0);
			assert!(
				llt_bits(&A, true, Par::Seq, &bits)
					== llt_bits(&A, false, Par::Seq, &bits),
				"f64 n={n}"
			);
		}
		for n in [1usize, 2, 17, 31, 32] {
			let A = hpd::<c64>(n, c64::new);
			let bits = |x: &c64| (x.re.to_bits(), x.im.to_bits());
			assert!(
				llt_bits(&A, true, Par::Seq, &bits)
					== llt_bits(&A, false, Par::Seq, &bits),
				"c64 n={n}"
			);
		}
		assert!(
			leaf_stage::STAGED.load(core::sync::atomic::Ordering::Relaxed)
				> before
		);
	}
}
