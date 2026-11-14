use crate::internal_prelude::*;
use crate::linalg::cholesky::ldlt::factor::LdltParams;
/// dynamic $LL^\top$ regularization.
///
/// values below `epsilon` in absolute value, or with the wrong sign are set to
/// `delta` with their corrected sign
#[derive(Copy, Clone, Debug)]
pub struct LltRegularization<T> {
	/// regularized value
	pub dynamic_regularization_delta: T,
	/// regularization threshold
	pub dynamic_regularization_epsilon: T,
}
/// info about the result of the $LL^\top$ factorization
#[derive(Copy, Clone, Debug)]
pub struct LltInfo {
	/// number of pivots whose value or sign had to be corrected
	pub dynamic_regularization_count: usize,
}
/// error in the $LL^\top$ factorization
#[derive(Copy, Clone, Debug)]
pub enum LltError {
	NonPositivePivot { index: usize },
}
impl core::fmt::Display for LltError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Debug::fmt(self, f)
	}
}
impl core::error::Error for LltError {}
impl<T: RealField> Default for LltRegularization<T> {
	fn default() -> Self {
		Self {
			dynamic_regularization_delta: zero(),
			dynamic_regularization_epsilon: zero(),
		}
	}
}
#[derive(Copy, Clone, Debug)]
pub struct LltParams {
	pub recursion_threshold: usize,
	pub block_size: usize,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}
/// Blocking of the `f64`/`c64` LLT (tensor4all/colmat-rs#192): `LLT_BLOCK_SIZE`-wide steps of the
/// right-looking recursion and `LLT_RECURSION_THRESHOLD`-wide SIMD leaves, against the 128/64 the
/// LDLT defaults give. Each step's diagonal block is factored before its panel solve and its
/// trailing update can start, so it is the serial part of a team call: at n = 512 on 8 threads the
/// 128-wide diagonal blocks were 46 % (`f64`) and 39 % (`c64`) of the factorization and did not
/// shrink with the team. Median ms, column-major, a team of `T` on one L3 domain; evo2 (Zen 5),
/// 128/64 -> 64/32:
///
/// | dtype, n, T  | 128/64 | 64/32 |
/// |--------------|--------|-------|
/// | f64, 511, 1  | 0.863  | 0.872 |
/// | f64, 511, 8  | 0.355  | 0.291 |
/// | f64, 1024, 1 | 5.652  | 5.603 |
/// | f64, 1024, 8 | 1.301  | 1.110 |
/// | c64, 256, 1  | 0.497  | 0.466 |
/// | c64, 256, 8  | 0.286  | 0.224 |
/// | c64, 511, 1  | 2.902  | 2.897 |
/// | c64, 511, 8  | 0.891  | 0.731 |
/// | c64, 1024, 1 | 19.60  | 19.89 |
/// | c64, 1024, 8 | 3.598  | 3.295 |
///
/// A different blocking is a different summation order: the factor differs from the 128/64 one by
/// rounding, and it is the same at every team width. LDLT and `f32`/`c32` keep the LDLT defaults:
/// they were not scanned.
const LLT_BLOCK_SIZE: usize = 64;
/// See [`LLT_BLOCK_SIZE`].
const LLT_RECURSION_THRESHOLD: usize = 32;
impl<T: ComplexField> Auto<T> for LltParams {
	#[inline]
	fn auto() -> Self {
		let ldlt = <LdltParams as Auto<T>>::auto();
		let (recursion_threshold, block_size) =
			if const { T::IS_NATIVE_F64 || T::IS_NATIVE_C64 } {
				(LLT_RECURSION_THRESHOLD, LLT_BLOCK_SIZE)
			} else {
				(ldlt.recursion_threshold, ldlt.block_size)
			};
		Self {
			recursion_threshold,
			block_size,
			non_exhaustive: NonExhaustive(()),
		}
	}
}
#[inline]
pub fn cholesky_in_place_scratch<T: ComplexField>(
	dim: usize,
	par: ParShape,
	params: Spec<LltParams, T>,
) -> StackReq {
	_ = par;
	_ = params;
	temp_mat_scratch::<T>(dim, 1)
}
pub fn cholesky_in_place<T: ComplexField>(
	A: MatMut<'_, T>,
	regularization: LltRegularization<T::Real>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<LltParams, T>,
) -> Result<LltInfo, LltError> {
	let params = params.config;
	let N = A.nrows();
	let mut D = unsafe { temp_mat_uninit(N, 1, stack).0 };
	let D = D.as_mat_mut();
	match linalg::cholesky::ldlt::factor::cholesky_block_left_looking(
		A,
		D.col_mut(0).transpose_mut(),
		params.block_size,
		params.recursion_threshold,
		params.block_size,
		true,
		regularization.dynamic_regularization_delta > zero()
			&& regularization.dynamic_regularization_epsilon > zero(),
		&regularization.dynamic_regularization_epsilon,
		&regularization.dynamic_regularization_delta,
		None,
		par,
	) {
		Ok(count) => Ok(LltInfo {
			dynamic_regularization_count: count,
		}),
		Err(index) => Err(LltError::NonPositivePivot { index }),
	}
}
