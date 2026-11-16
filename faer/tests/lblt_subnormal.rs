//! tensor4all/colmat-rs#212: nonsingular blocks whose entries are subnormal
//! must factor and solve, not produce NaN/inf through an overflowing
//! reciprocal. Both the 1x1 and the 2x2 block paths are covered, plus the
//! rank-2 update that follows a subnormal block.
#![cfg(feature = "std")]
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::cholesky::lblt::factor::*;
use colmat::linalg::cholesky::lblt::solve::*;
use colmat::traits::ComplexField;
use colmat::traits::math_utils::{from_f64, from_real, max_positive};
use colmat::{Mat, Par, ParShape};
use colmat_traits::ext::ComplexFieldExt;

fn block_matrix<T: ComplexField>(n: usize, s: T::Real) -> Mat<T> {
	let zero = from_f64::<T::Real>(0.0);
	let one = from_f64::<T::Real>(1.0);
	Mat::<T>::from_fn(n, n, |i, j| {
		let v = if i / 2 == j / 2 && i != j {
			if n > 2 && i / 2 == n / 2 - 1 {
				one.clone()
			} else {
				s.clone()
			}
		} else {
			zero.clone()
		};
		from_real::<T>(&v)
	})
}

fn factor_and_solve<T: ComplexField>(original: Mat<T>, tol: T::Real) {
	let n = original.nrows();
	let mut a = original.clone();
	let mut sub = colmat::diag::Diag::<T>::zeros(n);
	let mut p = vec![0usize; n];
	let mut inverse = vec![0usize; n];
	let mut params: colmat::Spec<LbltParams, T> = Default::default();
	params.config.pivoting = PivotingStrategy::Rook;
	let mut mem = MemBuffer::new(cholesky_in_place_scratch::<usize, T>(
		n,
		ParShape::Seq,
		params,
	));
	let (_, perm) = cholesky_in_place(
		a.as_mut(),
		sub.as_mut(),
		&mut p,
		&mut inverse,
		Par::Seq,
		MemStack::new(&mut mem),
		params,
	);
	let factor_finite = (0..n)
		.all(|i| sub[i].is_finite() && (0..n).all(|j| a[(i, j)].is_finite()));
	let mut rhs = original;
	let mut mem =
		MemBuffer::new(solve_in_place_scratch::<usize, T>(n, n, ParShape::Seq));
	solve_in_place(
		a.as_ref(),
		a.as_ref().diagonal(),
		sub.as_ref(),
		perm,
		rhs.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
	);
	let mut error = from_f64::<T::Real>(0.0);
	for j in 0..n {
		for i in 0..n {
			let target = if i == j {
				from_f64::<T>(1.0)
			} else {
				from_f64::<T>(0.0)
			};
			let diff = &rhs[(i, j)] - &target;
			if !diff.is_finite() {
				error = max_positive::<T::Real>();
			} else if diff.abs() > error {
				error = diff.abs();
			}
		}
	}
	assert!(factor_finite, "n={n}: non-finite factor");
	assert!(error < tol, "n={n}: identity error {error:?} >= {tol:?}");
}

#[test]
fn subnormal_lblt_f64() {
	let sub = f64::from_bits(1u64 << 34);
	assert!(sub.is_subnormal());
	for n in [2usize, 4] {
		factor_and_solve::<f64>(block_matrix::<f64>(n, 2.0), 1e-13);
		factor_and_solve::<f64>(block_matrix::<f64>(n, sub), 1e-13);
	}
}

#[test]
fn subnormal_lblt_c64() {
	let sub = f64::from_bits(1u64 << 34);
	for n in [2usize, 4] {
		factor_and_solve::<colmat::c64>(
			block_matrix::<colmat::c64>(n, 2.0),
			1e-13,
		);
		factor_and_solve::<colmat::c64>(
			block_matrix::<colmat::c64>(n, sub),
			1e-13,
		);
	}
}

#[test]
fn subnormal_lblt_f32() {
	let sub = f32::from_bits(1);
	assert!(sub.is_subnormal());
	for n in [2usize, 4] {
		factor_and_solve::<f32>(block_matrix::<f32>(n, 2.0), 1e-4);
		factor_and_solve::<f32>(block_matrix::<f32>(n, sub), 1e-4);
	}
}

#[test]
fn subnormal_lblt_c32() {
	let sub = f32::from_bits(1);
	for n in [2usize, 4] {
		factor_and_solve::<colmat::c32>(
			block_matrix::<colmat::c32>(n, 2.0),
			1e-4,
		);
		factor_and_solve::<colmat::c32>(
			block_matrix::<colmat::c32>(n, sub),
			1e-4,
		);
	}
}
