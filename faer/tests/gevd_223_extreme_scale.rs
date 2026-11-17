//! tensor4all/colmat-rs#223: the coefficient pair of a `1x1` block has to be
//! normalized the way LAPACK `?tgevc` does it, with the reciprocal of the
//! maximum of the two scaled diagonal entries.
//!
//! `S = [[1, 1], [0, s11]]`, `T = diag(1, s11 / 2)`: both blocks have the
//! eigenvalue `2` and the exact right vector of the first one is `(1, 1)/sqrt 2`.
//! Without the reciprocal the normalization factor carries a `temp^2` and
//! underflows for a subnormal `s11`, which turns the pair into `(0, 0)` and the
//! column into the null vector of `S`.
mod common;
use common::*;

use colmat::linalg::gevd::{
	gevd_from_schur_cplx, gevd_from_schur_real, gevd_from_schur_scratch,
};

fn check_vectors<T: ComplexField>(
	u: MatRef<'_, T>,
	up: impl Fn(&T) -> c64,
	tol: f64,
) {
	// block 0 is `1 / 1` with the vector `e0`; block 1 is `s11 / (s11 / 2) = 2`
	// with the vector `(1, 1) / sqrt 2`, which is the one that underflows
	let half = std::f64::consts::FRAC_1_SQRT_2;
	let want = [(1.0, 0.0), (half, half)];
	assert_eq!(u.ncols(), want.len());
	for (j, &(w0, w1)) in want.iter().enumerate() {
		let x0 = up(&u[(0, j)]).norm();
		let x1 = up(&u[(1, j)]).norm();
		assert!(
			(x0 - w0).abs() <= tol && (x1 - w1).abs() <= tol,
			"column {j}: |x| = ({x0}, {x1}), want ({w0}, {w1})"
		);
	}
}

fn check_real<T: RealField>(
	up: impl Fn(&T) -> c64,
	s11: f64,
	tol: f64,
) {
	let s = Mat::<T>::from_fn(2, 2, |r, c| match (r, c) {
		(0, 0) | (0, 1) => one::<T>(),
		(1, 1) => from_f64::<T>(s11),
		_ => zero::<T>(),
	});
	let t = Mat::<T>::from_fn(2, 2, |r, c| match (r, c) {
		(0, 0) => one::<T>(),
		(1, 1) => from_f64::<T>(s11 * 0.5),
		_ => zero::<T>(),
	});
	let mut u = Mat::<T>::zeros(2, 2);
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<T>(2, Par::Seq.shape()));
	gevd_from_schur_real(
		s.as_ref(),
		t.as_ref(),
		None,
		None,
		None,
		Some(u.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
	check_vectors(u.as_ref(), up, tol);
}

fn check_cplx<TC: ComplexField>(
	lift: impl Fn(f64) -> TC,
	up: impl Fn(&TC) -> c64,
	s11: f64,
	tol: f64,
) {
	let s = Mat::<TC>::from_fn(2, 2, |r, c| match (r, c) {
		(0, 0) | (0, 1) => lift(1.0),
		(1, 1) => lift(s11),
		_ => zero::<TC>(),
	});
	let t = Mat::<TC>::from_fn(2, 2, |r, c| match (r, c) {
		(0, 0) => lift(1.0),
		(1, 1) => lift(s11 * 0.5),
		_ => zero::<TC>(),
	});
	let mut u = Mat::<TC>::zeros(2, 2);
	let mut mem =
		MemBuffer::new(gevd_from_schur_scratch::<TC>(2, Par::Seq.shape()));
	gevd_from_schur_cplx(
		s.as_ref(),
		t.as_ref(),
		None,
		None,
		None,
		Some(u.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		Default::default(),
	)
	.unwrap();
	check_vectors(u.as_ref(), up, tol);
}

#[test]
fn real_pencil_extreme_scale() {
	check_real::<f64>(|x: &f64| c64::new(*x, 0.0), 2e-200, 1e-6);
	check_real::<f32>(|x: &f32| c64::new(*x as f64, 0.0), 1e-24, 1e-3);
}

#[test]
fn cplx_pencil_extreme_scale() {
	check_cplx::<c64>(|x| c64::new(x, 0.0), |x: &c64| *x, 2e-200, 1e-6);
	check_cplx::<c32>(
		|x| c32::new(x as f32, 0.0),
		|x: &c32| c64::new(x.re as f64, x.im as f64),
		1e-24,
		1e-3,
	);
}
