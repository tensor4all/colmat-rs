use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::matrix_free::{InitialGuessStatus, lsmr::*};
use colmat::traits::{ComplexField, math_utils::from_f64};
use colmat::{Mat, Par, c32, c64};
fn check<T: ComplexField>(tolerance: f64, value: fn(f64, f64) -> T) {
	let a = Mat::<T>::from_fn(4, 4, |i, j| {
		value(
			if i == j { (i + 1) as f64 } else { 0.1 },
			if i + 1 == j { 0.05 } else { 0.0 },
		)
	});
	let m = Mat::<T>::from_fn(4, 4, |i, j| {
		value(
			if i == j { 1.0 + 0.1 * i as f64 } else { 0.0 },
			if i == j { 0.02 * i as f64 } else { 0.0 },
		)
	});
	// Transposing the owned 2x4 storage gives a strided 4x2 block RHS.
	let storage = Mat::<T>::from_fn(2, 4, |j, i| {
		value(1.0 + (i + 2 * j) as f64, 0.1 * (i + j) as f64)
	});
	let b = storage.transpose();
	let reference = (m.adjoint() * (a.adjoint() * b)).norm_l2();
	for nonzero in [false, true] {
		let initial = Mat::<T>::from_fn(4, 2, |i, j| {
			if nonzero {
				value(0.05 * (i + j + 1) as f64, 0.01 * j as f64)
			} else {
				from_f64(0.0)
			}
		});
		let mut x = initial.clone();
		let residual = b - &a * &initial;
		let actual = (m.adjoint() * (a.adjoint() * &residual)).norm_l2();
		assert!(actual > from_f64::<T::Real>(0.1));
		let mut calls = 0;
		let mut mem = MemBuffer::new(lsmr_scratch(&m, &a, 2, Par::Seq.shape()));
		let info = lsmr(
			x.as_mut(),
			&m,
			&a,
			b,
			LsmrParams {
				initial_guess: if nonzero {
					InitialGuessStatus::MaybeNonZero
				} else {
					InitialGuessStatus::Zero
				},
				abs_tolerance: &actual + &reference,
				rel_tolerance: from_f64(0.0),
				max_iters: 0,
				..Default::default()
			},
			|_| {
				calls += 1;
			},
			Par::Seq,
			MemStack::new(&mut mem),
		)
		.unwrap();
		let tol = from_f64::<T::Real>(tolerance)
			* (&actual + from_f64::<T::Real>(1.0));
		assert!(
			info.abs_residual >= &actual - &tol
				&& info.abs_residual <= &actual + &tol
		);
		let relative = &actual / &reference;
		assert!(
			info.rel_residual >= &relative - &tol
				&& info.rel_residual <= &relative + &tol
		);
		assert_eq!(info.iter_count, 0);
		assert!(!info.stopped);
		assert_eq!(calls, 0);
		assert!((&x - &initial).norm_l2() <= tol);
	}
}
#[test]
fn f32_initial() {
	check::<f32>(1e-4, |r, _| r as f32);
}
#[test]
fn f64_initial() {
	check::<f64>(1e-12, |r, _| r);
}
#[test]
fn c32_initial() {
	check::<c32>(1e-4, |r, i| c32::new(r as f32, i as f32));
}
#[test]
fn c64_initial() {
	check::<c64>(1e-12, c64::new);
}
