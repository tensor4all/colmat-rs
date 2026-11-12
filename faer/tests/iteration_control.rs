use core::ops::ControlFlow;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::matrix_free::{
	IdentityPrecond, InitialGuessStatus, bicgstab as bi,
	conjugate_gradient as cg, lsmr as ls,
};
use colmat::traits::{ComplexField, math_utils::from_f64};
use colmat::{Mat, Par, c32, c64};

fn check<T: ComplexField>(tol: f64, value: fn(f64, f64) -> T) {
	let a = Mat::<T>::from_fn(4, 4, |i, j| {
		value(
			if i == j {
				(i + 1) as f64
			} else if i.abs_diff(j) == 1 {
				0.1
			} else {
				0.0
			},
			if i + 1 == j {
				0.15
			} else if j + 1 == i {
				-0.15
			} else {
				0.0
			},
		)
	});
	let b = Mat::<T>::from_fn(4, 1, |i, _| value(1.0, 0.2 * (i + 1) as f64));
	let identity = IdentityPrecond { dim: 4 };
	for solver in 0..3 {
		let mut x = Mat::<T>::zeros(4, 1);
		let mut calls = 0;
		let mut observed = Mat::<T>::zeros(4, 1);
		let callback = |v: colmat::MatRef<'_, T>| {
			calls += 1;
			observed.copy_from(v);
			ControlFlow::Break(())
		};
		let (stopped, iterations) = match solver {
			0 => {
				let mut mem =
					MemBuffer::new(cg::conjugate_gradient_scratch::<T>(
						identity,
						&a,
						1,
						Par::Seq.shape(),
					));
				let info = cg::conjugate_gradient(
					x.as_mut(),
					identity,
					&a,
					b.as_ref(),
					cg::CgParams {
						initial_guess: InitialGuessStatus::Zero,
						rel_tolerance: from_f64(tol),
						max_iters: 8,
						..Default::default()
					},
					callback,
					Par::Seq,
					MemStack::new(&mut mem),
				)
				.unwrap();
				(info.stopped, info.iter_count)
			},
			1 => {
				let mut mem = MemBuffer::new(bi::bicgstab_scratch::<T>(
					identity,
					identity,
					&a,
					1,
					Par::Seq.shape(),
				));
				let info = bi::bicgstab(
					x.as_mut(),
					identity,
					identity,
					&a,
					b.as_ref(),
					bi::BicgParams {
						initial_guess: InitialGuessStatus::Zero,
						rel_tolerance: from_f64(tol),
						max_iters: 8,
						..Default::default()
					},
					callback,
					Par::Seq,
					MemStack::new(&mut mem),
				)
				.unwrap();
				(info.stopped, info.iter_count)
			},
			_ => {
				let mut mem = MemBuffer::new(ls::lsmr_scratch::<T>(
					identity,
					&a,
					1,
					Par::Seq.shape(),
				));
				let info = ls::lsmr(
					x.as_mut(),
					identity,
					&a,
					b.as_ref(),
					ls::LsmrParams {
						initial_guess: InitialGuessStatus::Zero,
						rel_tolerance: from_f64(tol),
						max_iters: 8,
						..Default::default()
					},
					callback,
					Par::Seq,
					MemStack::new(&mut mem),
				)
				.unwrap();
				(info.stopped, info.iter_count)
			},
		};
		assert!(stopped);
		assert_eq!(iterations, 1);
		assert_eq!(calls, 1);
		assert!((&x - &observed).norm_l2() <= from_f64::<T::Real>(tol));
		assert!(
			(&a * &x - &b).norm_l2() > from_f64::<T::Real>(tol),
			"test must stop before convergence"
		);
	}
	// Both the legacy unit callback and continuing ControlFlow keep convergence.
	for controlled in [false, true] {
		let mut x = Mat::<T>::zeros(4, 1);
		let mut calls = 0;
		let mut mem = MemBuffer::new(bi::bicgstab_scratch::<T>(
			identity,
			identity,
			identity,
			1,
			Par::Seq.shape(),
		));
		let params = bi::BicgParams::<T::Real>::default();
		let info = if controlled {
			bi::bicgstab(
				x.as_mut(),
				identity,
				identity,
				identity,
				b.as_ref(),
				params,
				|_| {
					calls += 1;
					ControlFlow::<()>::Continue(())
				},
				Par::Seq,
				MemStack::new(&mut mem),
			)
		} else {
			bi::bicgstab(
				x.as_mut(),
				identity,
				identity,
				identity,
				b.as_ref(),
				params,
				|_| {
					calls += 1;
				},
				Par::Seq,
				MemStack::new(&mut mem),
			)
		}
		.unwrap();
		assert!(!info.stopped);
		assert_eq!(info.iter_count, 1);
		assert_eq!(calls, 1, "converged half step must reach callback");
		assert!((&x - &b).norm_l2() <= from_f64::<T::Real>(tol));
	}
}
#[test]
fn f32_control() {
	check::<f32>(1e-4, |r, _| r as f32);
}
#[test]
fn f64_control() {
	check::<f64>(1e-12, |r, _| r);
}
#[test]
fn c32_control() {
	check::<c32>(1e-4, |r, i| c32::new(r as f32, i as f32));
}
#[test]
fn c64_control() {
	check::<c64>(1e-12, c64::new);
}

#[test]
fn initial_convergence_does_not_call_stop_callback() {
	for (n, k) in [(0, 0), (4, 1), (4, 2)] {
		let a = IdentityPrecond { dim: n };
		let b = Mat::<f64>::from_fn(n, k, |i, j| 1.0 + (i + j) as f64);
		for solver in 0..3 {
			let mut x = b.clone();
			let mut calls = 0;
			let callback = |_: colmat::MatRef<'_, f64>| {
				calls += 1;
				ControlFlow::Break(())
			};
			let (stopped, iterations) = match solver {
				0 => {
					let mut mem =
						MemBuffer::new(cg::conjugate_gradient_scratch::<f64>(
							a,
							a,
							k,
							Par::Seq.shape(),
						));
					let info = cg::conjugate_gradient(
						x.as_mut(),
						a,
						a,
						b.as_ref(),
						Default::default(),
						callback,
						Par::Seq,
						MemStack::new(&mut mem),
					)
					.unwrap();
					(info.stopped, info.iter_count)
				},
				1 => {
					let mut mem = MemBuffer::new(bi::bicgstab_scratch::<f64>(
						a,
						a,
						a,
						k,
						Par::Seq.shape(),
					));
					let info = bi::bicgstab(
						x.as_mut(),
						a,
						a,
						a,
						b.as_ref(),
						Default::default(),
						callback,
						Par::Seq,
						MemStack::new(&mut mem),
					)
					.unwrap();
					(info.stopped, info.iter_count)
				},
				_ => {
					let mut mem = MemBuffer::new(ls::lsmr_scratch::<f64>(
						a,
						a,
						k,
						Par::Seq.shape(),
					));
					let info = ls::lsmr(
						x.as_mut(),
						a,
						a,
						b.as_ref(),
						Default::default(),
						callback,
						Par::Seq,
						MemStack::new(&mut mem),
					)
					.unwrap();
					(info.stopped, info.iter_count)
				},
			};
			assert!(!stopped);
			assert_eq!(iterations, 0);
			assert_eq!(calls, 0);
			assert_eq!((&x - &b).norm_l2(), 0.0);
		}
	}
}
