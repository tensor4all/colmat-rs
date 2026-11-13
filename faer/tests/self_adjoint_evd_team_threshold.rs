//! tensor4all/colmat-rs#165, increment 2: the auto `SelfAdjointEvdParams::recursion_threshold`
//! is a function of the team. With two or more participants the tridiagonal solve enters the
//! divide and conquer from n > 16 (the measured `recursion_threshold = 16` of the 2026-11-10 and
//! 2026-11-12 records); without a team, or on a team of one, it stays the QR iteration below
//! n = 128 as before. Any value the caller sets other than the auto one is used as given at every
//! thread count, so `usize::MAX` still forces the QR iteration on a team.
//!
//! The bit-for-bit comparison is what observes the path: at a fixed thread count each path is
//! deterministic, and the QR iteration and the divide and conquer give different bits for the
//! same input. Before this change, the auto call on a team ran the QR iteration at n = 64 and
//! n = 128, so it matched `usize::MAX` and not `16`.
#![cfg(all(feature = "std", feature = "threadteam"))]
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::evd::{
	ComputeEigenvectors, SelfAdjointEvdParams, self_adjoint_evd,
	self_adjoint_evd_scratch, tridiagonal_self_adjoint_evd,
	tridiagonal_self_adjoint_evd_scratch,
};
use colmat::traits::ComplexField;
use colmat::traits::math_utils::{abs, conj, eps, from_f64};
use colmat::{Auto, Mat, MatRef, Par, Spec, c32, c64};

/// The bytes of a native scalar (`f32`, `f64`, `c32`, `c64`: no padding).
fn bits<T>(x: &T) -> &[u8] {
	// SAFETY: a native scalar is plain data of `size_of::<T>()` initialised bytes
	unsafe {
		core::slice::from_raw_parts(
			x as *const T as *const u8,
			core::mem::size_of::<T>(),
		)
	}
}

fn same_bits<T>(a: &[T], b: &[T]) -> bool {
	a.len() == b.len() && a.iter().zip(b).all(|(x, y)| bits(x) == bits(y))
}

/// A deterministic value in `[-0.5, 0.5)` for the entry `k`.
fn rnd(k: u64) -> f64 {
	(k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64 / (1u64 << 53) as f64
		- 0.5
}

fn hermitian<T: ComplexField>(
	n: usize,
	make: &dyn Fn(f64, f64) -> T,
) -> Mat<T> {
	Mat::from_fn(n, n, |i, j| {
		let (p, q) = if i >= j { (i, j) } else { (j, i) };
		let k = (p * n + q) as u64;
		let v = make(rnd(2 * k), if p == q { 0.0 } else { rnd(2 * k + 1) });
		if i >= j { v } else { conj(&v) }
	})
}

/// `None` is the auto threshold; `Some(t)` sets `t`.
fn params<T: ComplexField>(
	threshold: Option<usize>,
) -> Spec<SelfAdjointEvdParams, T> {
	let mut p = <SelfAdjointEvdParams as Auto<T>>::auto();
	if let Some(t) = threshold {
		p.recursion_threshold = t;
	}
	Spec::new(p)
}

/// The eigenvalues and the eigenvectors (column major, flattened) of the dense call.
fn dense<T: ComplexField>(
	a: MatRef<'_, T>,
	par: Par,
	threshold: Option<usize>,
) -> (Vec<T>, Vec<T>) {
	let n = a.nrows();
	let p = params::<T>(threshold);
	let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<T>(
		n,
		ComputeEigenvectors::Yes,
		par.shape(),
		p,
	));
	let mut s = Mat::<T>::zeros(n, 1);
	let mut u = Mat::<T>::zeros(n, n);
	self_adjoint_evd(
		a,
		s.as_mut().col_mut(0).as_diagonal_mut(),
		Some(u.as_mut()),
		par,
		MemStack::new(&mut mem),
		p,
	)
	.unwrap();
	let s = (0..n).map(|i| s[(i, 0)].clone()).collect();
	let u = (0..n)
		.flat_map(|j| (0..n).map(move |i| (i, j)))
		.map(|(i, j)| u[(i, j)].clone())
		.collect();
	(s, u)
}

/// The same for the public tridiagonal solve, on the tridiagonal `(diag, subdiag)`.
fn tridiagonal<T: ComplexField>(
	diag: &[T],
	subdiag: &[T],
	par: Par,
	threshold: Option<usize>,
) -> (Vec<T>, Vec<T>) {
	let n = diag.len();
	let p = params::<T>(threshold);
	let mut mem = MemBuffer::new(tridiagonal_self_adjoint_evd_scratch::<T>(
		n,
		ComputeEigenvectors::Yes,
		par.shape(),
		p,
	));
	let d = Mat::from_fn(n, 1, |i, _| diag[i].clone());
	let e = Mat::from_fn(n, 1, |i, _| subdiag[i].clone());
	let mut s = Mat::<T>::zeros(n, 1);
	let mut u = Mat::<T>::zeros(n, n);
	tridiagonal_self_adjoint_evd(
		d.as_ref().col(0).as_diagonal(),
		e.as_ref().col(0).as_diagonal(),
		s.as_mut().col_mut(0).as_diagonal_mut(),
		Some(u.as_mut()),
		par,
		MemStack::new(&mut mem),
		p,
	)
	.unwrap();
	let s = (0..n).map(|i| s[(i, 0)].clone()).collect();
	let u = (0..n)
		.flat_map(|j| (0..n).map(move |i| (i, j)))
		.map(|(i, j)| u[(i, j)].clone())
		.collect();
	(s, u)
}

/// `max |A U - U S| <= 60 n eps max |A|`, the CHKST-style residual bound of the LAPACK fixtures.
fn residual_ok<T: ComplexField>(a: MatRef<'_, T>, s: &[T], u: &[T]) -> bool {
	let n = a.nrows();
	let u = Mat::from_fn(n, n, |i, j| u[j * n + i].clone());
	let us = Mat::from_fn(n, n, |i, j| u[(i, j)].clone() * s[j].clone());
	let r = a * &u - us;
	let mut rmax = from_f64::<T::Real>(0.0);
	let mut amax = from_f64::<T::Real>(0.0);
	for j in 0..n {
		for i in 0..n {
			let (x, y) = (abs(&r[(i, j)]), abs(&a[(i, j)]));
			if x > rmax {
				rmax = x;
			}
			if y > amax {
				amax = y;
			}
		}
	}
	rmax <= from_f64::<T::Real>(60.0 * n as f64) * eps::<T::Real>() * amax
}

fn check<T: ComplexField>(n: usize, make: &dyn Fn(f64, f64) -> T) {
	let a = hermitian::<T>(n, make);
	let a = a.as_ref();
	// one thread: the auto threshold is the QR iteration below 128 (unchanged)
	let seq_auto = dense::<T>(a, Par::Seq, None);
	let seq_qr = dense::<T>(a, Par::Seq, Some(usize::MAX));
	if n < 128 {
		assert!(
			same_bits(&seq_auto.0, &seq_qr.0)
				&& same_bits(&seq_auto.1, &seq_qr.1),
			"n = {n}: Par::Seq auto is not the QR path"
		);
	}
	// a team of one is one thread
	colmat::team::with_team(1, |par| {
		let one_auto = dense::<T>(a, par, None);
		assert!(
			same_bits(&one_auto.0, &seq_auto.0)
				&& same_bits(&one_auto.1, &seq_auto.1),
			"n = {n}: a team of one is not Par::Seq"
		);
	});
	for threads in [2, 4] {
		colmat::team::with_team(threads, |mut par| {
			let auto = dense::<T>(a, par.reborrow(), None);
			let dc = dense::<T>(a, par.reborrow(), Some(16));
			let qr = dense::<T>(a, par.reborrow(), Some(usize::MAX));
			assert!(
				same_bits(&auto.0, &dc.0) && same_bits(&auto.1, &dc.1),
				"n = {n}, {threads} threads: the auto threshold is not the divide and conquer from 16"
			);
			assert!(
				!same_bits(&qr.1, &dc.1),
				"n = {n}, {threads} threads: the two paths are not told apart"
			);
			assert!(
				residual_ok(a, &auto.0, &auto.1),
				"n = {n}, {threads} threads: residual"
			);
			assert!(
				residual_ok(a, &qr.0, &qr.1),
				"n = {n}, {threads} threads: an explicit usize::MAX still solves"
			);

			// the public tridiagonal solve takes the same gate
			let diag: Vec<T> =
				(0..n).map(|i| from_f64(2.0 + rnd(7 * i as u64))).collect();
			let sub: Vec<T> = (0..n)
				.map(|i| {
					if i + 1 < n {
						from_f64(rnd(7 * i as u64 + 3))
					} else {
						from_f64(0.0)
					}
				})
				.collect();
			let t_auto = tridiagonal::<T>(&diag, &sub, par.reborrow(), None);
			let t_dc = tridiagonal::<T>(&diag, &sub, par.reborrow(), Some(16));
			assert!(
				same_bits(&t_auto.0, &t_dc.0) && same_bits(&t_auto.1, &t_dc.1),
				"n = {n}, {threads} threads: tridiagonal_self_adjoint_evd's auto threshold is not 16"
			);
		});
	}
}

#[test]
fn auto_threshold_takes_divide_and_conquer_on_a_team() {
	for n in [64, 128] {
		check::<f64>(n, &|re, _| re);
		check::<c64>(n, &|re, im| c64::new(re, im));
		check::<f32>(n, &|re, _| re as f32);
		check::<c32>(n, &|re, im| c32::new(re as f32, im as f32));
	}
}
