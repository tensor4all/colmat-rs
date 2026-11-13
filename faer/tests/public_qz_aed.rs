//! `aggressive_early_deflation` marks a fully deflated window with the
//! `kwtop - 1` sentinel (`usize::MAX` at row zero). The deflation count must
//! recognise that sentinel instead of subtracting it from `ihi`, which
//! overflows in a checked build (tensor4all/faer#254).

use colmat::dyn_stack::{MemBuffer, MemStack, StackReq};
use colmat::linalg::gevd::{self, ComputeEigenvectors};
use colmat::{Col, Mat, Par, ParShape};

fn sample(i: usize, j: usize) -> f64 {
	let mut x = (i as u64 + 7).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64 + 19);
	x ^= x >> 30;
	x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
	x ^= x >> 27;
	(x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

fn pencil(n: usize) -> (Mat<f64>, Mat<f64>) {
	let a = Mat::<f64>::from_fn(n, n, |i, j| {
		if i == j {
			1.0 + i as f64 / n as f64
		} else if i == j + 1 {
			0.2
		} else if i < j {
			0.02 * sample(i, j)
		} else {
			0.0
		}
	});
	let b = Mat::<f64>::from_fn(n, n, |i, j| {
		if i == j {
			2.0
		} else if i < j {
			0.01 * sample(j, i)
		} else {
			0.0
		}
	});
	(a, b)
}

#[test]
fn fully_deflated_row_zero_window_does_not_overflow() {
	for n in [18, 78] {
		let (mut a, mut b) = pencil(n);
		let mut q = Mat::<f64>::identity(n, n);
		let mut z = Mat::<f64>::identity(n, n);
		let mut ar = Col::<f64>::zeros(n);
		let mut ai = Col::<f64>::zeros(n);
		let mut beta = Col::<f64>::zeros(n);
		let params: gevd::GeneralizedSchurParams = colmat::auto!(f64);

		let req = gevd::qz_real::hessenberg_to_qz_scratch::<f64>(n, ParShape::Seq, params);
		// Ample, valid storage: this test isolates the index arithmetic from
		// the separately reported workspace query defect (#252).
		let mut memory = MemBuffer::new(StackReq::all_of(&[
			req,
			StackReq::new::<u8>(1_048_576),
		]));
		gevd::qz_real::hessenberg_to_qz(
			a.as_mut(),
			b.as_mut(),
			Some(q.as_mut()),
			Some(z.as_mut()),
			ar.as_mut(),
			ai.as_mut(),
			beta.as_mut(),
			ComputeEigenvectors::Yes,
			Par::Seq,
			params,
			MemStack::new(&mut memory),
		);

		// The pencil is non-singular (B is unit-triangular-scaled with 2 on the
		// diagonal), so every generalised eigenvalue must have beta != 0.
		for k in 0..n {
			assert!(
				beta[k] != 0.0,
				"n={n}: beta[{k}] == 0 in a non-singular pencil"
			);
			assert!(ar[k].is_finite() && ai[k].is_finite(), "n={n}: non-finite eigenvalue at {k}");
		}
	}
}

// tensor4all/colmat-rs#117. Once the active block is narrower than the
// blocking threshold, LAPACK's DLAQZ0 hands the whole block to AED
// (`NW = ISTOP-ISTART+1`), so the window's own QZ deflates it outright. The
// window was additionally clamped to the even-rounded `(n-3)/3` cap; a clamped
// window starts below `istart`, its spike is not negligible, nothing deflates,
// and the driver re-ran AED without a sweep until `maxit` was exhausted. Each
// of those thousands of window updates rounded into (S, T, Q, Z): at n = 78
// with the maximum recommended window the DGET54-style ratio was 178.6 and
// ||Q^T Q - I|| was 3.7e-12 (LAPACK: 0.77 and 3e-14).

extern "C" fn maximum_window(n: usize, _active: usize) -> usize {
	let w = n.saturating_sub(3) / 3;
	w + w % 2
}

fn qz_cases() -> [(usize, Option<extern "C" fn(usize, usize) -> usize>, Option<usize>); 3] {
	// (n, recommended window, blocking threshold)
	[(18, None, Some(15)), (78, Some(maximum_window), None), (100, Some(maximum_window), None)]
}

fn qz_params<T: colmat::traits::ComplexField>(
	window: Option<extern "C" fn(usize, usize) -> usize>,
	blocking_threshold: Option<usize>,
) -> gevd::GeneralizedSchurParams {
	let mut params: gevd::GeneralizedSchurParams = colmat::auto!(T);
	if let Some(window) = window {
		params.recommended_deflation_window = window;
	}
	if let Some(threshold) = blocking_threshold {
		params.blocking_threshold = threshold;
	}
	params
}

fn one_norm<T: colmat::traits::ComplexField>(m: colmat::MatRef<'_, T>) -> f64
where
	T::Real: Into<f64>,
{
	use colmat::traits::math_utils::abs;
	(0..m.ncols())
		.map(|j| (0..m.nrows()).map(|i| abs(&m[(i, j)]).into()).sum::<f64>())
		.fold(0.0, f64::max)
}

/// DGET54's coupled ratio and DGET51's orthogonality ratios, both in units of
/// `n * eps`, so the threshold 20 is LAPACK's for all four scalar types.
fn qz_quality<T: colmat::traits::ComplexField>(
	a: &Mat<T>,
	b: &Mat<T>,
	s: &Mat<T>,
	t: &Mat<T>,
	q: &Mat<T>,
	z: &Mat<T>,
) -> (f64, f64, f64)
where
	T::Real: Into<f64>,
{
	let n = a.nrows();
	let eps: f64 = colmat::traits::math_utils::eps::<T::Real>().into();
	let ea = q * s * z.adjoint() - a;
	let eb = q * t * z.adjoint() - b;
	let scale = one_norm(a.as_ref()).max(one_norm(b.as_ref()));
	let coupled = one_norm(ea.as_ref()).max(one_norm(eb.as_ref())) / scale / (2.0 * n as f64 * eps);
	let orth = |m: &Mat<T>| one_norm((m.adjoint() * m - Mat::<T>::identity(n, n)).as_ref()) / (n as f64 * eps);
	(coupled, orth(q), orth(z))
}

fn real_qz_window_quality<T: colmat::traits::RealField + Into<f64>>(lift: fn(f64) -> T) {
	for (n, window, threshold) in qz_cases() {
		let (a, b) = pencil(n);
		let a = Mat::<T>::from_fn(n, n, |i, j| lift(a[(i, j)]));
		let b = Mat::<T>::from_fn(n, n, |i, j| lift(b[(i, j)]));
		let (mut s, mut t) = (a.clone(), b.clone());
		let mut q = Mat::<T>::identity(n, n);
		let mut z = Mat::<T>::identity(n, n);
		let mut ar = Col::<T>::zeros(n);
		let mut ai = Col::<T>::zeros(n);
		let mut beta = Col::<T>::zeros(n);
		let params = qz_params::<T>(window, threshold);
		let req = gevd::qz_real::hessenberg_to_qz_scratch::<T>(n, ParShape::Seq, params);
		let mut memory = MemBuffer::new(req);
		gevd::qz_real::hessenberg_to_qz(
			s.as_mut(),
			t.as_mut(),
			Some(q.as_mut()),
			Some(z.as_mut()),
			ar.as_mut(),
			ai.as_mut(),
			beta.as_mut(),
			ComputeEigenvectors::Yes,
			Par::Seq,
			params,
			MemStack::new(&mut memory),
		);
		let (coupled, q_orth, z_orth) = qz_quality(&a, &b, &s, &t, &q, &z);
		let ty = core::any::type_name::<T>();
		assert!(coupled < 20.0, "{ty} n={n}: coupled ratio {coupled:e}");
		assert!(q_orth < 20.0, "{ty} n={n}: Q orthogonality ratio {q_orth:e}");
		assert!(z_orth < 20.0, "{ty} n={n}: Z orthogonality ratio {z_orth:e}");
	}
}

fn complex_qz_window_quality<T: colmat::traits::ComplexField>(lift: fn(f64, f64) -> T)
where
	T::Real: Into<f64>,
{
	for (n, window, threshold) in qz_cases() {
		let (a, b) = pencil(n);
		// An imaginary upper triangle keeps the pencil genuinely complex.
		let a = Mat::<T>::from_fn(n, n, |i, j| lift(a[(i, j)], if i <= j { 0.01 * sample(i + 3, j) } else { 0.0 }));
		let b = Mat::<T>::from_fn(n, n, |i, j| lift(b[(i, j)], 0.0));
		let (mut s, mut t) = (a.clone(), b.clone());
		let mut q = Mat::<T>::identity(n, n);
		let mut z = Mat::<T>::identity(n, n);
		let mut alpha = Col::<T>::zeros(n);
		let mut beta = Col::<T>::zeros(n);
		let params = qz_params::<T>(window, threshold);
		let req = gevd::qz_cplx::hessenberg_to_qz_scratch::<T>(n, ParShape::Seq, params);
		let mut memory = MemBuffer::new(req);
		gevd::qz_cplx::hessenberg_to_qz(
			s.as_mut(),
			t.as_mut(),
			Some(q.as_mut()),
			Some(z.as_mut()),
			alpha.as_mut(),
			beta.as_mut(),
			ComputeEigenvectors::Yes,
			Par::Seq,
			params,
			MemStack::new(&mut memory),
		);
		let (coupled, q_orth, z_orth) = qz_quality(&a, &b, &s, &t, &q, &z);
		let ty = core::any::type_name::<T>();
		assert!(coupled < 20.0, "{ty} n={n}: coupled ratio {coupled:e}");
		assert!(q_orth < 20.0, "{ty} n={n}: Q unitarity ratio {q_orth:e}");
		assert!(z_orth < 20.0, "{ty} n={n}: Z unitarity ratio {z_orth:e}");
	}
}

#[test]
fn real_qz_small_active_block_window_keeps_accuracy() {
	real_qz_window_quality::<f64>(|x| x);
	real_qz_window_quality::<f32>(|x| x as f32);
}

#[test]
fn complex_qz_small_active_block_window_keeps_accuracy() {
	complex_qz_window_quality::<colmat::c64>(colmat::c64::new);
	complex_qz_window_quality::<colmat::c32>(|re, im| colmat::c32::new(re as f32, im as f32));
}

extern "C" fn two_shifts(_n: usize, _active: usize) -> usize {
	2
}

/// The multishift sweep's own workspace: with few shifts the chase block `nblock = nsr + itemp1`
/// exceeds `2 * nsr`, and the sweep's `n x (ns + np)` temporaries did not fit the `n x 2 nsr` it
/// reserved. On `4abe04d` the exact query panicked for complex QZ at n = 15 (c32, c64), 18 and 20
/// (c32) with two shifts; the AED reservation of #117 happens to cover these sizes, so this pins
/// the sweep's own bound.
#[test]
fn qz_two_shift_sweep_fits_the_queried_workspace() {
	fn real<T: colmat::traits::RealField>(n: usize, threshold: usize, lift: fn(f64) -> T) {
		let (a, b) = pencil(n);
		let mut s = Mat::<T>::from_fn(n, n, |i, j| lift(a[(i, j)]));
		let mut t = Mat::<T>::from_fn(n, n, |i, j| lift(b[(i, j)]));
		let mut q = Mat::<T>::identity(n, n);
		let mut z = Mat::<T>::identity(n, n);
		let (mut ar, mut ai, mut be) = (Col::<T>::zeros(n), Col::<T>::zeros(n), Col::<T>::zeros(n));
		let mut params = qz_params::<T>(None, Some(threshold));
		params.recommended_shift_count = two_shifts;
		let req = gevd::qz_real::hessenberg_to_qz_scratch::<T>(n, ParShape::Seq, params);
		let mut memory = MemBuffer::new(req);
		gevd::qz_real::hessenberg_to_qz(
			s.as_mut(), t.as_mut(), Some(q.as_mut()), Some(z.as_mut()), ar.as_mut(), ai.as_mut(), be.as_mut(),
			ComputeEigenvectors::Yes, Par::Seq, params, MemStack::new(&mut memory),
		);
		assert!(ar.is_all_finite() && ai.is_all_finite() && be.is_all_finite());
	}
	fn cplx<T: colmat::traits::ComplexField>(n: usize, threshold: usize, lift: fn(f64, f64) -> T) {
		let (a, b) = pencil(n);
		let mut s = Mat::<T>::from_fn(n, n, |i, j| lift(a[(i, j)], if i <= j { 0.01 * sample(i + 3, j) } else { 0.0 }));
		let mut t = Mat::<T>::from_fn(n, n, |i, j| lift(b[(i, j)], 0.0));
		let mut q = Mat::<T>::identity(n, n);
		let mut z = Mat::<T>::identity(n, n);
		let (mut al, mut be) = (Col::<T>::zeros(n), Col::<T>::zeros(n));
		let mut params = qz_params::<T>(None, Some(threshold));
		params.recommended_shift_count = two_shifts;
		let req = gevd::qz_cplx::hessenberg_to_qz_scratch::<T>(n, ParShape::Seq, params);
		let mut memory = MemBuffer::new(req);
		gevd::qz_cplx::hessenberg_to_qz(
			s.as_mut(), t.as_mut(), Some(q.as_mut()), Some(z.as_mut()), al.as_mut(), be.as_mut(),
			ComputeEigenvectors::Yes, Par::Seq, params, MemStack::new(&mut memory),
		);
		assert!(al.is_all_finite() && be.is_all_finite());
	}
	for (n, threshold) in [(15, 15), (18, 15), (20, 20), (24, 15), (40, 15)] {
		real::<f64>(n, threshold, |x| x);
		real::<f32>(n, threshold, |x| x as f32);
		cplx::<colmat::c64>(n, threshold, colmat::c64::new);
		cplx::<colmat::c32>(n, threshold, |re, im| colmat::c32::new(re as f32, im as f32));
	}
}
