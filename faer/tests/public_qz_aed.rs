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
