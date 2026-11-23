// Generalized eigenproblem S u = lambda T u with the upper triangular pencil
//     S = [[1, 1], [0, s11]],  T = diag(1, s11 / 2).
// The eigenvalues are 1 and 2; the eigenvector of lambda = 2 is (1, 1)/sqrt(2).
// s11 is a diagonal about 1e-200, while the factor norms are O(1).
//
// The shared conversion stage computes the 1 x 1 block's coefficient pair as
// (S_jj * temp * ascale, T_jj * temp * bscale) with temp = max(...). LAPACK's
// ?tgevc divides by that maximum instead, so the pair is O(1) rather than
// carrying temp^2 = 4e-400, which underflows to zero; the normalization then
// produces NaN from an Ok call.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::gevd;
use faer::{diag::Diag, Mat, Par};

fn main() {
	let mut bad = false;

	for s11 in [2.0f64, 2e-8, 2e-100, 2e-150, 2e-200] {
		let s = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
			(0, 0) => 1.0,
			(0, 1) => 1.0,
			(1, 1) => s11,
			_ => 0.0,
		});
		let t = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
			(0, 0) => 1.0,
			(1, 1) => 0.5 * s11,
			_ => 0.0,
		});
		let mut a = s.clone();
		let mut b = t.clone();
		let (mut alpha, mut imag, mut beta) = (
			Diag::<f64>::zeros(2),
			Diag::<f64>::zeros(2),
			Diag::<f64>::zeros(2),
		);
		let mut right = Mat::<f64>::zeros(2, 2);
		let mut mem = MemBuffer::new(gevd::gevd_scratch::<f64>(
			2,
			gevd::ComputeEigenvectors::No,
			gevd::ComputeEigenvectors::Yes,
			Par::Seq,
			Default::default(),
		));
		let out = gevd::gevd_real(
			a.as_mut(),
			b.as_mut(),
			alpha.as_mut(),
			imag.as_mut(),
			beta.as_mut(),
			None,
			Some(right.as_mut()),
			Par::Seq,
			MemStack::new(&mut mem),
			Default::default(),
		);

		let expected = [0.5f64.sqrt(), 0.5f64.sqrt()];
		match out {
			Err(err) => {
				println!(
					"s11 = {s11:e}: expected {expected:?}, actual Err({err:?})"
				);
				bad = true;
			},
			Ok(()) => {
				// The eigenvalue 2 column, up to the eigenvalue's rounding.
				let mut k = 0usize;
				let mut best = f64::INFINITY;
				for i in 0..2 {
					let d = (alpha[i] / beta[i] - 2.0).abs();
					if d < best {
						best = d;
						k = i;
					}
				}
				let lambda = alpha[k] / beta[k];
				let v = [right[(0, k)], right[(1, k)]];
				// |S v beta - T v alpha|, relative to the larger term.
				let ab = [
					s[(0, 0)] * v[0] + s[(0, 1)] * v[1],
					s[(1, 0)] * v[0] + s[(1, 1)] * v[1],
				];
				let bb = [
					t[(0, 0)] * v[0] + t[(0, 1)] * v[1],
					t[(1, 0)] * v[0] + t[(1, 1)] * v[1],
				];
				let lhs = [ab[0] * beta[k], ab[1] * beta[k]];
				let rhs = [bb[0] * alpha[k], bb[1] * alpha[k]];
				let res = [(lhs[0] - rhs[0]).abs(), (lhs[1] - rhs[1]).abs()];
				let scale = lhs[0]
					.abs()
					.max(lhs[1].abs())
					.max(rhs[0].abs())
					.max(rhs[1].abs())
					.max(f64::MIN_POSITIVE);
				let rel = res[0].max(res[1]) / scale;

				let finite = v.iter().all(|x| x.is_finite());
				println!(
					"s11 = {s11:e}: expected {expected:?} for lambda 2, actual v = {v:?}, lambda = {lambda:e}, relative residual {rel:.2e}"
				);
				if !finite || !(rel <= 1e-10) || (lambda - 2.0).abs() > 1e-6 {
					bad = true;
				}
			},
		}
	}

	if bad {
		println!();
		println!(
			"WRONG: the generalized eigenvector of a representable pencil is missing or NaN"
		);
		std::process::exit(1);
	}
}
