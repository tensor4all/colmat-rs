// The thin SVD of a well-conditioned matrix whose entries are scaled to either
// end of the f64 range. The dense driver runs the bidiagonalization on the input
// as given, without LAPACK ?gesvd/?gesdd's xLASCL step that scales the matrix
// into [smlnum, bignum] first. Near overflow the reflectors' norms overflow and
// the call returns NoConvergence; near underflow the factors lose their digits.
//
// The check rebuilds A from U, S and V after dividing S by the scale, so the
// check itself does not overflow or underflow. A correct factorization has a
// relative residual of the order of machine epsilon.
use faer::Mat;

fn base(m: usize, n: usize) -> Mat<f64> {
	Mat::from_fn(m, n, |i, j| {
		((7 * i + 13 * j + 3) % 11) as f64 / 11.0 - 0.45
			+ if i == j { 1.0 } else { 0.0 }
	})
}

// Returns None for Err, Some(residual) for Ok.
fn check(m: usize, n: usize, scale: f64) -> Option<f64> {
	let a = base(m, n);
	let a = Mat::from_fn(m, n, |i, j| a[(i, j)] * scale);
	let svd = a.as_ref().thin_svd().ok()?;
	let k = svd.S().column_vector().nrows();
	let us = Mat::from_fn(m, k, |i, l| {
		svd.U()[(i, l)] * (svd.S().column_vector()[l] / scale)
	});
	let rebuilt = &us * svd.V().adjoint();
	let (mut num, mut den) = (0.0, 0.0);
	for j in 0..n {
		for i in 0..m {
			let y = a[(i, j)] / scale;
			num += (rebuilt[(i, j)] - y).powi(2);
			den += y * y;
		}
	}
	Some((num / den).sqrt())
}

fn main() {
	let mut bad = false;

	for (m, n) in [(6usize, 4usize), (4, 4), (4, 6)] {
		for scale in [
			1.0f64,
			0.25 * f64::MAX,
			0.5 * f64::MAX,
			16.0 * f64::MIN_POSITIVE,
			f64::MIN_POSITIVE,
		] {
			match check(m, n, scale) {
				None => {
					println!(
						"{m} x {n}, scale {scale:e}: Err(NoConvergence) (expected Ok with relative residual at most 1e-13)"
					);
					bad = true;
				},
				Some(actual) => {
					println!(
						"{m} x {n}, scale {scale:e}: relative residual {actual:.2e} (expected at most 1e-13)"
					);
					if !(actual <= 1e-13) {
						bad = true;
					}
				},
			}
		}
	}

	if bad {
		println!();
		println!(
			"WRONG: thin SVD fails or loses its factors for representable input"
		);
		std::process::exit(1);
	}
}
