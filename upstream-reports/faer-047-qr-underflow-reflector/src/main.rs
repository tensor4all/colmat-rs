// Unpivoted QR of a well-conditioned matrix whose entries are scaled near the
// bottom of the f64 range. The Householder reflector is formed from the column
// at its original scale: its tau and the products in the trailing update are
// about 1e-308 (or 1e-616), so the update underflows and R is wrong while the
// call still reports success.
//
// The check rebuilds A from Q and R after dividing both by the scale, so the
// check itself does not underflow. A correct factorization has a relative
// residual of the order of machine epsilon, as LAPACK's ?geqrf does.
use faer::Mat;

fn base(m: usize, n: usize) -> Mat<f64> {
	Mat::from_fn(m, n, |i, j| {
		((7 * i + 13 * j + 3) % 11) as f64 / 11.0 - 0.45
			+ if i == j { 1.0 } else { 0.0 }
	})
}

fn residual(m: usize, n: usize, scale: f64) -> f64 {
	let a = base(m, n);
	let a = Mat::from_fn(m, n, |i, j| a[(i, j)] * scale);
	let qr = a.as_ref().qr();
	let r = qr.thin_R();
	let r = Mat::from_fn(r.nrows(), r.ncols(), |i, j| r[(i, j)] / scale);
	let rebuilt = qr.compute_thin_Q() * &r;
	let (mut num, mut den) = (0.0, 0.0);
	for j in 0..n {
		for i in 0..m {
			let y = a[(i, j)] / scale;
			num += (rebuilt[(i, j)] - y).powi(2);
			den += y * y;
		}
	}
	(num / den).sqrt()
}

fn main() {
	let mut bad = false;

	for (m, n) in [(6usize, 4usize), (4, 4), (4, 6)] {
		for scale in [1.0f64, 16.0 * f64::MIN_POSITIVE, f64::MIN_POSITIVE] {
			let actual = residual(m, n, scale);
			println!(
				"{m} x {n}, scale {scale:e}: relative residual {actual:.2e} (expected at most 1e-13)"
			);
			if !(actual <= 1e-13) {
				bad = true;
			}
		}
	}

	if bad {
		println!();
		println!(
			"WRONG: QR reports success with a factorization that does not reconstruct the input"
		);
		std::process::exit(1);
	}
}
