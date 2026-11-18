use colmat::traits::ext::{ComplexField, ComplexFieldExt};
use colmat::traits::math_utils::{eps, min_positive, one};
use colmat::{Mat, Par, c32, c64};

fn check<T: ComplexField>(make: impl Fn(f64, f64) -> T) {
	let min = min_positive::<T::Real>();
	let tol = eps::<T::Real>() * T::Real::from_f64(64.0);
	for (m, n) in [(6, 4), (4, 4), (4, 6)] {
		let b = Mat::from_fn(m, n, |i, j| {
			let re = ((7 * i + 13 * j + 3) % 11) as f64 / 11.0 - 0.45
				+ if i == j { 1.0 } else { 0.0 };
			let im = (((3 * i + 5 * j) % 5) as f64 - 2.0) / 32.0;
			make(re, im)
		});
		let ordinary = b.qr(Par::Seq);
		for scale in
			[one::<T::Real>(), &min * T::Real::from_f64(16.0), min.copy()]
		{
			let a = Mat::from_fn(m, n, |i, j| b[(i, j)].mul_pow2(&scale));
			let qr = a.qr(Par::Seq);
			let q = qr.compute_thin_Q(Par::Seq);
			let r = qr.thin_R();
			let inv = scale.recip();
			let r = Mat::from_fn(r.nrows(), n, |i, j| r[(i, j)].mul_pow2(&inv));
			let target = Mat::from_fn(m, n, |i, j| a[(i, j)].mul_pow2(&inv));
			let residual = (&q * &r - &target).norm_l2() / target.norm_l2();
			assert!(
				residual < tol,
				"{m}x{n} at {scale:?}: residual {residual:?}"
			);

			let k = m.min(n);
			let orth = (q.adjoint() * &q - Mat::<T>::identity(k, k)).norm_l2();
			assert!(orth < tol, "{m}x{n} at {scale:?}: Q^H Q error {orth:?}");
			for j in 0..k {
				let x = &qr.Q_coeff()[(0, j)];
				let y = &ordinary.Q_coeff()[(0, j)];
				if y.is_finite() {
					assert!(
						(x - y).abs() < tol,
						"{m}x{n} at {scale:?}, tau {j}: {x:?} vs {y:?}"
					);
				} else {
					assert_eq!(x, y);
				}
			}
		}
	}
}

#[test]
fn qr_near_underflow() {
	check(|re, _| re as f32);
	check(|re, _| re);
	check(|re, im| c32::new(re as f32, im as f32));
	check(c64::new);
}
