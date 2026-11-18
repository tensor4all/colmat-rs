use colmat::linalg::svd::SvdError;
use colmat::traits::ext::{ComplexField, ComplexFieldExt};
use colmat::traits::math_utils::{eps, min_positive};
use colmat::{Mat, MatRef, Par, c32, c64};

fn check<T: ComplexField>(make: impl Fn(f64, f64) -> T, max: f64) {
	let min = min_positive::<T::Real>();
	let tol = eps::<T::Real>() * T::Real::from_f64(64.0);
	// Both sides of the QR-reduction threshold, including transposed inputs.
	for (m, n) in [(6, 4), (4, 4), (4, 6), (9, 4), (4, 9)] {
		let b = Mat::from_fn(m, n, |i, j| {
			let (re, im) = if i < 6 && j < 6 {
				(
					((7 * i + 13 * j + 3) % 11) as f64 / 11.0 - 0.45
						+ if i == j { 1.0 } else { 0.0 },
					(((3 * i + 5 * j) % 5) as f64 - 2.0) / 32.0,
				)
			} else {
				(0.0, 0.0)
			};
			make(re, im)
		});
		let expected = b.singular_values(Par::Seq).unwrap();
		for scale in [
			T::Real::from_f64(0.25 * max),
			T::Real::from_f64(0.5 * max),
			&min * T::Real::from_f64(16.0),
			min.copy(),
		] {
			// Copy into padding to exercise a view with neither stride equal to one.
			let storage = Mat::from_fn(2 * m, n, |i, j| {
				if i % 2 == 0 {
					b[(i / 2, j)].mul_pow2(&scale)
				} else {
					T::nan()
				}
			});
			// SAFETY: row 2*i and column j are inside storage, and storage remains live.
			let a = unsafe {
				MatRef::from_raw_parts(
					storage.as_ptr(),
					m,
					n,
					2isize,
					storage.col_stride(),
				)
			};
			let inv = scale.recip();
			let target = Mat::from_fn(m, n, |i, j| a[(i, j)].mul_pow2(&inv));
			let contiguous = a.to_owned();
			for a in [a, contiguous.as_ref().as_dyn_stride()] {
				let svd = a
					.thin_svd(Par::Seq)
					.unwrap_or_else(|e| panic!("{m}x{n} at {scale:?}: {e:?}"));
				let k = m.min(n);
				let s = svd.S().column_vector();
				let us = Mat::from_fn(m, k, |i, j| {
					&svd.U()[(i, j)] * s[j].mul_pow2(&inv)
				});
				let residual = (&us * svd.V().adjoint() - &target).norm_l2()
					/ target.norm_l2();
				assert!(
					residual < tol,
					"{m}x{n} at {scale:?}: residual {residual:?}"
				);
				let values = a.singular_values(Par::Seq).unwrap();
				for (s, want) in values.iter().zip(&expected) {
					let error = (s.mul_pow2(&inv) - want).abs();
					assert!(
						error < tol,
						"{m}x{n} at {scale:?}: value error {error:?}"
					);
				}
			}
		}
	}
}

#[test]
fn dense_svd_range() {
	check(|re, _| re as f32, f32::MAX as f64);
	check(|re, _| re, f64::MAX);
	check(|re, im| c32::new(re as f32, im as f32), f32::MAX as f64);
	check(c64::new, f64::MAX);
}

#[test]
fn dense_svd_rejects_nonfinite_and_unrepresentable_values() {
	for x in [f64::NAN, f64::INFINITY, f64::MAX] {
		let a = Mat::from_fn(4, 1, |_, _| x);
		assert_eq!(a.thin_svd(Par::Seq).err(), Some(SvdError::NoConvergence));
		assert_eq!(
			a.singular_values(Par::Seq).err(),
			Some(SvdError::NoConvergence)
		);
	}
}
