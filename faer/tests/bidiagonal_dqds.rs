//! Public dqds contract for the LASQ1 adapter in AtelierArith/Colmat.jl#306.
//! Fixtures and analytic 2x2 checks are original; the existing native full
//! bidiagonal SVD supplies an independent rotation-based comparison.
mod common;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::svd::{
	SvdError, bidiagonal_dqds, bidiagonal_dqds_scratch, bidiagonal_svd,
	bidiagonal_svd_scratch,
};
use colmat::{Mat, Par, ParShape};
macro_rules! contracts {
	($name:ident,$t:ty,$tol:expr) => {
		#[test]
		fn $name() {
			for n in [0usize, 1, 2, 3, 7] {
				let input: Vec<$t> = (0..n)
					.map(|i| {
						if i % 2 == 0 {
							-(i as $t + 1.0)
						} else {
							i as $t + 1.0
						}
					})
					.collect();
				let e = vec![0.25 as $t; n.saturating_sub(1)];
				let mut d = input.clone();
				let mut memory =
					MemBuffer::new(bidiagonal_dqds_scratch::<$t>(n));
				bidiagonal_dqds(&mut d, &e, MemStack::new(&mut memory))
					.unwrap();
				let mut expected = input.clone();
				let mut u = Mat::<$t>::zeros(n, n);
				let mut v = Mat::<$t>::zeros(n, n);
				let mut mem = MemBuffer::new(bidiagonal_svd_scratch::<$t>(
					n,
					ParShape::Seq,
					Default::default(),
				));
				bidiagonal_svd(
					&mut expected,
					&e,
					true,
					u.as_mut(),
					v.as_mut(),
					Par::Seq,
					MemStack::new(&mut mem),
					Default::default(),
				)
				.unwrap();
				for (actual, expected) in d.iter().zip(&expected) {
					assert!(
						(actual - expected).abs()
							< $tol * (1.0 + expected.abs())
					);
				}
				assert!(d.windows(2).all(|x| x[0] >= x[1]));
				assert!(d.iter().all(|x| *x >= 0.0));
			}
			let mut d = [3.0 as $t, 4.0];
			let mut memory = MemBuffer::new(bidiagonal_dqds_scratch::<$t>(2));
			bidiagonal_dqds(&mut d, &[1.0], MemStack::new(&mut memory))
				.unwrap();
			assert!((d[0] - (18.0 as $t).sqrt()).abs() < $tol);
			assert!((d[1] - (8.0 as $t).sqrt()).abs() < $tol);
			// Squared-factor dynamic range cannot represent this invertible matrix.
			// Error rollback must also hold after entering the private iteration.
			let mut graded = [<$t>::MIN_POSITIVE, <$t>::MAX / 4.0];
			let before = graded.map(|x| x.to_bits());
			let edge = [<$t>::MAX / 16.0];
			let mut memory = MemBuffer::new(bidiagonal_dqds_scratch::<$t>(2));
			assert!(matches!(
				bidiagonal_dqds(&mut graded, &edge, MemStack::new(&mut memory)),
				Err(SvdError::NoConvergence)
			));
			assert_eq!(graded.map(|x| x.to_bits()), before);
			// Singleton subnormals use no scratch and do not square or reciprocate.
			let tiny = <$t>::from_bits(1);
			let mut d = [-tiny];
			let mut memory = MemBuffer::new(bidiagonal_dqds_scratch::<$t>(1));
			bidiagonal_dqds(&mut d, &[], MemStack::new(&mut memory)).unwrap();
			assert_eq!(d, [tiny]);
			for input in
				[[0.0 as $t, 2.0], [<$t>::NAN, 2.0], [<$t>::INFINITY, 2.0]]
			{
				let mut d = input;
				let e = [0.25 as $t];
				let before = d.map(|x| x.to_bits());
				let mut memory =
					MemBuffer::new(bidiagonal_dqds_scratch::<$t>(2));
				assert!(matches!(
					bidiagonal_dqds(&mut d, &e, MemStack::new(&mut memory)),
					Err(SvdError::NoConvergence)
				));
				assert_eq!(d.map(|x| x.to_bits()), before);
				assert_eq!(e, [0.25]);
			}
		}
	};
}
contracts!(dqds_real32, f32, 2e-4);
contracts!(dqds_real64, f64, 1e-11);
