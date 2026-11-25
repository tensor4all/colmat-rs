//! The negative control of colmat-rs#297: random shapes with every dimension in
//! 1..=40, the product of `f64` and `c64` matrices against a naive sum, and the
//! full-pivoting LU solve of a random square system. It passes with and without
//! `x86-v4`; the proposed CI `x86-v4` lane (#298) runs it next to the regression tests of the
//! defects that kept `x86-v4` out of the defaults (tensor4all/faer#102), so a failure
//! there separates "the AVX-512 path is wrong everywhere" from "one algorithm
//! is wrong on it".
use colmat::prelude::*;
use colmat::{Mat, Par, c64};

/// A fixed-seed LCG, so a failure names the same shape on every run.
struct Lcg(u64);

impl Lcg {
	/// Uniform in [-1, 1).
	fn next(&mut self) -> f64 {
		self.0 = self
			.0
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
	}

	/// Uniform in 1..=40.
	fn size(&mut self) -> usize {
		1 + ((self.next() + 1.0) * 20.0) as usize % 40
	}
}

#[test]
fn random_sizes_1_to_40_matmul_and_full_piv_lu() {
	let mut r = Lcg(0x297);
	for _ in 0..300 {
		let (m, n, k) = (r.size(), r.size(), r.size());

		let a = Mat::<f64>::from_fn(m, k, |_, _| r.next());
		let b = Mat::<f64>::from_fn(k, n, |_, _| r.next());
		let c = &a * &b;
		for i in 0..m {
			for j in 0..n {
				let e: f64 = (0..k).map(|p| a[(i, p)] * b[(p, j)]).sum();
				assert!(
					(c[(i, j)] - e).abs() <= 1e-10 * k as f64,
					"f64 matmul {m}x{k}x{n}: entry ({i}, {j})"
				);
			}
		}

		let az = Mat::<c64>::from_fn(m, k, |_, _| c64::new(r.next(), r.next()));
		let bz = Mat::<c64>::from_fn(k, n, |_, _| c64::new(r.next(), r.next()));
		let cz = &az * &bz;
		for i in 0..m {
			for j in 0..n {
				let mut e = c64::new(0.0, 0.0);
				for p in 0..k {
					e += az[(i, p)] * bz[(p, j)];
				}
				assert!(
					(cz[(i, j)] - e).norm() <= 1e-10 * k as f64,
					"c64 matmul {m}x{k}x{n}: entry ({i}, {j})"
				);
			}
		}

		let s = Mat::<f64>::from_fn(n, n, |_, _| r.next());
		let rhs = Mat::<f64>::from_fn(n, 1, |_, _| r.next());
		let x = s.full_piv_lu(Par::Seq).solve(&rhs, Par::Seq);
		assert!(
			x.is_all_finite(),
			"full_piv_lu {n}x{n}: non-finite solution"
		);
	}
}

/// Not a check: prints the instruction set pulp dispatches colmat's own SIMD
/// loops to, so a CI log says whether the run exercised AVX-512 (`V4`) or fell
/// back to AVX2 (`V3`) on its runner.
#[cfg(target_arch = "x86_64")]
#[test]
fn report_the_dispatched_instruction_set() {
	let name = match colmat::traits::pulp::Arch::new() {
		#[cfg(feature = "x86-v4")]
		colmat::traits::pulp::Arch::V4(_) => "V4 (AVX-512)",
		colmat::traits::pulp::Arch::V3(_) => "V3 (AVX2)",
		_ => "scalar",
	};
	println!("pulp dispatches to {name}");
}
