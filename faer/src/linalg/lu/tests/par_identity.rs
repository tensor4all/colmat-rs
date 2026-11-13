// Bit identity of the parallel partial-pivoting LU recursion against its sequential path
// (tensor4all/faer#123). The residual family in `lapack_lu.rs` measures against the
// factorization's own permutation, so it cannot see a changed pivot sequence or a column range
// updated twice; this compares the permutation and every factor bit.

/// Exact bits of one entry, real or complex, as `(re, im)`.
trait LookaheadBits {
	fn lookahead_bits(&self) -> (u64, u64);
}
impl LookaheadBits for f64 {
	fn lookahead_bits(&self) -> (u64, u64) {
		(self.to_bits(), 0)
	}
}
impl LookaheadBits for f32 {
	fn lookahead_bits(&self) -> (u64, u64) {
		(self.to_bits() as u64, 0)
	}
}
impl LookaheadBits for crate::c64 {
	fn lookahead_bits(&self) -> (u64, u64) {
		(self.re.to_bits(), self.im.to_bits())
	}
}
impl LookaheadBits for crate::c32 {
	fn lookahead_bits(&self) -> (u64, u64) {
		(self.re.to_bits() as u64, self.im.to_bits() as u64)
	}
}

/// A deterministic operand in `[-0.5, 0.5)`: the pivots are distinct and nothing is degenerate.
fn lookahead_fixture(n: usize) -> Mat<Scalar> {
	let mut seed = 0x2545f4914f6cdd1du64;
	Mat::from_fn(n, n, |_, _| {
		seed = seed
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		let v = ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
		scalar(v as Real, (v * 0.5) as Real)
	})
}

/// Factors `a` in place and returns the factor's entry bits and the permutation.
fn lookahead_factor(
	a: Mat<Scalar>,
	par: Par,
	never: bool,
) -> (alloc::vec::Vec<(u64, u64)>, alloc::vec::Vec<usize>) {
	let (m, n) = a.shape();
	use crate::utils::cost::ParThreshold;
	let params = Spec::new(pp::factor::PartialPivLuParams {
		par_threshold: if never {
			ParThreshold::Never
		} else {
			ParThreshold::Derived
		},
		..auto!(Scalar)
	});
	let req = pp::factor::lu_in_place_scratch::<usize, Scalar>(m, n, par.shape(), params);
	let mut mem = MemBuffer::new(req);
	let mut a = a;
	let mut p = alloc::vec![0usize; m];
	let mut ip = alloc::vec![0usize; m];
	pp::factor::lu_in_place(
		a.as_mut(),
		&mut p,
		&mut ip,
		par,
		MemStack::new(&mut mem),
		params,
	);
	let bits = (0..m)
		.flat_map(|i| (0..n).map(move |j| (i, j)))
		.map(|(i, j)| a[(i, j)].lookahead_bits())
		.collect();
	(bits, p)
}

/// The parallel paths of the recursion -- the per-chunk swap and solve, the trailing product and
/// the swap stage -- must not change a single bit of the factorization: partial-pivoting LU is
/// unique for its permutation, so any difference is a different computation, not a different
/// rounding. This is what the residual family above cannot check (its reference is built from the
/// factorization's own permutation).
#[test]
fn parallel_path_is_bit_identical_to_sequential() {
	let shapes = [
		9usize, 10, 11, 12, 13, 15, 16, 17, 23, 24, 25, 31, 32, 33, 35, 40, 41, 47, 48, 49, 64,
		72, 100, 129, 256, 1024,
	];
	for &n in &shapes {
		let a = lookahead_fixture(n);
		let (bits_ref, p_ref) = lookahead_factor(a.clone(), Par::Seq, true);
		for width in [1usize, 2, 4, 8] {
			crate::team::with_team(width, |par| {
				let (bits, p) = lookahead_factor(a.clone(), par, false);
				core::assert_eq!(p, p_ref, "n={n} width={width}: permutation");
				core::assert_eq!(bits, bits_ref, "n={n} width={width}: L/U bits");
			});
		}
	}
}
