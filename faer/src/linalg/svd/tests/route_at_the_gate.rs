// The `recursion_threshold` entry gate chooses the *route* the bidiagonal SVD
// takes - a `qr_algorithm` over the whole bidiagonal below it,
// `divide_and_conquer` above it - and the default sits where the route changes,
// so the sizes on both sides answer the same question twice by different means
// and must agree. The gate is compared in two places that have to stay the same
// comparison (`bidiag_real_svd_scratch` reserves the divide-and-conquer scratch
// from one, `compute_bidiag_real_svd` chooses the route from the other), so the
// sizes are the gate and both of its neighbours, where a `<=` that moves in
// only one of them is visible, then the round sizes further from it.
//
// The default is read from `Auto`, so the test does not move when the value
// does; `usize::MAX` forces the QR route at every size, as `tests/qr_stall.rs`
// runs it, and `4` forces the recursion, as `tests/lapack.rs`'s `(8, 4)` does.
use super::super::super::*;
use super::{Real, Scalar};
use crate::assert;
use dyn_stack::MemBuffer;

fn seeded(n: usize) -> Mat<Scalar> {
	let mut a = Mat::<Scalar>::zeros(n, n);
	let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (n as u64);
	for j in 0..n {
		for i in 0..n {
			state = state
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			let unit = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
			a[(i, j)] = Scalar::from(unit as Real);
		}
	}
	a
}

/// (singular values, the reconstruction residual, the `U` orthogonality
/// residual, the `V` orthogonality residual)
fn thin_svd_at(n: usize, threshold: usize) -> (Vec<Real>, Real, Real, Real) {
	let a = seeded(n);
	#[allow(clippy::field_reassign_with_default)] // `Spec` reaches these fields through `DerefMut`, so struct-update syntax cannot set them
	let mut params = Spec::<SvdParams, Scalar>::default();
	params.config.recursion_threshold = threshold;
	let mut s = Col::<Scalar>::zeros(n);
	let mut u = Mat::<Scalar>::zeros(n, n);
	let mut v = Mat::<Scalar>::zeros(n, n);
	let mut mem = MemBuffer::new(svd_scratch::<Scalar>(
		n,
		n,
		ComputeSvdVectors::Thin,
		ComputeSvdVectors::Thin,
		ParShape::Seq,
		params,
	));
	svd(
		a.as_ref(),
		s.as_mut().as_diagonal_mut(),
		Some(u.as_mut()),
		Some(v.as_mut()),
		Par::Seq,
		MemStack::new(&mut mem),
		params,
	)
	.unwrap();
	let back = &u * s.as_diagonal() * v.adjoint();
	let scale = a.norm_l2();
	let u_res = (u.adjoint() * &u - Mat::<Scalar>::identity(n, n)).norm_max();
	let v_res = (v.adjoint() * &v - Mat::<Scalar>::identity(n, n)).norm_max();
	(
		(0..n).map(|i| real(&s[i])).collect(),
		(&back - &a).norm_l2() / scale,
		u_res,
		v_res,
	)
}

#[test]
fn both_routes_agree_on_both_sides_of_the_default_gate() {
	let gate = <SvdParams as Auto<Scalar>>::auto().recursion_threshold;
	assert!(
		gate > 4,
		"the gate has to leave both routes reachable below it"
	);
	for n in [gate - 1, gate, gate + 1, 32, 64, 96] {
		if n < 8 {
			continue;
		}
		let (s_gate, rec_gate, u_gate, v_gate) = thin_svd_at(n, gate);
		let (s_qr, rec_qr, u_qr, v_qr) = thin_svd_at(n, usize::MAX);
		let (s_dc, rec_dc, u_dc, v_dc) = thin_svd_at(n, 4);
		let tol = 256.0 * (n as Real) * eps::<Real>();
		let s_max = s_gate[0];
		assert!(s_max > 0.0, "n = {n}: a zero spectrum");
		for i in 0..n {
			let context = format!("{n}, {i}, gate {gate}");
			assert!(
				s_gate[i] >= 0.0 && (i == 0 || s_gate[i - 1] >= s_gate[i]),
				"{context}: the default route is not a nonincreasing \
				 nonnegative spectrum"
			);
			// the spectrum is the answer, not the route: all three agree to n *
			// eps * sigma_max
			for (name, other) in [("QR", &s_qr), ("D&C", &s_dc)] {
				let diff = (s_gate[i] - other[i]).abs();
				assert!(
					diff <= tol * s_max,
					"{context}: the default route and {name} disagree on \
					 sigma[{i}]: {} against {}",
					s_gate[i],
					other[i]
				);
			}
		}
		for (name, rec, ure, vre) in [
			("default", rec_gate, u_gate, v_gate),
			("QR", rec_qr, u_qr, v_qr),
			("D&C", rec_dc, u_dc, v_dc),
		] {
			let context = format!("{n}, gate {gate}, route {name}");
			assert!(rec <= tol, "{context}: reconstruction residual {rec:e}");
			assert!(ure <= tol, "{context}: U orthogonality residual {ure:e}");
			assert!(vre <= tol, "{context}: V orthogonality residual {vre:e}");
		}
	}
}
