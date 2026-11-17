//! #209 / #218: the values-only `svd` of `[[small, 0.25], [0, 1]]` must keep the `1/hypot(1, 0.25)`
//! factor on its small singular value.
//!
//! `dqds` squares the normalized bidiagonal factors, so a normalized diagonal below
//! `sqrt(min_positive)` squares to a subnormal and the value released from it does not keep its
//! relative accuracy. The kernel reports non-convergence for those inputs (#209) and the values-only
//! `svd` finishes on its rotation path (#110). That path used to deflate `|diag[i]| <= eps` before
//! its exact 2x2 branch, which dropped the factor and returned `Ok` with 3.08 % relative error at
//! both precisions — worse than the 2.09 % (f32) / 0.13 % (f64) the dqds value had. The exact
//! branch now runs first, and the small value comes back within an epsilon.
#![cfg(feature = "std")]
use colmat::diag::Diag;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::svd::{ComputeSvdVectors, SvdParams, svd, svd_scratch};
use colmat::{Auto, Mat, Par, ParShape};

/// Values-only `svd` of the upper bidiagonal `[[a00, e], [0, a11]]` as a dense 2x2 matrix.
macro_rules! bidiag_values {
	($t:ty, $a00:expr, $e:expr, $a11:expr) => {{
		let a = Mat::<$t>::from_fn(2, 2, |i, j| match (i, j) {
			(0, 0) => $a00,
			(0, 1) => $e,
			(1, 1) => $a11,
			_ => 0.0 as $t,
		});
		let params = <SvdParams as Auto<$t>>::auto().into();
		let mut mem = MemBuffer::new(svd_scratch::<$t>(
			2,
			2,
			ComputeSvdVectors::No,
			ComputeSvdVectors::No,
			ParShape::Seq,
			params,
		));
		let mut s = Diag::<$t>::zeros(2);
		svd(
			a.as_ref(),
			s.as_mut(),
			None,
			None,
			Par::Seq,
			MemStack::new(&mut mem),
			params,
		)
		.unwrap();
		let s = s.column_vector();
		(s[0], s[1])
	}};
}

/// `|got / want - 1|` in f64, or `got == 0` when `want` is zero.
fn within(got: f64, want: f64, tol: f64) -> bool {
	if want == 0.0 {
		got == 0.0
	} else {
		(got / want - 1.0).abs() <= tol
	}
}

macro_rules! check_bidiagonal {
	($t:ty, $a00:expr, $e:expr, $a11:expr, $want_large:expr, $want_small:expr) => {{
		let eps = <$t>::EPSILON as f64;
		let (large, small) = bidiag_values!($t, $a00, $e, $a11);
		let (large, small) = (large as f64, small as f64);
		assert!(
			large >= small && small >= 0.0,
			"not nonnegative descending: [{large:e}, {small:e}]"
		);
		assert!(
			within(large, $want_large, 20.0 * eps),
			"large value {large:e} vs {:e}",
			$want_large
		);
		assert!(
			within(small, $want_small, 20.0 * eps),
			"small value {small:e} vs {:e}, {} eps",
			$want_small,
			(small / $want_small - 1.0).abs() / eps
		);
	}};
}

/// `hypot(1, 0.25)`: the factor the small value is divided by.
const W: f64 = 1.030_776_406_404_415_1;

#[test]
fn the_mwe_small_value_keeps_the_hypot_factor_f32() {
	let small = 1e-22f32;
	check_bidiagonal!(f32, small, 0.25, 1.0, W, small as f64 / W);
	// controls: `1e-10` squares to a normal factor and never reaches the guard; `1e-30` squares
	// below `min_positive`, so it takes the same fallback as the MWE at a different scale.
	check_bidiagonal!(f32, 1e-10f32, 0.25, 1.0, W, 1e-10 / W);
	check_bidiagonal!(f32, 1e-30f32, 0.25, 1.0, W, 1e-30 / W);
}

#[test]
fn the_mwe_small_value_keeps_the_hypot_factor_f64() {
	let small = 1e-161f64;
	check_bidiagonal!(f64, small, 0.25, 1.0, W, small / W);
	check_bidiagonal!(f64, 1e-100f64, 0.25, 1.0, W, 1e-100 / W);
	check_bidiagonal!(f64, 1e-300f64, 0.25, 1.0, W, 1e-300 / W);
}

/// The exact 2x2 branch now absorbs the cases the `|diag[i]| <= eps` deflation used to take: a
/// negligible value in either position, either sign, and an exact zero. It must stay accurate,
/// nonnegative and ordered (the pass at the end of `qr_algorithm` fixes the sign and the order).
#[test]
fn the_exact_branch_covers_negligible_and_zero_diagonals() {
	let small = 1e-22f64;
	check_bidiagonal!(f32, 1.0f32, 0.25, 1e-22f32, W, 1e-22 / W);
	check_bidiagonal!(f32, -1e-22f32, 0.25, 1.0f32, W, small / W);
	check_bidiagonal!(f32, 1e-22f32, 0.25, -1.0f32, W, small / W);
	check_bidiagonal!(f64, 1.0f64, 0.25, 1e-161f64, W, 1e-161 / W);
	check_bidiagonal!(f64, -1e-161f64, 0.25, 1.0f64, W, 1e-161 / W);
	// exact zero: one singular value is zero, the other is `hypot(e, a11)`
	check_bidiagonal!(f32, 0.0f32, 0.25, 1.0f32, W, 0.0);
	check_bidiagonal!(f32, 0.0f32, 0.25, 0.0f32, 0.25, 0.0);
	check_bidiagonal!(f64, 0.0f64, 0.25, 0.0f64, 0.25, 0.0);
}
