// Self-adjoint eigenvalues of s * (J - I), the 3 x 3 symmetric matrix with zero
// diagonal and s in every off-diagonal entry. Its exact spectrum is
// s * [-1, -1, 2], which is representable for both s below: 2e-308 is just
// under f64::MIN_POSITIVE and 1e-310 is subnormal.
//
// The dense reduction flushes an off-diagonal entry below `min_positive` to
// zero before the tridiagonal solver, which has its own range normalization,
// sees it. The call returns Ok with an all-zero spectrum.
//
// The diagonal control is the same input with the off-diagonal entries removed:
// the values are representable and the solver returns them exactly, so the loss
// is in the dense reduction rather than in the eigenvalues' storage.
use faer::{Mat, Side};

fn main() {
	let mut bad = false;

	for s in [1.0f64, 2e-308, 1e-310] {
		let a = Mat::<f64>::from_fn(3, 3, |i, j| if i == j { 0.0 } else { s });
		let expected = [-s, -s, 2.0 * s];
		match a.self_adjoint_eigenvalues(Side::Lower) {
			Ok(got) => {
				println!(
					"dense, s = {s:e}: expected {expected:?}, actual {got:?}"
				);
				let ok = got.len() == expected.len()
					&& got
						.iter()
						.zip(expected)
						.all(|(g, e)| (*g - e).abs() <= 0.125 * e.abs());
				if !ok {
					bad = true;
				}
			},
			Err(err) => {
				println!(
					"dense, s = {s:e}: expected {expected:?}, actual Err({err:?})"
				);
				bad = true;
			},
		}
	}

	let s = 2e-308f64;
	let diag = Mat::<f64>::from_fn(3, 3, |i, j| {
		if i != j {
			0.0
		} else if i == 2 {
			2.0 * s
		} else {
			-s
		}
	});
	let expected = [-s, -s, 2.0 * s];
	match diag.self_adjoint_eigenvalues(Side::Lower) {
		Ok(got) => {
			println!(
				"diagonal control, s = {s:e}: expected {expected:?}, actual {got:?}"
			);
			let ok = got.len() == expected.len()
				&& got
					.iter()
					.zip(expected)
					.all(|(g, e)| (*g - e).abs() <= 0.125 * e.abs());
			if !ok {
				bad = true;
			}
		},
		Err(err) => {
			println!(
				"diagonal control, s = {s:e}: expected {expected:?}, actual Err({err:?})"
			);
			bad = true;
		},
	}

	if bad {
		println!();
		println!(
			"WRONG: a nonzero subnormal spectrum came back as zeros with Ok"
		);
		std::process::exit(1);
	}
}
