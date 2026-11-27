//! The AVX2 route is an overwrite-only product. Scaling or adding must use the general route.
use colmat::linalg::matmul::matmul;
use colmat::{Accum, MatMut, MatRef, Par};

#[test]
fn compact_square_respects_scale_and_accumulation() {
	let n = 513;
	let input: Vec<f64> = (0..n * n).map(|i| (i % 113) as f64 - 57.0).collect();
	let mut diagonal = vec![0.0; n * n];
	for i in 0..n {
		diagonal[i * (n + 1)] = (i % 7) as f64 - 3.0;
	}
	for (accum, scale, initial) in [
		(Accum::Replace, 1.0, f64::NAN),
		(Accum::Add, 1.0, 13.0),
		(Accum::Replace, -2.0, 13.0),
	] {
		let mut output = vec![initial; n * n];
		matmul(
			MatMut::from_column_major_slice_mut(&mut output, n, n),
			accum,
			MatRef::from_column_major_slice(&input, n, n),
			MatRef::from_column_major_slice(&diagonal, n, n),
			scale,
			Par::Seq,
		);
		for (i, &actual) in output.iter().enumerate() {
			let old = if accum == Accum::Add { initial } else { 0.0 };
			assert_eq!(
				actual,
				old + scale * input[i] * diagonal[(i / n) * (n + 1)]
			);
		}
	}
}
