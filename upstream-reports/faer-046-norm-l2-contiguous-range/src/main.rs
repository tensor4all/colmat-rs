// The 2-norm of a contiguous vector whose norm is finite and representable.
// The three-band accumulator scales the entries by sqrt(min_positive) and
// sqrt(max_positive) before squaring them. A single entry above about
// 0.5 * MAX overflows the first band's square, and an entry below about
// 2e-316 underflows the second band's square, so the reduction returns inf or 0.
//
// faer-008 covers the scipy fallback for a strided view. This program uses an
// owned column, which takes the contiguous SIMD path.
use faer::Mat;

fn main() {
	let mut bad = false;

	let cases = [
		(1usize, 0.5 * f64::MAX),
		(1, 0.75 * f64::MAX),
		(16, 1e-300),
		(16, 5e-320),
		(1, 5e-320),
	];

	for (len, value) in cases {
		let x = Mat::<f64>::from_fn(len, 1, |_, _| value);
		let expected = (len as f64).sqrt() * value;
		let actual = x.norm_l2();
		let ok =
			actual.is_finite() && (actual - expected).abs() <= 1e-12 * expected;
		println!("{len} x {value:e}: expected {expected:e}, actual {actual:e}");
		if !ok {
			bad = true;
		}
	}

	if bad {
		println!();
		println!("WRONG: a vector with a finite norm came back as inf or 0");
		std::process::exit(1);
	}
}
