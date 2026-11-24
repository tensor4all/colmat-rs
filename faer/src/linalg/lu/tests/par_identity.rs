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
	use crate::utils::cost::ParThreshold;
	lookahead_factor_with(
		a,
		par,
		if never {
			ParThreshold::Never
		} else {
			ParThreshold::Derived
		},
	)
}

/// [`lookahead_factor`] with an explicit floor for the parallel regions.
fn lookahead_factor_with(
	a: Mat<Scalar>,
	par: Par,
	par_threshold: crate::utils::cost::ParThreshold,
) -> (alloc::vec::Vec<(u64, u64)>, alloc::vec::Vec<usize>) {
	let (m, n) = a.shape();
	let params = Spec::new(pp::factor::PartialPivLuParams {
		par_threshold,
		..auto!(Scalar)
	});
	let req = pp::factor::lu_in_place_scratch::<usize, Scalar>(
		m,
		n,
		par.shape(),
		params,
	);
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
		9usize, 10, 11, 12, 13, 15, 16, 17, 23, 24, 25, 31, 32, 33, 35, 40, 41,
		47, 48, 49, 64, 72, 100, 129, 256, 1024,
	];
	for &n in &shapes {
		let a = lookahead_fixture(n);
		let (bits_ref, p_ref) = lookahead_factor(a.clone(), Par::Seq, true);
		for width in [1usize, 2, 4, 8] {
			crate::team::with_team(width, |par| {
				let (bits, p) = lookahead_factor(a.clone(), par, false);
				core::assert_eq!(p, p_ref, "n={n} width={width}: permutation");
				core::assert_eq!(
					bits,
					bits_ref,
					"n={n} width={width}: L/U bits"
				);
			});
		}
	}
}

/// Every region on the whole team (`ParThreshold::Fixed(0)`), at widths whose column chunks would
/// be a single column (tensor4all/colmat-rs#191): the complex products round a one-column product
/// differently from the same column inside a wider one, so a region split into one-column chunks
/// changed bits at 8 participants with an 8-wide `c64` leaf, and at 16 with the 16-wide `f32`/`c32`
/// leaf. The shapes keep the test small; the floor makes their narrow levels split anyway.
#[test]
fn parallel_path_is_bit_identical_with_one_column_per_member() {
	use crate::utils::cost::ParThreshold;
	for &n in &[17usize, 33, 64, 100] {
		let a = lookahead_fixture(n);
		let (bits_ref, p_ref) = lookahead_factor(a.clone(), Par::Seq, true);
		for width in [3usize, 8, 16] {
			crate::team::with_team(width, |par| {
				let (bits, p) = lookahead_factor_with(
					a.clone(),
					par,
					ParThreshold::Fixed(0),
				);
				core::assert_eq!(p, p_ref, "n={n} width={width}: permutation");
				core::assert_eq!(
					bits,
					bits_ref,
					"n={n} width={width}: L/U bits"
				);
			});
		}
	}
}

/// One factorization with the look-ahead switched on or off ([`pp::factor::LookaheadProbe`]),
/// returning the packed `L \ U` factor (column-major copy), the permutation, the transposition
/// count and the number of levels of *this call* that looked ahead. The counter lives in this
/// call's params, so a test running beside it on another thread cannot bump it.
fn lookahead_run(
	a: MatMut<'_, Scalar>,
	par: Par,
	recursion_threshold: Option<usize>,
	lookahead: bool,
) -> (Mat<Scalar>, alloc::vec::Vec<usize>, usize, usize) {
	use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
	let (m, n) = a.shape();
	// one counter per call; leaked so that the `Copy` params can carry it (a few bytes per call)
	let ran: &'static AtomicUsize =
		alloc::boxed::Box::leak(alloc::boxed::Box::new(AtomicUsize::new(0)));
	let mut config = pp::factor::PartialPivLuParams {
		par_threshold: crate::utils::cost::ParThreshold::Fixed(0),
		..auto!(Scalar)
	};
	if let Some(t) = recursion_threshold {
		config.recursion_threshold = t;
	}
	config.lookahead_probe = pp::factor::LookaheadProbe {
		off: !lookahead,
		ran: Some(ran),
	};
	let params = Spec::new(config);
	let req = pp::factor::lu_in_place_scratch::<usize, Scalar>(
		m,
		n,
		par.shape(),
		params,
	);
	let mut mem = MemBuffer::new(req);
	let (mut p, mut ip) = (alloc::vec![0usize; m], alloc::vec![0usize; m]);
	let mut a = a;
	let (info, _) = pp::factor::lu_in_place(
		a.rb_mut(),
		&mut p,
		&mut ip,
		par,
		MemStack::new(&mut mem),
		params,
	);
	(a.rb().to_owned(), p, info.transposition_count, ran.load(Relaxed))
}

/// Bound on how far the look-ahead-on factor may be from the look-ahead-off one: the LAPACK test
/// ratio of this suite, `difference_ratio(on, off, eps) = ||on - off||_1 / (||off||_1 *
/// max(m, n) * eps)` on the packed `L \ U` factor, below LAPACK's usual threshold of 30. That is
/// `||on - off||_1 <= 30 * max(m, n) * eps * ||off||_1`.
const LOOKAHEAD_RATIO_BOUND: f64 = 30.0;

/// Asserts that look-ahead-on and look-ahead-off agree: the same permutation and transposition
/// count, and factors within [`LOOKAHEAD_RATIO_BOUND`].
fn assert_lookahead_agrees(
	on: &(Mat<Scalar>, alloc::vec::Vec<usize>, usize, usize),
	off: &(Mat<Scalar>, alloc::vec::Vec<usize>, usize, usize),
	tag: &str,
) {
	core::assert_eq!(on.1, off.1, "{tag}: permutation");
	core::assert_eq!(on.2, off.2, "{tag}: transposition count");
	let ratio = difference_ratio(on.0.as_ref(), off.0.as_ref(), Real::EPSILON);
	core::assert!(
		(ratio as f64) < LOOKAHEAD_RATIO_BOUND,
		"{tag}: L/U differ by ratio {ratio} >= {LOOKAHEAD_RATIO_BOUND}"
	);
}

/// The look-ahead (tensor4all/colmat-rs#239): a level updates its right child's first panel, the
/// leader factors that panel alone while the other members update the rest, and the right child
/// then skips the panel. The claim is that this changes the time, not the factorization: the same
/// call at the same team width with the look-ahead switched off
/// ([`pp::factor::LookaheadProbe::off`]) gives the same permutation and transposition count, and a
/// factor within [`LOOKAHEAD_RATIO_BOUND`] of it. Exact bit identity is **not** asserted.
///
/// Documented exception (do not re-open without new evidence): on aarch64 (Apple M-series,
/// tensor4all/colmat-rs#244), the `f32`/`f64` 257 x 257 column-major factor at team width 4
/// differs in its bits between look-ahead on and off, deterministically over three reruns, while
/// every x86_64 run (evo, the ubuntu CI lanes) was bit-identical at every case here. The
/// permutation agreed there. The cause is host-dependent summation order in the panel path: with
/// the look-ahead a level's trailing product is two calls (the panel's columns on the team, the
/// rest on the other members) instead of one, and the host's product kernels may block and sum the
/// two differently from the single call. Bit identity across that split is not a property the
/// product promises, so the feature does not promise it either.
///
/// Every region on the whole team (`ParThreshold::Fixed(0)`) so the levels look ahead on any host,
/// at shapes whose right children reach the panel through halves that do not halve evenly (a
/// 2^k + 1 tail, a short right child), and at the benchmarked 8T n = 1024. There the top level's
/// 64-column panel beside a 448-column far product sits exactly on the gate (`2 P p (p - 1) = F`,
/// not `>`) and does not look ahead; the 512-column levels below do (panel 64, far 192), so the
/// path runs at the shape the timings were taken at. Column-major and row-major (whose leaves share
/// one buffer with the panel), square, tall and wide. At 4 and 8 members the look-ahead must have
/// run in this call; at 3 it must not (the `LOOKAHEAD_MIN_WIDTH` gate), and width 3 asserts only
/// that: both runs there execute the pre-#239 code, so comparing them would test whether the team
/// path reproduces itself, which is not this change's claim (below).
///
/// This test does not compare against `Par::Seq` either. On aarch64 the `f32`/`f64` 200 x 200
/// column-major factor at width 3 -- where the look-ahead did not run (per-call count 0) --
/// differed from the sequential one in some runs and matched it in others: the team product's
/// summation order there depends on scheduling. That is pre-existing (#239 does not touch that
/// branch); the narrower team-vs-sequential checks above
/// ([`parallel_path_is_bit_identical_to_sequential`],
/// [`parallel_path_is_bit_identical_with_one_column_per_member`]) are main's and unchanged.
#[test]
fn lookahead_does_not_change_the_factorization() {
	for &(m, n) in &[
		(200usize, 200usize),
		(257, 257),
		(300, 211),
		(211, 300),
		(1024, 1024),
	] {
		let col_major =
			lookahead_fixture(Ord::max(m, n)).get(..m, ..n).to_owned();
		// the same values, stored row-major (`row_major.transpose()` is the `m x n` input)
		let row_major = col_major.transpose().to_owned();
		for is_row_major in [false, true] {
			let run = |par: Par, lookahead: bool| {
				if is_row_major {
					let mut t = row_major.clone();
					lookahead_run(
						t.as_mut().transpose_mut(),
						par,
						None,
						lookahead,
					)
				} else {
					let mut a = col_major.clone();
					lookahead_run(a.as_mut(), par, None, lookahead)
				}
			};
			for width in [3usize, 4, 8] {
				crate::team::with_team(width, |mut par| {
					let off = run(par.reborrow(), false);
					let on = run(par, true);
					let tag = alloc::format!(
						"{m}x{n} row_major={is_row_major} width={width}"
					);
					core::assert_eq!(
						off.3,
						0,
						"{tag}: the switch did not turn the look-ahead off"
					);
					if width >= 4 {
						assert_lookahead_agrees(&on, &off, &tag);
						core::assert!(
							on.3 > 0,
							"{tag}: the look-ahead did not run in this call"
						);
					} else {
						core::assert_eq!(
							on.3,
							0,
							"{tag}: the look-ahead ran below its minimum width"
						);
					}
				});
			}
		}
	}
}

/// A caller-raised `recursion_threshold` makes the look-ahead panel a leaf wider than
/// `LOOKAHEAD_PANEL` (threshold 128: a 300-column right child gives an 80-column panel, see
/// `lookahead_panel`). The right child then skips that whole leaf; the factorization must agree
/// with the same call without the look-ahead as in
/// [`lookahead_does_not_change_the_factorization`], and the path must have run.
#[test]
fn lookahead_with_raised_threshold_does_not_change_the_factorization() {
	for &(m, n) in &[(600usize, 600usize), (700, 600)] {
		let a = lookahead_fixture(Ord::max(m, n)).get(..m, ..n).to_owned();
		crate::team::with_team(8, |mut par| {
			let off = lookahead_run(
				a.clone().as_mut(),
				par.reborrow(),
				Some(128),
				false,
			);
			let on = lookahead_run(a.clone().as_mut(), par, Some(128), true);
			let tag = alloc::format!("{m}x{n} threshold=128 width=8");
			core::assert_eq!(
				off.3,
				0,
				"{tag}: the switch did not turn the look-ahead off"
			);
			assert_lookahead_agrees(&on, &off, &tag);
			core::assert!(
				on.3 > 0,
				"{tag}: the look-ahead did not run in this call"
			);
		});
	}
}
