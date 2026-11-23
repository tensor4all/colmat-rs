//! Public contract of the polar decomposition (`linalg::polar`, colmat-rs#60): both methods on
//! square, tall and wide inputs, including rank-deficient, ill-conditioned and extreme-scale ones,
//! at the exact scratch size; views; `Par::Seq` against a two-thread team; and the reference
//! values of MatrixFactorizations.jl (`fixtures/polar_reference/`).
use colmat::Side;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::polar::{
	PolarError, PolarMethod, PolarParams, left_polar, left_polar_scratch,
	right_polar, right_polar_scratch,
};
use colmat::{Auto, Mat, Par, c32, c64};
mod common;
use common::*;

const METHODS: [PolarMethod; 2] = [PolarMethod::Svd, PolarMethod::Qdwh];

/// A scalar built from a real and an imaginary part (the imaginary part is dropped for a real
/// type), so that the complex cases are not real matrices in disguise.
trait Scalar: ComplexField {
	fn make(re: f64, im: f64) -> Self;
}
impl Scalar for f32 {
	fn make(re: f64, _: f64) -> Self {
		re as f32
	}
}
impl Scalar for f64 {
	fn make(re: f64, _: f64) -> Self {
		re
	}
}
impl Scalar for c32 {
	fn make(re: f64, im: f64) -> Self {
		c32::new(re as f32, im as f32)
	}
}
impl Scalar for c64 {
	fn make(re: f64, im: f64) -> Self {
		c64::new(re, im)
	}
}

/// Entries from the shared `sample` hash at the linear position: `sample(i, j)` itself gives
/// nearly singular matrices (condition 3e9 at 8 x 8, 5e9 at 33 x 33), which the `graded` cases
/// cover on purpose.
fn sample_mat<T: Scalar>(m: usize, n: usize, seed: usize) -> Mat<T> {
	Mat::from_fn(m, n, |i, j| {
		T::make(sample(i + m * j, 2 * seed), sample(i + m * j, 2 * seed + 1))
	})
}

fn params<T: ComplexField>(method: PolarMethod) -> PolarParams {
	let mut params = <PolarParams as Auto<T>>::auto();
	params.method = method;
	params
}

/// $A = W P$ for $m \ge n$, or $A = P W^H$ for $m < n$, as `(W or Wh, P)`, from outputs that
/// held garbage, at the exact scratch size.
fn polar<T: ComplexField>(
	a: MatRef<'_, T>,
	method: PolarMethod,
	par: Par<'_>,
) -> Result<(Mat<T>, Mat<T>), PolarError> {
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let params = params::<T>(method).into();
	let mut w = Mat::<T>::full(m, n, T::from_f64(-7.5));
	let mut p = Mat::<T>::full(k, k, T::from_f64(f64::NAN));
	if m >= n {
		let mut mem =
			MemBuffer::new(left_polar_scratch::<T>(m, n, par.shape(), params));
		left_polar(
			a,
			w.as_mut(),
			p.as_mut(),
			par,
			MemStack::new(&mut mem),
			params,
		)?;
	} else {
		let mut mem =
			MemBuffer::new(right_polar_scratch::<T>(m, n, par.shape(), params));
		right_polar(
			a,
			p.as_mut(),
			w.as_mut(),
			par,
			MemStack::new(&mut mem),
			params,
		)?;
	}
	Ok((w, p))
}

/// The defining properties: $A = W P$ (or $P W^H$), $P$ exactly hermitian and positive
/// semidefinite, and, when `isometry`, $W$ with orthonormal columns (rows); each within
/// `tolerance(max(m, n))`, relative to $\|A\|_F$ where it scales with $A$.
fn check<T: ComplexField>(
	a: MatRef<'_, T>,
	w: MatRef<'_, T>,
	p: MatRef<'_, T>,
	isometry: bool,
	tag: &str,
) {
	let (m, n) = a.shape();
	let tol = tolerance::<T>(Ord::max(m, n));
	let bound = tol.clone() * a.norm_l2();
	let rebuilt = if m >= n { w * p } else { p * w };
	let residual = (&rebuilt - a).norm_l2();
	assert!(
		residual <= bound,
		"{tag}: |A - WP| = {residual:?}, bound {bound:?}"
	);
	if isometry {
		let k = Ord::min(m, n);
		let gram = if m >= n {
			w.adjoint() * w
		} else {
			w * w.adjoint()
		};
		let defect = (&gram - Mat::<T>::identity(k, k)).norm_l2();
		assert!(defect <= tol, "{tag}: |W^H W - I| = {defect:?}");
	}
	for j in 0..p.ncols() {
		for i in 0..p.nrows() {
			assert!(p[(i, j)] == conj(&p[(j, i)]), "{tag}: P is not hermitian");
		}
	}
	if p.nrows() > 0 {
		let lambda = p.self_adjoint_eigenvalues(Side::Lower, Par::Seq).unwrap();
		let lambda_min = lambda[0].clone();
		assert!(
			lambda_min >= -bound.clone(),
			"{tag}: lambda_min(P) = {lambda_min:?}, bound {bound:?}"
		);
	}
}

/// `other` agrees with `reference`: $W$ within `tolerance * cond(A)` (the condition number of the
/// polar factor is $1 / \sigma_{\min}$ relative to $\|A\|$), $P$ within `tolerance * |A|`.
fn agree<T: ComplexField>(
	a: MatRef<'_, T>,
	reference: (&Mat<T>, &Mat<T>),
	other: (&Mat<T>, &Mat<T>),
	tag: &str,
) {
	let tol = tolerance::<T>(Ord::max(a.nrows(), a.ncols()));
	let sigma = a.singular_values(Par::Seq).unwrap();
	let cond = sigma[0].clone() / sigma[sigma.len() - 1].clone();
	let w_bound = tol.clone() * cond;
	let p_bound = tol * a.norm_l2();
	let w_diff = (other.0 - reference.0).norm_l2();
	let p_diff = (other.1 - reference.1).norm_l2();
	assert!(
		w_diff <= w_bound,
		"{tag}: |W - W_ref| = {w_diff:?}, bound {w_bound:?}"
	);
	assert!(
		p_diff <= p_bound,
		"{tag}: |P - P_ref| = {p_diff:?}, bound {p_bound:?}"
	);
}

/// Both methods on `a` and on its transpose (the other side), with the properties above and,
/// when `full_rank`, agreement between the methods.
fn contract<T: ComplexField>(a: MatRef<'_, T>, full_rank: bool, tag: &str) {
	for a in [a, a.transpose()] {
		let tag = format!(
			"{tag} {}x{} {}",
			a.nrows(),
			a.ncols(),
			core::any::type_name::<T>()
		);
		let svd = polar(a, PolarMethod::Svd, Par::Seq).unwrap();
		let qdwh = polar(a, PolarMethod::Qdwh, Par::Seq).unwrap();
		check(
			a,
			svd.0.as_ref(),
			svd.1.as_ref(),
			true,
			&format!("{tag} svd"),
		);
		check(
			a,
			qdwh.0.as_ref(),
			qdwh.1.as_ref(),
			full_rank,
			&format!("{tag} qdwh"),
		);
		if full_rank && Ord::min(a.nrows(), a.ncols()) > 0 {
			agree(a, (&svd.0, &svd.1), (&qdwh.0, &qdwh.1), &tag);
		}
	}
}

fn contracts<T: Scalar>() {
	for (m, n) in [
		(0, 0),
		(3, 0),
		(1, 1),
		(2, 2),
		(8, 8),
		(8, 3),
		(33, 33),
		(100, 33),
	] {
		contract(sample_mat::<T>(m, n, 0).as_ref(), true, "sample");
	}
	// A = 0: P = 0, and any isometry is a polar factor
	contract(Mat::<T>::zeros(5, 3).as_ref(), false, "zero");
	contract(Mat::<T>::zeros(4, 4).as_ref(), false, "zero");

	// rank deficient: a repeated column, and a zero column
	let mut repeated = sample_mat::<T>(9, 6, 1);
	let column = repeated.col(1).to_owned();
	repeated.col_mut(4).copy_from(&column);
	contract(repeated.as_ref(), false, "repeated column");
	let mut zero_column = sample_mat::<T>(7, 7, 2);
	zero_column.col_mut(3).fill(zero());
	contract(zero_column.as_ref(), false, "zero column");

	// graded columns, condition about eps^{-1/2} (moderate) and eps^{-4/5} (severe; the
	// forward comparison is vacuous there, the defining properties are not)
	let epsilon = (2.0f64).powi(1 - nbits::<T>() as i32);
	for power in [0.5, 0.8] {
		let mut graded = sample_mat::<T>(12, 7, 3);
		for j in 0..7 {
			let scale =
				from_f64::<T::Real>(epsilon.powf(power * j as f64 / 6.0));
			graded
				.col_mut(j)
				.iter_mut()
				.for_each(|x| *x = mul_real(x, &scale));
		}
		contract(graded.as_ref(), true, &format!("graded {power}"));
	}

	// extreme scales: entries `eps^-1` above the underflow threshold and near overflow; W is
	// scale-free and P scales with A
	let base = sample_mat::<T>(6, 4, 4);
	let scale_of = |a: &Mat<T>, scale: T::Real| {
		Mat::from_fn(a.nrows(), a.ncols(), |i, j| mul_real(&a[(i, j)], &scale))
	};
	for scale in [
		min_positive::<T::Real>() / eps::<T::Real>(),
		max_positive::<T::Real>() * from_f64::<T::Real>(1.0 / 64.0),
	] {
		let tag = format!("scale {scale:?}");
		contract(scale_of(&base, scale).as_ref(), true, &tag);
	}
	// at the underflow threshold itself (subnormal entries included) square QDWH only, which
	// calls neither the thin SVD nor a QR of A. Both are inaccurate there (relative residual
	// 0.79 and 1.0 on this `f64` 6 x 4 matrix; inherited, recorded in
	// docs/worklogs/2026-11-17-polar.md), and the SVD method and tall QDWH inherit that.
	let tiny = scale_of(&sample_mat::<T>(4, 4, 4), min_positive::<T::Real>());
	for a in [tiny.as_ref(), tiny.transpose()] {
		let (w, p) = polar(a, PolarMethod::Qdwh, Par::Seq).unwrap();
		check(a, w.as_ref(), p.as_ref(), true, "min_positive qdwh");
	}
}

/// Square QDWH on a matrix whose Frobenius norm is subnormal (the reciprocal of the scale
/// overflows), built exactly as $A = 2^{-1064} B$ with integer $B$: $W$ matches the one of $B$,
/// and $A = W P$ holds up to the rounding of subnormal results.
#[test]
fn polar_qdwh_subnormal_norm() {
	// diagonally dominant, so well conditioned
	let b = Mat::<f64>::from_fn(4, 4, |i, j| {
		if i == j {
			10.0
		} else {
			((3 * i + 5 * j) % 7) as f64 - 3.0
		}
	});
	let scale = 2.0f64.powi(-1064);
	let a = Mat::<f64>::from_fn(4, 4, |i, j| b[(i, j)] * scale);
	assert!(a.norm_l2() < f64::MIN_POSITIVE);
	for (a, b) in [(a.as_ref(), b.as_ref()), (a.transpose(), b.transpose())] {
		let (w_b, p_b) = polar(b, PolarMethod::Qdwh, Par::Seq).unwrap();
		let (w, p) = polar(a, PolarMethod::Qdwh, Par::Seq).unwrap();
		check(b, w_b.as_ref(), p_b.as_ref(), true, "subnormal: B");
		let w_diff = (&w - &w_b).norm_l2();
		assert!(w_diff <= tolerance::<f64>(4), "|W - W_B| = {w_diff:e}");
		// each entry of `P` and of `W P` rounds to the subnormal spacing `2^-1074` about `n` times
		let rebuilt = if a.nrows() >= a.ncols() {
			&w * &p
		} else {
			&p * &w
		};
		let residual = (&rebuilt - a).norm_l2();
		let bound =
			tolerance::<f64>(4) * a.norm_l2() + 64.0 * 2.0f64.powi(-1074);
		assert!(
			residual <= bound,
			"|A - WP| = {residual:e}, bound {bound:e}"
		);
	}
}

#[test]
fn polar_contract() {
	contracts::<f64>();
	contracts::<f32>();
	contracts::<c64>();
	contracts::<c32>();
}

/// QDWH stays backward stable on the nearly singular `sample(i, j)` matrices (condition about
/// 5e9 at 33 x 33). With an unpivoted $QR$ in the first steps it returned an orthonormal $W$
/// that is not the polar factor: $\|A - W P\|_F / \|A\|_F \approx 5 \cdot 10^{-10}$.
#[test]
fn polar_qdwh_backward_stable_nearly_singular() {
	fn run<T: Scalar>() {
		for (m, n) in [(33, 33), (40, 40), (100, 33)] {
			let a = Mat::<T>::from_fn(m, n, |i, j| {
				T::make(sample(i, j), sample(j + 7, i))
			});
			contract(a.as_ref(), true, "nearly singular");
		}
	}
	run::<f64>();
	run::<c64>();
}

#[test]
fn polar_rejects_non_finite_input() {
	fn rejects<T: Scalar>() {
		for (m, n) in [(6, 4), (5, 5)] {
			for bad in [f64::NAN, f64::INFINITY] {
				let mut a = sample_mat::<T>(m, n, 5);
				a[(2, 1)] = T::from_f64(bad);
				for method in METHODS {
					// left on the matrix, right on its transpose
					for a in [a.as_ref(), a.transpose()] {
						assert_eq!(
							polar(a, method, Par::Seq).err(),
							Some(PolarError::NoConvergence),
							"{m}x{n} {method:?} {bad}"
						);
					}
				}
			}
		}
	}
	rejects::<f64>();
	rejects::<f32>();
	rejects::<c64>();
	rejects::<c32>();
}

/// Finite entries near the largest finite value: `norm_l2` overflows there, which QDWH must not
/// turn into $W = 0$, and the hermitian part of $P$ must not overflow either. Checked with $A$
/// and $P$ divided by the largest entry, so that the products stay finite. Both methods must
/// succeed, also on the six complex cases whose largest singular value exceeds the largest finite
/// value (1.09 to 1.21 times it), where the SVD itself reports `NoConvergence` (#233).
#[test]
fn polar_handles_entries_near_overflow() {
	// returns the number of cases whose largest singular value is not representable
	fn near_overflow<T: Scalar>(largest: f64) -> usize {
		let mut sigma_overflows = 0;
		let big = 0.5 * largest;
		for (m, n) in [(1, 1), (5, 3), (3, 5), (4, 4)] {
			let sample = sample_mat::<T>(m, n, 9);
			let top = (0..m)
				.flat_map(|i| (0..n).map(move |j| (i, j)))
				.map(|(i, j)| sample[(i, j)].abs())
				.fold(zero::<T::Real>(), |a, b| a.fmax(b));
			// largest entry 1 first, then `big`: `big / top` itself overflows for `top < 1`
			let unit = Mat::<T>::from_fn(m, n, |i, j| {
				sample[(i, j)].mul_real(recip(&top))
			});
			let scale = from_f64::<T::Real>(big);
			let a =
				Mat::<T>::from_fn(m, n, |i, j| unit[(i, j)].mul_real(&scale));
			sigma_overflows += a.singular_values(Par::Seq).is_err() as usize;
			for method in METHODS {
				let tag = format!(
					"{} {m}x{n} {method:?}",
					std::any::type_name::<T>()
				);
				let (w, p) = polar(a.as_ref(), method, Par::Seq)
					.unwrap_or_else(|e| panic!("{tag}: {e:?}"));
				let p_unit = Mat::<T>::from_fn(p.nrows(), p.ncols(), |i, j| {
					p[(i, j)].mul_real(recip(&scale))
				});
				if m >= n {
					close((&w * &p_unit).as_ref(), unit.as_ref());
					unitary(w.as_ref());
				} else {
					close((&p_unit * &w).as_ref(), unit.as_ref());
					unitary(w.transpose());
				}
			}
		}
		sigma_overflows
	}
	let sigma_overflows = near_overflow::<f64>(f64::MAX)
		+ near_overflow::<c64>(f64::MAX)
		+ near_overflow::<f32>(f32::MAX as f64)
		+ near_overflow::<c32>(f32::MAX as f64);
	// the complex 5 x 3, 3 x 5 and 4 x 4 at both precisions
	assert_eq!(sigma_overflows, 6);
}

/// #233: a rank-one input whose only singular value, $s \sqrt 2$ for two entries
/// $s = 0.75 \cdot$ MAX, is not representable, while every entry of $P$ ($s / \sqrt 2$) is. The
/// SVD method keeps $\Sigma$ at the SVD's working scale and must return both factors; checked
/// at ordinary scale, with $A$ and $P$ divided by $s$. QDWH is left out: on a rank-deficient
/// input its $W$ is not orthonormal (`PolarMethod::Qdwh`).
#[test]
fn polar_svd_with_unrepresentable_singular_value() {
	fn run<T: Scalar>(largest: f64) {
		let s = from_f64::<T::Real>(0.75 * largest);
		let unit = Mat::<T>::from_fn(9, 3, |i, j| {
			if i == 0 && j < 2 {
				T::from_f64(1.0)
			} else {
				zero()
			}
		});
		let a = Mat::<T>::from_fn(9, 3, |i, j| unit[(i, j)].mul_real(&s));
		for (a, unit) in [
			(a.as_ref(), unit.as_ref()),
			(a.transpose(), unit.transpose()),
		] {
			let (m, n) = a.shape();
			let tag = format!("{} {m}x{n}", std::any::type_name::<T>());
			assert!(a.singular_values(Par::Seq).is_err(), "{tag}");
			let (w, p) = polar(a, PolarMethod::Svd, Par::Seq)
				.unwrap_or_else(|e| panic!("{tag}: {e:?}"));
			let p_unit = Mat::<T>::from_fn(p.nrows(), p.ncols(), |i, j| {
				p[(i, j)].mul_real(recip(&s))
			});
			if m >= n {
				close((&w * &p_unit).as_ref(), unit);
				unitary(w.as_ref());
			} else {
				close((&p_unit * &w).as_ref(), unit);
				unitary(w.transpose());
			}
		}
	}
	run::<f64>(f64::MAX);
	run::<c64>(f64::MAX);
	run::<f32>(f32::MAX as f64);
	run::<c32>(f32::MAX as f64);
}

/// Finite entries whose $P$ is not representable: every entry $0.9 \cdot$ MAX gives entries of
/// $P$ of $1.35 \cdot$ MAX on 3 x 2 and 2 x 3, where QDWH returned `Ok` with an infinite $P$.
/// Either method may refuse such input, but must not return a non-finite factor.
#[test]
fn polar_never_returns_non_finite_factors() {
	fn run<T: Scalar>(largest: f64) {
		for (m, n) in [(2, 2), (3, 2), (2, 3)] {
			let a = Mat::<T>::full(m, n, T::from_f64(0.9 * largest));
			for method in METHODS {
				if let Ok((w, p)) = polar(a.as_ref(), method, Par::Seq) {
					assert!(
						is_finite(&w.norm_max()) && is_finite(&p.norm_max()),
						"{} {m}x{n} {method:?}",
						std::any::type_name::<T>()
					);
				}
			}
		}
	}
	run::<f64>(f64::MAX);
	run::<c64>(f64::MAX);
	run::<f32>(f32::MAX as f64);
	run::<c32>(f32::MAX as f64);
}

/// A row-major input, and outputs that are views into larger buffers (column stride above the row
/// count) whose other entries stay untouched. `right_polar` also writes through transposed views.
fn views<T: Scalar>(m: usize, n: usize) {
	let a = sample_mat::<T>(m, n, 6);
	let a_row_major = a.transpose().to_owned();
	let a_view = a_row_major.transpose();
	let k = Ord::min(m, n);
	let guard = T::from_f64(-3.25);
	for method in METHODS {
		let tag = format!("{m}x{n} {method:?}");
		let (w_ref, p_ref) = polar(a.as_ref(), method, Par::Seq).unwrap();
		let mut w_buf = Mat::<T>::full(m + 3, n + 3, guard.clone());
		let mut p_buf = Mat::<T>::full(k + 3, k + 3, guard.clone());
		{
			let params = params::<T>(method).into();
			let w = w_buf.as_mut().submatrix_mut(1, 2, m, n);
			let p = p_buf.as_mut().submatrix_mut(1, 2, k, k);
			if m >= n {
				let mut mem = MemBuffer::new(left_polar_scratch::<T>(
					m,
					n,
					ParShape::Seq,
					params,
				));
				left_polar(
					a_view,
					w,
					p,
					Par::Seq,
					MemStack::new(&mut mem),
					params,
				)
				.unwrap();
			} else {
				let mut mem = MemBuffer::new(right_polar_scratch::<T>(
					m,
					n,
					ParShape::Seq,
					params,
				));
				right_polar(
					a_view,
					p,
					w,
					Par::Seq,
					MemStack::new(&mut mem),
					params,
				)
				.unwrap();
			}
		}
		let w = w_buf.as_ref().submatrix(1, 2, m, n).to_owned();
		let p = p_buf.as_ref().submatrix(1, 2, k, k).to_owned();
		check(a.as_ref(), w.as_ref(), p.as_ref(), true, &tag);
		agree(a.as_ref(), (&w_ref, &p_ref), (&w, &p), &tag);
		for (buf, rows, cols) in [(&w_buf, m, n), (&p_buf, k, k)] {
			for j in 0..buf.ncols() {
				for i in 0..buf.nrows() {
					let written = (1..1 + rows).contains(&i)
						&& (2..2 + cols).contains(&j);
					if !written {
						assert!(
							buf[(i, j)] == guard,
							"{tag}: guard ({i}, {j}) overwritten"
						);
					}
				}
			}
		}
	}
}

#[test]
fn polar_views() {
	for (m, n) in [(8, 3), (3, 8), (8, 8)] {
		views::<f64>(m, n);
		views::<c64>(m, n);
	}
}

#[cfg(feature = "threadteam")]
#[test]
fn polar_parallel_matches_sequential() {
	fn team_agrees<T: Scalar>(m: usize, n: usize) {
		let a = sample_mat::<T>(m, n, 7);
		for method in METHODS {
			let tag = format!("{m}x{n} {method:?}");
			let seq = polar(a.as_ref(), method, Par::Seq).unwrap();
			let team = colmat::team::with_team(2, |par| {
				polar(a.as_ref(), method, par)
			})
			.unwrap();
			check(
				a.as_ref(),
				team.0.as_ref(),
				team.1.as_ref(),
				true,
				&format!("{tag} team"),
			);
			agree(a.as_ref(), (&seq.0, &seq.1), (&team.0, &team.1), &tag);
		}
	}
	for (m, n) in [(257, 129), (129, 257), (129, 129)] {
		team_agrees::<f64>(m, n);
		team_agrees::<c64>(m, n);
	}
}

/// One case of `fixtures/polar_reference/polar.txt`.
struct Case {
	left: bool,
	complex: bool,
	mats: Vec<(String, Mat<f64>)>,
}

fn parse(text: &str) -> Vec<Case> {
	let mut cases: Vec<Case> = Vec::new();
	let mut lines = text.lines().filter(|l| !l.starts_with('#'));
	while let Some(line) = lines.next() {
		let words: Vec<&str> = line.split_whitespace().collect();
		if words[0] == "case" {
			cases.push(Case {
				left: words[1] == "left",
				complex: words[2] == "c64",
				mats: Vec::new(),
			});
			continue;
		}
		let (rows, cols): (usize, usize) =
			(words[1].parse().unwrap(), words[2].parse().unwrap());
		let mut mat = Mat::<f64>::zeros(rows, cols);
		for i in 0..rows {
			let values: Vec<f64> = lines
				.next()
				.unwrap()
				.split_whitespace()
				.map(|v| v.parse().unwrap())
				.collect();
			assert_eq!(values.len(), cols);
			for (j, v) in values.into_iter().enumerate() {
				mat[(i, j)] = v;
			}
		}
		cases
			.last_mut()
			.unwrap()
			.mats
			.push((words[0].to_string(), mat));
	}
	cases
}

/// The named matrix of `case`.
fn get<T: Scalar>(case: &Case, name: &str) -> Mat<T> {
	let find = |key: String| {
		case.mats
			.iter()
			.find(|(n, _)| *n == key)
			.map(|(_, m)| m.clone())
			.unwrap()
	};
	let (re, im) = if case.complex {
		(find(format!("{name}_re")), find(format!("{name}_im")))
	} else {
		let re = find(name.to_string());
		let im = Mat::zeros(re.nrows(), re.ncols());
		(re, im)
	};
	Mat::from_fn(re.nrows(), re.ncols(), |i, j| {
		T::make(re[(i, j)], im[(i, j)])
	})
}

/// A full-rank polar decomposition is unique, so both methods match MatrixFactorizations.jl
/// entry by entry, within the bounds of `agree`.
fn matches_reference<T: Scalar>(case: &Case) {
	let a = get::<T>(case, "a");
	let (w_ref, p_ref) = (get::<T>(case, "w"), get::<T>(case, "p"));
	assert_eq!(case.left, a.nrows() >= a.ncols());
	for method in METHODS {
		let tag = format!("{}x{} {method:?}", a.nrows(), a.ncols());
		let (w, p) = polar(a.as_ref(), method, Par::Seq).unwrap();
		agree(a.as_ref(), (&w_ref, &p_ref), (&w, &p), &tag);
	}
}

#[test]
fn polar_match_matrixfactorizations_jl() {
	let cases = parse(include_str!("fixtures/polar_reference/polar.txt"));
	assert_eq!(cases.len(), 8);
	for case in &cases {
		if case.complex {
			matches_reference::<c64>(case);
		} else {
			matches_reference::<f64>(case);
		}
	}
}
