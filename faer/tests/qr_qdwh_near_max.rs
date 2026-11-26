//! tensor4all/colmat-rs#265 and #266: finite input near the overflow threshold.
//!
//! `qr` formed `x_0 + sign * ||x||` unscaled, which overflows when `|x_0| + ||x|| > MAX` even
//! though `||x||` is finite, and returned a `Q` with `Q R != A`. QDWH inherits it (#265) and
//! also formed `P = W^H A` at the input's scale, which overflows for `diag(MAX, MAX / 2)` and
//! for `sigma_max > MAX` with a representable `P` (#266).
//!
//! every assertion is against an exact power-of-two rescaling of the input, so nothing in the
//! check itself overflows.

use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::polar::{
	PolarMethod, PolarParams, left_polar, left_polar_scratch, right_polar,
	right_polar_scratch,
};
use colmat::{Auto, Mat, Par, c32, c64};

macro_rules! near_max_tests {
	($modname:ident, $t:ty, $real:ty, $cplx:expr, $mk:expr, $exp:expr) => {
		mod $modname {
			use super::*;
			type T = $t;
			type R = $real;

			fn mk(re: R, im: R) -> T {
				let f: fn(R, R) -> T = $mk;
				f(re, im)
			}
			/// exact: `2^-EXP`, so `MAX * 2^-EXP` is O(1) and nothing below overflows
			fn down() -> R {
				(2.0 as R).powi(-$exp)
			}
			fn eps() -> R {
				R::EPSILON
			}
			fn pattern(rows: &[&[f64]], frac: f64) -> Mat<T> {
				// complex inputs rotate every entry by the same phase (cos, sin) = (0.8, 0.6):
				// same moduli and norms, nonzero imaginary parts
				let (c, s) = if $cplx { (0.8, 0.6) } else { (1.0, 0.0) };
				Mat::from_fn(rows.len(), rows[0].len(), |i, j| {
					let v = (frac * rows[i][j]) as R * R::MAX;
					mk(v * c as R, v * s as R)
				})
			}
			fn scaled(a: &Mat<T>) -> Mat<T> {
				Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
					a[(i, j)] * mk(down(), 0.0)
				})
			}
			fn fro(a: &Mat<T>) -> f64 {
				a.norm_l2() as f64
			}
			fn ortho(q: &Mat<T>) -> f64 {
				let n = q.ncols();
				let g = q.adjoint() * q;
				let i = Mat::<T>::identity(n, n);
				fro(&(&g - &i))
			}

			fn qr_case(name: &str, a: &Mat<T>) {
				let (m, n) = a.shape();
				let qr = a.qr(Par::Seq);
				let q = qr.compute_thin_Q(Par::Seq);
				let r = qr.thin_R().to_owned();
				let (qs, rs, as_) = (q, scaled(&r), scaled(a));
				let o = ortho(&qs);
				let res = fro(&(&qs * &rs - &as_)) / fro(&as_);
				println!(
					"qr  {name}: {m}x{n} ||QhQ-I|| = {o:.2e}, ||QR-A||/||A|| = {res:.2e}"
				);
				let tol = 16.0 * (m as f64) * eps() as f64;
				assert!(o <= tol, "{name}: orthogonality {o:e} > {tol:e}");
				assert!(res <= tol, "{name}: QR residual {res:e} > {tol:e}");
			}

			fn polar_qdwh(name: &str, a: &Mat<T>, right: bool) {
				let (m, n) = a.shape();
				let mut prm = <PolarParams as Auto<T>>::auto();
				prm.method = PolarMethod::Qdwh;
				let prm = prm.into();
				let (mut w, mut p) = if right {
					(Mat::<T>::zeros(m, n), Mat::<T>::zeros(m, m))
				} else {
					(Mat::<T>::zeros(m, n), Mat::<T>::zeros(n, n))
				};
				let r = if right {
					let mut mem = MemBuffer::new(right_polar_scratch::<T>(
						m,
						n,
						Par::Seq.shape(),
						prm,
					));
					// right polar: A = P W, P is m x m and W is m x n
					right_polar(
						a.as_ref(),
						p.as_mut(),
						w.as_mut(),
						Par::Seq,
						MemStack::new(&mut mem),
						prm,
					)
				} else {
					let mut mem = MemBuffer::new(left_polar_scratch::<T>(
						m,
						n,
						Par::Seq.shape(),
						prm,
					));
					left_polar(
						a.as_ref(),
						w.as_mut(),
						p.as_mut(),
						Par::Seq,
						MemStack::new(&mut mem),
						prm,
					)
				};
				println!("qdwh {name}: {m}x{n} right={right}: {r:?}");
				assert!(
					r.is_ok(),
					"{name}: QDWH returned {r:?} for a representable P"
				);
				assert!(
					p.norm_max().is_finite() as bool,
					"{name}: P not finite"
				);
				let as_ = scaled(a);
				let ps = scaled(&p);
				let res = if right {
					fro(&(&ps * &w - &as_)) / fro(&as_)
				} else {
					fro(&(&w * &ps - &as_)) / fro(&as_)
				};
				let o = if right {
					ortho(&w.adjoint().to_owned())
				} else {
					ortho(&w)
				};
				println!(
					"     ||W^H W-I|| = {o:.2e}, ||A-WP||/||A|| = {res:.2e}"
				);
				let tol = 64.0 * (m.max(n) as f64) * eps() as f64;
				assert!(o <= tol, "{name}: orthogonality {o:e} > {tol:e}");
				assert!(res <= tol, "{name}: polar residual {res:e} > {tol:e}");
			}

			#[test]
			fn qr_single_column() {
				for f in [0.7, 0.5, 0.25, 0.125] {
					qr_case(
						&format!("[f f]^T, f={f}"),
						&pattern(&[&[1.0], &[1.0]], f),
					);
				}
				// head near MAX, tail smaller: |x_0| + ||x|| > MAX with ||x|| < MAX
				qr_case("head .8 tail .5", &pattern(&[&[0.8], &[0.5]], 1.0));
			}

			#[test]
			fn qr_multi_column() {
				qr_case(
					"issue A2",
					&pattern(
						&[&[-1.0, -2.0], &[0.0, -2.0], &[-1.0, -2.0]],
						0.25,
					),
				);
				// one column at the limit, one small
				qr_case(
					"big+small column",
					&pattern(
						&[&[0.6, 1e-300], &[0.6, 2e-300], &[0.2, 1e-300]],
						1.0,
					),
				);
				qr_case(
					"MAX/2, MAX/4 columns",
					&pattern(&[&[0.5, 0.25], &[0.0, 0.25], &[0.5, 0.0]], 1.0),
				);
			}

			/// a column of entries near `MIN_POSITIVE` next to one near `MAX`: the prescale must not
			/// cost the small column its accuracy, so it is checked column by column, unscaled
			#[test]
			fn qr_tiny_column_next_to_huge_one() {
				let tiny = R::MIN_POSITIVE * 1000.0;
				let a = Mat::from_fn(3, 2, |i, j| {
					if j == 0 {
						mk([0.6, 0.6, 0.2][i] as R * R::MAX, 0.0)
					} else {
						mk([1.0, 2.0, 1.0][i] as R * tiny, 0.0)
					}
				});
				let qr = a.qr(Par::Seq);
				let q = qr.compute_thin_Q(Par::Seq);
				let r = qr.thin_R().to_owned();
				let rec = &q * &r;
				let col1 = Mat::from_fn(3, 1, |i, _| rec[(i, 1)] - a[(i, 1)]);
				let a1 = Mat::from_fn(3, 1, |i, _| a[(i, 1)]);
				let res = fro(&col1) / fro(&a1);
				println!(
					"qr tiny column next to huge: column-1 residual = {res:.2e}"
				);
				assert!(
					res <= 16.0 * 3.0 * eps() as f64,
					"small column residual {res:e}"
				);
			}

			#[test]
			fn qdwh_tall_and_wide() {
				let a1 = pattern(&[&[1.0], &[1.0]], 0.5);
				polar_qdwh("A1", &a1, false);
				let a2 = pattern(
					&[&[-1.0, -2.0], &[0.0, -2.0], &[-1.0, -2.0]],
					0.25,
				);
				polar_qdwh("A2", &a2, false);
				polar_qdwh("A2^T", &a2.transpose().to_owned(), true);
			}

			#[test]
			fn qdwh_diagonal_at_max() {
				for d in
					[&[1.0, 0.5][..], &[1.0, 0.5, 0.25][..], &[0.5, 0.25][..]]
				{
					let n = d.len();
					let a = Mat::from_fn(n, n, |i, j| {
						if i == j {
							mk(d[i] as R * R::MAX, 0.0)
						} else {
							mk(0.0, 0.0)
						}
					});
					polar_qdwh(&format!("diag{d:?}"), &a, false);
				}
			}
		}
	};
}

near_max_tests!(f64_, f64, f64, false, |re, _| re, 1000);
near_max_tests!(f32_, f32, f32, false, |re, _| re, 100);
near_max_tests!(c64_, c64, f64, true, |re, im| c64::new(re, im), 1000);
near_max_tests!(c32_, c32, f32, true, |re, im| c32::new(re, im), 100);

/// #266 (b): `sigma_max = 1.30 MAX` but every component of `P` is at most `0.990 MAX`.
#[test]
fn qdwh_sigma_max_above_max_with_representable_p() {
	let z = |re: f64, im: f64| c64::new(re, im);
	let b = [
		[
			z(1.6179238213760842e308, 3.378294920858522e307),
			z(-8.70149072518396e307, -2.8338165275915896e307),
		],
		[
			z(9.24455710622842e307, -5.69871909840064e307),
			z(-5.507405895709892e307, -1.2523721215433915e308),
		],
	];
	let b = Mat::from_fn(2, 2, |i, j| b[i][j]);
	let mut prm = <PolarParams as Auto<c64>>::auto();
	prm.method = PolarMethod::Qdwh;
	let prm = prm.into();
	let (mut p, mut wh) = (Mat::<c64>::zeros(2, 2), Mat::<c64>::zeros(2, 2));
	let mut mem =
		MemBuffer::new(right_polar_scratch::<c64>(2, 2, Par::Seq.shape(), prm));
	let r = right_polar(
		b.as_ref(),
		p.as_mut(),
		wh.as_mut(),
		Par::Seq,
		MemStack::new(&mut mem),
		prm,
	);
	assert!(r.is_ok(), "QDWH returned {r:?} for a representable P");
	let s = 2.0f64.powi(-1000);
	let sc =
		|m: &Mat<c64>| Mat::from_fn(2, 2, |i, j| m[(i, j)] * c64::new(s, 0.0));
	let g = &wh * wh.adjoint();
	let o = (&g - Mat::<c64>::identity(2, 2)).norm_l2();
	let res = (&sc(&p) * &wh - sc(&b)).norm_l2() / sc(&b).norm_l2();
	println!(
		"qdwh (b): Ok, ||W W^H - I|| = {o:.2e}, ||A - P W||/||A|| = {res:.2e}"
	);
	assert!(o <= 64.0 * 2.0 * f64::EPSILON, "orthogonality {o:e}");
	assert!(res <= 64.0 * 2.0 * f64::EPSILON, "residual {res:e}");
}

/// the diagnosis in numbers (`cargo test -- --nocapture diagnosis`): for #265's `A1 = [0.5 MAX;
/// 0.5 MAX]` the reflector's `head + sign * norm` is `1.207 MAX`, so it is `inf` while `norm`
/// is finite, and its reciprocal, which scales the essential part, is `0`. for #266 (a) the
/// converged `W[0,0] = 1 + 2^-52` makes `W[0,0] * MAX` round to `inf`
#[test]
fn diagnosis_overflowing_intermediates() {
	let (head, tail) = (0.5 * f64::MAX, 0.5 * f64::MAX);
	let norm = head.hypot(tail);
	let head_with_beta = head + norm;
	println!(
		"#265 A1: norm = {norm:e} (finite), head + norm = {head_with_beta:e}, 1/(head + norm) = {:e}",
		head_with_beta.recip()
	);
	assert!(norm.is_finite() && head_with_beta.is_infinite());
	let w00 = 1.0 + f64::EPSILON;
	println!(
		"#266 (a): W[0,0] = {w00:e}, W[0,0] * MAX = {:e}",
		w00 * f64::MAX
	);
	assert!((w00 * f64::MAX).is_infinite());
}

/// a head near `MAX` with a zero tail (review of #303): the reflector is the identity, and
/// `beta` and the norm must come back at the input's scale, not at the rescaled one
#[test]
fn big_head_with_zero_tail_keeps_its_scale() {
	use colmat::linalg::householder::make_householder_in_place;
	let x0 = 0.9 * f64::MAX;
	let mut m = Mat::<f64>::zeros(3, 1);
	m[(0, 0)] = x0;
	let mut beta = x0;
	let info = make_householder_in_place(
		&mut beta,
		m.as_mut().col_mut(0).subrows_mut(1, 2),
	);
	println!("head = {beta:e} (x_0 = {x0:e}), norm = {:e}", info.norm);
	assert_eq!(beta, x0);
	assert_eq!(info.norm, x0);
}

/// the rescaled reflector itself, which `qr` never reaches (its prescale keeps every column
/// below `MAX / 8`): `x = [0.5 MAX; 0.5 MAX]` has `|head| + norm = 1.207 MAX`
#[test]
fn rescaled_reflector_on_its_own() {
	use colmat::linalg::householder::make_householder_in_place;
	let h = 0.5 * f64::MAX;
	let mut m = Mat::<f64>::zeros(2, 1);
	m[(1, 0)] = h;
	let mut beta = h;
	let info = make_householder_in_place(
		&mut beta,
		m.as_mut().col_mut(0).subrows_mut(1, 1),
	);
	let norm = h * 2.0f64.sqrt();
	let essential = m[(1, 0)];
	// exact: v = x_tail / (x_0 + norm), tau = (1 + v^2) / 2, beta = -norm
	let v = 0.5 / (0.5 + 2.0f64.sqrt() / 2.0);
	println!(
		"beta/norm = {:e}, essential = {essential:e} (exact {v:e}), tau = {:e}",
		beta / norm,
		info.tau
	);
	assert!(((beta + norm) / norm).abs() <= 4.0 * f64::EPSILON);
	assert!(((info.norm - norm) / norm).abs() <= 4.0 * f64::EPSILON);
	assert!(((essential - v) / v).abs() <= 4.0 * f64::EPSILON);
	assert!(
		((info.tau - 0.5 * (1.0 + v * v)) / info.tau).abs()
			<= 4.0 * f64::EPSILON
	);
}
