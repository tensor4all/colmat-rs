//! Independent divide-and-conquer weight and eigenvector repairs. Existing scalar arithmetic and its order are retained.
//! Ownership: one weight element or one complete output column per item; no partial buffers.
use super::*;
// Calibrated on witch, Xeon E5-2699v3, f64,512/input1, three1T process medians.
// Root non-deflated count465 was measured. These are experimental host-only rates,
// not portable defaults: this branch is an experiment pending canonical-host calibration.
// See docs/worklogs/2026-11-14-eigh-parallel-repair.md for evidence and adoption limits.
const WEIGHT_NS: f64 = 4.456651636027287;
const VECTOR_NS: f64 = 14.712088373655915;
fn width(par: &Par, units: usize, jobs: usize, ns: f64) -> usize {
	crate::utils::cost::par_width_time_upto(
		par,
		core::time::Duration::from_secs_f64(units as f64 * ns * 1e-9),
		1,
		jobs,
	)
}
#[inline(always)]
fn weight<T: RealField>(
	i: usize,
	zi: &mut T,
	d: ColRef<'_, T, usize, ContiguousFwd>,
	mus: ColRef<'_, T, usize, ContiguousFwd>,
	shifts: ColRef<'_, T, usize, ContiguousFwd>,
	n: usize,
) {
	let ref di = d[i].copy();
	let ref mu_i = mus[i].copy();
	let ref shift_i = shifts[i].copy();
	let mut prod = mu_i + (shift_i - di);
	(0..i).chain(i + 1..n).for_each(|k| {
		let ref dk = d[k].copy();
		let ref mu_k = mus[k].copy();
		let ref shift_k = shifts[k].copy();
		let ref numerator = mu_k + (shift_k - di);
		let ref denominator = dk - di;
		prod *= numerator / denominator;
	});
	let prod = prod.abs().sqrt();
	*zi = if *zi < zero() { -prod } else { prod };
}
pub(super) fn weights<T: RealField>(
	mut z: ColMut<'_, T, usize, ContiguousFwd>,
	d: ColRef<'_, T, usize, ContiguousFwd>,
	mus: ColRef<'_, T, usize, ContiguousFwd>,
	shifts: ColRef<'_, T, usize, ContiguousFwd>,
	n: usize,
	par: Par,
) {
	let w = width(&par, n.saturating_mul(n), n, WEIGHT_NS);
	if w < 2 {
		for i in 0..n {
			weight(i, &mut z[i], d, mus, shifts, n);
		}
		return;
	}
	#[cfg(feature = "rayon")]
	{
		use rayon::prelude::*;
		let target = z.rb_mut().subrows_mut(0, n).as_slice_mut();
		crate::utils::thread::with_width(par, w, |par| {
			crate::utils::thread::for_each(
				par,
				w,
				target.par_iter_mut().enumerate(),
				|(i, zi)| weight(i, zi, d, mus, shifts, n),
			);
		});
	}
	#[cfg(not(feature = "rayon"))]
	unreachable!("a parallel repair requires a team");
}
#[inline(always)]
fn column<T: RealField>(
	mut col: ColMut<'_, T, usize, ContiguousFwd>,
	pj: usize,
	rows: &[usize],
	d: ColRef<'_, T, usize, ContiguousFwd>,
	z: ColRef<'_, T, usize, ContiguousFwd>,
	mus: ColRef<'_, T, usize, ContiguousFwd>,
	shifts: ColRef<'_, T, usize, ContiguousFwd>,
	nd: usize,
	deflated: usize,
) {
	if pj >= nd {
		col.rb_mut().fill(zero());
		col[rows[pj]] = one();
	} else {
		let ref mu_j = mus[pj].copy();
		let ref shift_j = shifts[pj].copy();
		for (i, &row) in rows[..nd].iter().enumerate() {
			let ref zi = z[i].copy();
			let ref di = d[i].copy();
			col[row] = zi / ((di - shift_j) - mu_j);
		}
		for &row in &rows[nd..nd + deflated] {
			col[row] = zero();
		}
		let ref inv_norm = col.rb().norm_l2().recip();
		z!(col.rb_mut()).for_each(|unzip!(x)| *x *= inv_norm);
	}
}
pub(super) fn vectors<T: RealField>(
	mut u: MatMut<'_, T, usize, usize, ContiguousFwd>,
	order: &[usize],
	rows: &[usize],
	d: ColRef<'_, T, usize, ContiguousFwd>,
	z: ColRef<'_, T, usize, ContiguousFwd>,
	mus: ColRef<'_, T, usize, ContiguousFwd>,
	shifts: ColRef<'_, T, usize, ContiguousFwd>,
	nd: usize,
	deflated: usize,
	par: Par,
) {
	let n = u.ncols();
	let w = width(&par, n.saturating_mul(nd), n, VECTOR_NS);
	if w < 2 {
		for (j, &pj) in order.iter().enumerate() {
			column(
				u.rb_mut().col_mut(j),
				pj,
				rows,
				d,
				z,
				mus,
				shifts,
				nd,
				deflated,
			);
		}
		return;
	}
	#[cfg(feature = "rayon")]
	{
		use rayon::prelude::*;
		crate::utils::thread::with_width(par, w, |par| {
			crate::utils::thread::for_each(
				par,
				w,
				u.rb_mut().par_col_iter_mut().enumerate(),
				|(j, col)| {
					column(col, order[j], rows, d, z, mus, shifts, nd, deflated)
				},
			);
		});
	}
	#[cfg(not(feature = "rayon"))]
	unreachable!("a parallel repair requires a team");
}

#[cfg(all(test, feature = "threadteam"))]
mod tests {
	use super::*;
	use crate::assert;

	#[test]
	fn repairs_preserve_bits_and_fill_deflated_columns() {
		macro_rules! check {
			($ty:ty) => {{
				for (n, k) in [(0, 0), (1, 0), (17, 13), (256, 239), (512, 465)]
				{
					let d = Col::<$ty>::from_fn(n, |i| i as $ty + 1.0);
					let mus = Col::<$ty>::from_fn(n, |_| 0.125);
					let shifts = d.clone();
					let initial = Col::<$ty>::from_fn(n, |i| {
						if i % 2 == 0 { 1.0 } else { -1.0 }
					});
					let rows: Vec<usize> = (0..n).rev().collect();
					let order: Vec<usize> = (0..n).collect();
					let run = |mut par: Par| {
						let mut z = initial.clone();
						weights(
							z.as_mut().try_as_col_major_mut().unwrap(),
							d.as_ref().try_as_col_major().unwrap(),
							mus.as_ref().try_as_col_major().unwrap(),
							shifts.as_ref().try_as_col_major().unwrap(),
							k,
							par.reborrow(),
						);
						let mut u =
							Mat::<$ty>::from_fn(n, n, |_, _| <$ty>::NAN);
						vectors(
							u.as_mut().try_as_col_major_mut().unwrap(),
							&order,
							&rows,
							d.as_ref().try_as_col_major().unwrap(),
							z.as_ref().try_as_col_major().unwrap(),
							mus.as_ref().try_as_col_major().unwrap(),
							shifts.as_ref().try_as_col_major().unwrap(),
							k,
							n - k,
							par,
						);
						(z, u)
					};
					let (z0, u0) = run(Par::Seq);
					for threads in [2, 4, 8] {
						let (z1, u1) = crate::team::with_team(threads, run);
						for i in 0..n {
							assert_eq!(z0[i].to_bits(), z1[i].to_bits());
						}
						for j in 0..n {
							for i in 0..n {
								assert!(u1[(i, j)].is_finite());
								assert_eq!(
									u0[(i, j)].to_bits(),
									u1[(i, j)].to_bits()
								);
							}
						}
					}
				}
			}};
		}
		check!(f32);
		check!(f64);
	}
}
