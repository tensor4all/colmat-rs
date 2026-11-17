//! dqds: the singular values of a real bidiagonal matrix.
//!
//! The differential form of the implicit QL iteration with shifts, on the *squared* factors
//! `a_i = d_i^2` and `b_i = e_i^2`. One sweep is the transform
//!
//! ```text
//! delta = a_1 - tau
//! for i in 1..n-1:
//!     ahat_i = delta + b_i
//!     t      = a_{i+1} / ahat_i
//!     b_i    = b_i * t
//!     a_i    = ahat_i
//!     delta  = delta * t - tau
//! a_n = delta
//! ```
//!
//! which is the algorithm of Fernando & Parlett, *Accurate singular values and differential qd
//! algorithms* (Numer. Math. 67, 1994) and Parlett & Marques, *An implementation of the dqds
//! algorithm (positive case)* (LAPACK Working Note 155, 2000). It applies no rotations, so a sweep
//! costs one division and one multiplication per entry. The release test in the eigenvalue's units
//! and the correction of a shift rejected at the last factor are the paper's; the shift itself (the
//! trailing 2x2's eigenvalue backed off by its eigenvector-weighted second-order coupling) is this
//! crate's own, chosen by counting sweeps. Nothing here is translated from LAPACK or OpenBLAS
//! source.
//!
//! A sweep writes the factors it produces into a second buffer, so an accepted sweep is a swap of
//! roles and a rejected one (a non-positive or non-finite factor: the shift was too large) costs
//! nothing to undo. The shift is the smaller eigenvalue of the trailing 2x2 of `B^T B`, backed off
//! by its second-order coupling to the entry above when the smallest factor is at the bottom, and
//! half of the smallest factor otherwise; a shift that is rejected at the last factor alone is
//! corrected by that factor (the bottom eigenvalue is at about `tau + delta`), an interior rejection
//! cuts it by four, and a block that cannot be accepted at a zero shift is reported as
//! non-convergence. An eigenvalue is released when the off-diagonal coupling it to the rest is
//! negligible against it (`(100 eps)^2` of `sigma + a`, in squared units) or against the factor
//! above it; two are released at once by the trailing 2x2 in closed form when the coupling above
//! that 2x2 is negligible.
//!
//! Measured on the bidiagonal of a random dense 512x512 (`docs/worklogs/2026-10-23-dqds-shift.md`):
//! 3.5 accepted sweeps per eigenvalue with 7% rejected, against 5.7 accepted and 3.8 rejected for
//! the save-and-retry loop with the strict `eps^2 sqrt(a_i a_{i+1})` test this replaced.

use crate::internal_prelude::*;
use crate::linalg::svd::SvdError;

/// How many sweeps one entry may take before the block gives up.
const SWEEPS_PER_ENTRY: usize = 200;

/// The workspace of [`singular_values`] for an `n x n` bidiagonal: the squared factors and the
/// second buffer of the ping-pong, two arrays of `n` and two of `n - 1`.
pub(crate) fn singular_values_scratch<T: RealField>(n: usize) -> StackReq {
	StackReq::all_of(&[
		StackReq::new::<T>(n),
		StackReq::new::<T>(n.saturating_sub(1)),
		StackReq::new::<T>(n),
		StackReq::new::<T>(n.saturating_sub(1)),
	])
}

/// Singular values of the bidiagonal with diagonal `d` and off-diagonal `e`, in place. The work
/// arrays come from `stack` ([`singular_values_scratch`]); nothing is allocated on the heap.
///
/// On success `d` holds the singular values in **descending** order. `e` is never written, and on
/// an error `d` is unchanged too, so a caller can finish on another path from the same bidiagonal
/// (#110). `d` may hold negative entries on entry, since flipping the sign of a diagonal entry is
/// an orthogonal transform.
///
/// `d` must not contain a zero: the transform divides by the factors, and a rank-deficient
/// bidiagonal has a cheaper answer than this loop. The caller checks that and keeps its own
/// rotation-based path for it.
///
/// A diagonal entry whose square is below `min_positive` is rejected too (#209): a subnormal square
/// does not carry `eps`'s relative precision, so the value released from it is not guaranteed to
/// keep its relative accuracy. The test is conservative — a factor that happens to be a power of two
/// squares exactly and is rejected with the rest — and a caller told about the failure can finish on
/// the rotation path above with the bidiagonal intact.
pub(crate) fn singular_values<T: RealField>(
	d: &mut [T],
	e: &mut [T],
	stack: &mut MemStack,
) -> Result<(), SvdError> {
	let n = d.len();
	core::assert!(e.len() + 1 >= n, "off-diagonal shorter than the diagonal");
	if n == 0 {
		return Ok(());
	}

	// Scale before squaring: a finite bidiagonal scaled by 1e200 has representable singular values
	// and unrepresentable squares.
	let mut scale = zero::<T>();
	for i in 0..n {
		scale = scale.fmax(d[i].clone().abs());
		if i < e.len() {
			scale = scale.fmax(e[i].clone().abs());
		}
	}
	if !scale.is_finite() {
		return Err(SvdError::NoConvergence);
	}
	if scale == zero() {
		for x in d.iter_mut() {
			*x = zero();
		}
		return Ok(());
	}

	let inv_scale = scale.recip();
	let (mut a, stack) = stack.make_with::<T>(n, |_| zero());
	let (mut b, stack) = stack.make_with::<T>(n.saturating_sub(1), |_| zero());
	for i in 0..n {
		let v = d[i].clone() * inv_scale.clone();
		let v2 = v.clone() * v.clone();
		// A subnormal square does not carry `eps`'s relative precision, so the value released from it
		// is not guaranteed to keep its relative accuracy: report non-convergence instead, with `d`
		// still untouched, and let the caller finish on its rotation path (#209). Testing the stored
		// product is exact, unlike a threshold on a rounded square root of `min_positive`.
		// Off-diagonals are left alone: this guard is about the diagonal initialization, and a
		// subnormal square there is absorbed by the split below while its neighbours are in range.
		if v2 < min_positive::<T>() {
			return Err(SvdError::NoConvergence);
		}
		a[i] = v2;
	}
	for i in 0..n - 1 {
		let v = e[i].clone() * inv_scale.clone();
		b[i] = v.clone() * v.clone();
	}

	// The second buffer of the ping-pong: a sweep writes the factors it produces beside the ones it
	// reads, so a rejected sweep costs nothing to undo and an accepted one is a swap of roles.
	let (mut a2, stack) = stack.make_with::<T>(n, |_| zero());
	let (mut b2, _) = stack.make_with::<T>(n.saturating_sub(1), |_| zero());

	// Split at off-diagonals that are negligible against their neighbours before any shift exists.
	let tol2 = {
		let tol = from_f64::<T>(100.0) * eps::<T>();
		tol.clone() * tol
	};
	let mut lo = 0usize;
	while lo < n {
		let mut hi = lo + 1;
		while hi < n
			&& b[hi - 1] > tol2.clone() * a[hi - 1].clone().fmin(a[hi].clone())
		{
			hi += 1;
		}
		block(
			&mut a[lo..hi],
			&mut b[lo..hi - 1],
			&mut a2[lo..hi],
			&mut b2[lo..hi - 1],
			zero(),
		)?;
		lo = hi;
	}

	for i in 0..n {
		d[i] = a[i].clone().fmax(zero::<T>()).sqrt() * scale.clone();
	}
	// The public contract is descending order; the iteration produces none. Unstable: the stable
	// sort allocates its merge buffer on the heap once the slice outgrows a stack buffer (700 `f64`
	// values do).
	d.sort_unstable_by(|x, y| {
		y.partial_cmp(x).unwrap_or(core::cmp::Ordering::Equal)
	});
	Ok(())
}

/// What one sweep reports besides its factors.
struct Sweep<T> {
	/// The smallest factor it produced, or the first non-positive one when it failed.
	dmin: T,
	/// Whether every factor before the last one was positive: a failure at the last factor alone
	/// says the shift overshot the bottom eigenvalue by about `-dmin`.
	all_but_last_positive: bool,
	ok: bool,
}

#[allow(clippy::needless_range_loop)]
// index loop kept as written: the body indexes parallel buffers and strided views
/// One contiguous block, with the shift `sigma` its factors are already relative to. On return
/// `a` holds the block's eigenvalues (unsorted, in squared units, with `sigma` added back).
fn block<T: RealField>(
	a: &mut [T],
	b: &mut [T],
	a2: &mut [T],
	b2: &mut [T],
	sigma: T,
) -> Result<(), SvdError> {
	let n = a.len();
	if n == 0 {
		return Ok(());
	}
	let eps = eps::<T>();
	let tol2 = {
		let tol = from_f64::<T>(100.0) * eps.clone();
		tol.clone() * tol
	};
	let quarter = from_f64::<T>(0.25);
	let half = from_f64::<T>(0.5);
	let two = from_f64::<T>(2.0);
	let mut sweeps_left = SWEEPS_PER_ENTRY * n;

	// `cur` is the buffer that holds the current factors, `nxt` the one the next sweep writes.
	let mut cur_a: &mut [T] = a;
	let mut cur_b: &mut [T] = b;
	let mut nxt_a: &mut [T] = a2;
	let mut nxt_b: &mut [T] = b2;
	let mut in_primary = true;

	let mut sigma = sigma;
	let mut live = n;
	// The first sweep of a block has no shift: it supplies the factor information the shift needs.
	let mut have_sweep = false;
	// The sweep already reports the smallest factor it produced, which is the minimum over the
	// next iteration's factors. Carry it across iterations and rescan only when deflation
	// releases that entry.
	let mut dmin_cached: Option<T> = None;

	while live > 0 {
		// Deflate from the bottom: an off-diagonal that is negligible against the eigenvalue it
		// couples (`sigma + a`) or against the factor above it releases the bottom eigenvalue.
		while live >= 2 {
			let bb = cur_b[live - 2].clone();
			let eig = sigma.clone() + cur_a[live - 1].clone();
			if bb <= tol2.clone() * eig
				|| bb <= tol2.clone() * cur_a[live - 2].clone()
			{
				if dmin_cached.as_ref().is_some_and(|d| *d == cur_a[live - 1]) {
					dmin_cached = None;
				}
				let v = sigma.clone() + cur_a[live - 1].clone();
				cur_a[live - 1] = v.clone();
				nxt_a[live - 1] = v;
				live -= 1;
			} else {
				break;
			}
		}
		if live >= 3 {
			let bb = cur_b[live - 3].clone();
			let eig = sigma.clone()
				+ cur_a[live - 2].clone().fmin(cur_a[live - 1].clone());
			if bb <= tol2.clone() * eig
				|| bb <= tol2.clone() * cur_a[live - 3].clone()
			{
				if dmin_cached.as_ref().is_some_and(|d| {
					*d == cur_a[live - 1] || *d == cur_a[live - 2]
				}) {
					dmin_cached = None;
				}
				let (small, big) = two_by_two(
					cur_a[live - 2].clone(),
					cur_b[live - 2].clone(),
					cur_a[live - 1].clone(),
				);
				let (sm, bg) = (sigma.clone() + small, sigma.clone() + big);
				cur_a[live - 2] = bg.clone();
				nxt_a[live - 2] = bg;
				cur_a[live - 1] = sm.clone();
				nxt_a[live - 1] = sm;
				live -= 2;
				continue;
			}
		}
		if live == 1 {
			let v = sigma.clone() + cur_a[0].clone();
			cur_a[0] = v.clone();
			nxt_a[0] = v;
			break;
		}
		if live == 2 {
			// The trailing 2x2 of `B^T B`, in closed form, both roots.
			let (small, big) = two_by_two(
				cur_a[0].clone(),
				cur_b[0].clone(),
				cur_a[1].clone(),
			);
			let (s, g) = (sigma.clone() + small, sigma.clone() + big);
			cur_a[0] = g.clone();
			nxt_a[0] = g;
			cur_a[1] = s.clone();
			nxt_a[1] = s;
			break;
		}

		// The current factors are the pivots the last sweep produced (or the input), and the
		// smallest of them bounds the smallest eigenvalue of the shifted block from above.
		// only whether the smallest pivot is the bottom one is needed, not its index; `fmin`
		// keeps the scan branchless
		let dmin = match dmin_cached.take() {
			Some(d) => d,
			None => {
				let mut d = cur_a[0].clone();
				for x in &cur_a[1..live] {
					d = d.fmin(x.clone());
				}
				d
			},
		};
		let dmin_at_bottom = cur_a[live - 1] <= dmin;

		// The shift candidate.
		let mut tau = if !have_sweep {
			zero::<T>()
		} else {
			let (lam2, _) = two_by_two_shifted(
				cur_a[live - 2].clone() + cur_b[live - 3].clone(),
				cur_a[live - 2].clone() * cur_b[live - 2].clone(),
				cur_a[live - 1].clone() + cur_b[live - 2].clone(),
			);
			if dmin_at_bottom {
				// The bottom holds the smallest pivot: the trailing 2x2's smaller eigenvalue is a
				// close upper bound; back it off by its second-order coupling to the entry above,
				// weighted by the 2x2 eigenvector's component on that side.
				let d1 = cur_a[live - 2].clone() + cur_b[live - 3].clone();
				let f2 = cur_a[live - 2].clone() * cur_b[live - 2].clone();
				let coupling =
					cur_a[live - 3].clone() * cur_b[live - 3].clone();
				let gap = (cur_a[live - 3].clone()
					+ if live >= 4 {
						cur_b[live - 4].clone()
					} else {
						zero::<T>()
					}) - lam2.clone();
				let v1sq = {
					let dd = lam2.clone() - d1;
					let den = f2.clone() + dd.clone() * dd;
					if den > zero::<T>() {
						f2.clone() / den
					} else {
						from_f64::<T>(1.0)
					}
				};
				// Twice the second-order term: measured on random dense bidiagonals, the exact
				// term still overshoots on half the sweeps and twice it on 7%, with the same
				// sweep count per eigenvalue (3.5 at n = 512) as any larger factor.
				let back = if gap > zero::<T>() {
					two.clone() * coupling * v1sq / gap
				} else {
					lam2.clone()
				};
				let t =
					(lam2.clone() - back).fmax(zero::<T>()).fmin(dmin.clone());
				t.fmax(half.clone() * dmin.clone())
			} else {
				// The smallest pivot is interior: half of it, no more than the 2x2's eigenvalue.
				(half.clone() * dmin.clone()).fmin(lam2)
			}
		};
		tau = tau.fmax(zero::<T>());
		tau = tau.fmax(zero::<T>());

		let mut accepted = false;
		for attempt in 0..6 {
			let info = sweep(
				&cur_a[..live],
				&cur_b[..live - 1],
				tau.clone(),
				&mut nxt_a[..live],
				&mut nxt_b[..live - 1],
			);
			if info.ok {
				accepted = true;
				dmin_cached = Some(info.dmin.clone());
				break;
			}
			// The shift overshot. A failure at the last factor alone locates the bottom eigenvalue
			// at about `tau + dmin`; an interior failure says less, so the shift is cut hard.
			tau = if info.all_but_last_positive && attempt < 3 {
				((tau.clone() + info.dmin.clone())
					* (from_f64::<T>(1.0) - from_f64::<T>(4.0) * eps.clone()))
				.fmax(zero::<T>())
			} else if attempt < 3 {
				tau.clone() * quarter.clone()
			} else {
				zero()
			};
		}
		if !accepted {
			return Err(SvdError::NoConvergence);
		}
		core::mem::swap(&mut cur_a, &mut nxt_a);
		core::mem::swap(&mut cur_b, &mut nxt_b);
		in_primary = !in_primary;
		sigma += tau;
		have_sweep = true;
		sweeps_left = sweeps_left.saturating_sub(1);
		if sweeps_left == 0 {
			return Err(SvdError::NoConvergence);
		}
	}

	if !in_primary {
		// The results live in the caller's second buffer; the contract is the first.
		nxt_a[..n].clone_from_slice(&cur_a[..n]);
	}
	Ok(())
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
// a negated ordered comparison is the unordered-aware test the code wants; swapping it changes NaN behaviour
/// One dqds sweep from `(a, b)` into `(a2, b2)`.
fn sweep<T: RealField>(
	a: &[T],
	b: &[T],
	tau: T,
	a2: &mut [T],
	b2: &mut [T],
) -> Sweep<T> {
	let n = a.len();
	let mut delta = a[0].clone() - tau.clone();
	let mut dmin = delta.clone() + b[0].clone();
	for i in 0..n - 1 {
		let ahat = delta.clone() + b[i].clone();
		if !(ahat > zero()) || !ahat.is_finite() {
			return Sweep {
				dmin: ahat,
				all_but_last_positive: false,
				ok: false,
			};
		}
		if ahat < dmin {
			dmin = ahat.clone();
		}
		let t = a[i + 1].clone() / ahat.clone();
		b2[i] = b[i].clone() * t.clone();
		a2[i] = ahat;
		delta = delta.mul_add_impl(t, -tau.clone());
	}
	if !(delta > zero()) || !delta.is_finite() {
		return Sweep {
			dmin: delta,
			all_but_last_positive: true,
			ok: false,
		};
	}
	if delta < dmin {
		dmin = delta.clone();
	}
	a2[n - 1] = delta;
	Sweep {
		dmin,
		all_but_last_positive: true,
		ok: true,
	}
}

/// Both eigenvalues of the trailing 2x2 of `B^T B` for the factors `(a0, b0, a1)`: diagonal
/// `(a0, a1 + b0)`, off-diagonal `sqrt(a0 b0)`. Returned as `(smaller, larger)`.
fn two_by_two<T: RealField>(a0: T, b0: T, a1: T) -> (T, T) {
	two_by_two_shifted(a0.clone(), a0 * b0.clone(), a1 + b0)
}

/// Both eigenvalues of the symmetric 2x2 with diagonal `(d1, d2)` and squared off-diagonal `f2`,
/// the smaller one from the product of the roots so that it does not cancel.
fn two_by_two_shifted<T: RealField>(d1: T, f2: T, d2: T) -> (T, T) {
	let half = from_f64::<T>(0.5);
	let half_trace = (d1.clone() + d2.clone()) * half.clone();
	let half_diff = (d1.clone() - d2.clone()) * half;
	let disc = (half_diff.clone() * half_diff + f2.clone()).sqrt();
	let big = half_trace + disc;
	let det = d1 * d2 - f2;
	let small = if big > zero() {
		(det / big.clone()).fmax(zero::<T>())
	} else {
		zero::<T>()
	};
	(small, big)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::Mat;
	use dyn_stack::MemBuffer;

	/// `singular_values` with its scratch from a fresh buffer
	fn values<T: RealField>(d: &mut [T], e: &mut [T]) -> Result<(), SvdError> {
		singular_values(
			d,
			e,
			MemStack::new(&mut MemBuffer::new(singular_values_scratch::<T>(
				d.len(),
			))),
		)
	}

	#[allow(clippy::needless_range_loop)]
	// index loop kept as written: the body indexes parallel buffers and strided views
	/// The kernel against `thin_svd` of the same bidiagonal, over spectra that make the iteration
	/// work: well separated, clustered, repeated, badly scaled, and with negative diagonals.
	#[test]
	fn matches_the_reference_on_bidiagonals() {
		let mut state: u64 = 0x1234_5678_9ABC_DEF0;
		let mut next = move || {
			state ^= state >> 12;
			state ^= state << 25;
			state ^= state >> 27;
			let x = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
			(x >> 11) as f64 / (1u64 << 53) as f64
		};
		for n in 2..41usize {
			for case in 0..4 {
				let mut d = vec![0.0f64; n];
				let mut e = vec![0.0f64; n - 1];
				for i in 0..n {
					d[i] = match case {
						0 => 0.5 + next(),
						1 => {
							if i % 3 == 0 {
								-1.0
							} else {
								0.5 + next()
							}
						},
						2 => 1.0,
						_ => (0.5 + next()) * if i == 0 { 1e-6 } else { 1.0 },
					};
				}
				for i in 0..n - 1 {
					e[i] = match case {
						0 => 0.2 * next(),
						1 => 0.2 * next(),
						2 => 1e-3 * next(),
						_ => 0.2 * next(),
					};
				}
				let mut mat = Mat::<f64>::zeros(n, n);
				for i in 0..n {
					mat[(i, i)] = d[i];
					if i + 1 < n {
						mat[(i + 1, i)] = e[i];
					}
				}
				let mut want: Vec<f64> = mat
					.as_ref()
					.thin_svd(Par::Seq)
					.unwrap()
					.S()
					.column_vector()
					.iter()
					.copied()
					.collect();
				want.sort_by(|x, y| y.partial_cmp(x).unwrap());

				let mut dd = d.clone();
				let mut ee = e.clone();
				values(&mut dd, &mut ee).unwrap();
				for i in 0..n {
					let tol = 1e-10 * want[0].max(1.0);
					core::assert!(
						(dd[i] - want[i]).abs() <= tol,
						"n={n} case={case} i={i}: {} vs {}",
						dd[i],
						want[i]
					);
				}
			}
		}
	}

	/// The bidiagonal of a random dense matrix, the shape the campaign measures: every singular
	/// value agrees with the divide-and-conquer path to a relative 1e-12.
	#[test]
	fn matches_the_reference_on_a_dense_bidiagonal() {
		use crate::linalg::svd::bidiag;
		use crate::{Mat, Par};
		use dyn_stack::{MemBuffer, MemStack};
		let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
		let mut next = move || {
			state ^= state >> 12;
			state ^= state << 25;
			state ^= state >> 27;
			let x = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
			(x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
		};
		for n in [3usize, 17, 64, 200] {
			let a = Mat::<f64>::from_fn(n, n, |_, _| next());
			let mut bid = a.clone();
			let mut hl = Mat::<f64>::zeros(1, n);
			let mut hr = Mat::<f64>::zeros(1, n - 1);
			let mut mem =
				MemBuffer::new(bidiag::bidiag_in_place_scratch::<f64>(
					n,
					n,
					Par::Seq.shape(),
					Default::default(),
				));
			bidiag::bidiag_in_place(
				bid.as_mut(),
				hl.as_mut(),
				hr.as_mut(),
				Par::Seq,
				MemStack::new(&mut mem),
				Default::default(),
			);
			let mut d: Vec<f64> = (0..n).map(|i| bid[(i, i)]).collect();
			let mut e: Vec<f64> = (0..n - 1).map(|i| bid[(i, i + 1)]).collect();
			values(&mut d, &mut e).unwrap();
			let want: Vec<f64> = a
				.as_ref()
				.thin_svd(Par::Seq)
				.unwrap()
				.S()
				.column_vector()
				.iter()
				.copied()
				.collect();
			for i in 0..n {
				core::assert!(
					(d[i] - want[i]).abs() <= 1e-12 * want[i],
					"n={n} i={i}: {} vs {}",
					d[i],
					want[i]
				);
			}
		}
	}

	/// A diagonal bidiagonal is its own answer, sorted.
	#[test]
	fn diagonal_is_its_own_answer() {
		let mut d = vec![3.0f64, 1.0, 2.0];
		let mut e: Vec<f64> = vec![0.0; 2];
		values(&mut d, &mut e).unwrap();
		core::assert_eq!(d, vec![3.0, 2.0, 1.0]);
	}

	/// Scaling by a power of two must not change the answer's relative accuracy.
	#[test]
	fn scale_does_not_change_the_answer() {
		let d = vec![1.0f64, 2.0, 3.0, 4.0];
		let e = vec![0.3f64, 0.2, 0.1];
		let mut d1 = d.clone();
		let mut e1 = e.clone();
		values(&mut d1, &mut e1).unwrap();

		let mut d2: Vec<f64> = d.iter().map(|x| x * 1e150).collect();
		let mut e2: Vec<f64> = e.iter().map(|x| x * 1e150).collect();
		values(&mut d2, &mut e2).unwrap();
		for i in 0..d1.len() {
			let got = d2[i] * 1e-150;
			core::assert!(
				(got - d1[i]).abs() <= 1e-10 * d1[0],
				"{got} vs {}",
				d1[i]
			);
		}
	}

	/// #209: the smaller singular value of `B = [[small, 0.25], [0, 1]]` came back `Ok` with 2.09 %
	/// (f32) and 0.13 % (f64) relative error once its squared factor was subnormal
	/// (`1e-22^2 = 1e-44` at f32, `1e-161^2 = 1e-322` at f64): a subnormal keeps fewer mantissa
	/// bits than `eps`, so the relative accuracy is gone before the first sweep. Such an input
	/// reports non-convergence and leaves the diagonal bit-identical, and the caller recovers on its
	/// rotation path (#110) or with its own routine (#206's adapter). The issue's nearby
	/// normal-range controls must keep full accuracy.
	#[test]
	fn a_subnormal_squared_factor_is_not_a_success() {
		// f32: 1e-22^2 = 1e-44, below the smallest normal 1.18e-38.
		let mut d = [1e-22f32, 1.0];
		let original = d;
		let mut e = [0.25f32];
		core::assert_eq!(values(&mut d, &mut e), Err(SvdError::NoConvergence));
		core::assert_eq!(d.map(f32::to_bits), original.map(f32::to_bits));

		// f64: 1e-161^2 = 1e-322, below the smallest normal 2.2e-308.
		let mut d = [1e-161f64, 1.0];
		let original = d;
		let mut e = [0.25f64];
		core::assert_eq!(values(&mut d, &mut e), Err(SvdError::NoConvergence));
		core::assert_eq!(d.map(f64::to_bits), original.map(f64::to_bits));

		// Controls: 1e-10^2 = 1e-20 (f32) and 1e-100^2 = 1e-200 (f64) are normal. The expected
		// small value is `small / hypot(1, 0.25)`; the constant keeps `hypot` out of the test.
		let mut d = [1e-10f32, 1.0];
		let mut e = [0.25f32];
		values(&mut d, &mut e).unwrap();
		let got = d[1] / (1e-10f32 * 0.970_142_5);
		core::assert!(
			(got - 1.0).abs() <= 20.0 * f32::EPSILON,
			"f32: {:e}",
			d[1]
		);

		let mut d = [1e-100f64, 1.0];
		let mut e = [0.25f64];
		values(&mut d, &mut e).unwrap();
		let got = d[1] / (1e-100 * 0.970_142_500_145_331_9);
		core::assert!(
			(got - 1.0).abs() <= 20.0 * f64::EPSILON,
			"f64: {:e}",
			d[1]
		);
	}

	/// The rejection above is a ratio, not a magnitude: a uniformly tiny bidiagonal normalizes to the
	/// same factors and stays `Ok`, and a subnormal squared *off-diagonal* is not a failure while its
	/// neighbours are in range, because the split below absorbs it.
	#[test]
	fn the_squared_range_check_is_a_ratio_and_spares_off_diagonals() {
		// `[[1, 0.25], [0, 1]]` and the same matrix scaled to `1e-30` / `1e-300` answer alike, though
		// the tiny copies sit far below `sqrt(min_positive)`. Dividing by the stride the kernel
		// itself normalizes by keeps the comparison free of hand-computed constants.
		let mut unit = [1.0f32, 1.0];
		let mut unit_e = [0.25f32];
		values(&mut unit, &mut unit_e).unwrap();
		let mut tiny = [1e-30f32, 1e-30];
		let mut tiny_e = [2.5e-31f32];
		values(&mut tiny, &mut tiny_e).unwrap();
		for (got, want) in tiny.iter().zip(unit.iter()) {
			let got = got / 1e-30f32;
			core::assert!(
				(got / want - 1.0).abs() <= 20.0 * f32::EPSILON,
				"f32: {got:e} vs {want:e}"
			);
		}

		let mut unit = [1.0f64, 1.0];
		let mut unit_e = [0.25f64];
		values(&mut unit, &mut unit_e).unwrap();
		let mut tiny = [1e-300f64, 1e-300];
		let mut tiny_e = [2.5e-301f64];
		values(&mut tiny, &mut tiny_e).unwrap();
		for (got, want) in tiny.iter().zip(unit.iter()) {
			let got = got / 1e-300f64;
			core::assert!(
				(got / want - 1.0).abs() <= 20.0 * f64::EPSILON,
				"f64: {got:e} vs {want:e}"
			);
		}

		// 1e-22^2 is subnormal at f32, but a diagonal of one never releases a smaller value than
		// one, so nothing is lost.
		let mut d = [1.0f32, 1.0];
		let mut e = [1e-22f32];
		values(&mut d, &mut e).unwrap();
		for got in d {
			core::assert!(
				(got - 1.0).abs() <= 20.0 * f32::EPSILON,
				"f32: {got:e}"
			);
		}
	}
}
