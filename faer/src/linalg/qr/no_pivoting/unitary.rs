//! the unitary factor $Q$ of a packed $QR$ factorization, and its orthonormal
//! complement (the left null space of the factored matrix).
//!
//! The output conventions (the full $Q$ being the $m \times m$ unitary, the
//! complement being its trailing columns, and the positive diagonal gauge) follow
//! MatrixAlgebraKit.jl `src/implementations/qr.jl` and `src/common/gauge.jl` (MIT,
//! Copyright (c) QuantumKitHub contributors). They are the same conventions as
//! tenferro-linalg's `QrGauge::PositiveDiagonal`, applied to the factors here.
use crate::assert;
use crate::internal_prelude::*;
/// Applies Q to an already initialized identity block with seed rows starting
/// at `offset`. Reverse panels cannot affect columns whose seeds precede them.
#[track_caller]
pub(crate) fn apply_to_identity<T: ComplexField>(
	mut out: MatMut<'_, T>,
	offset: usize,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	mut par: Par,
	mut stack: &mut MemStack,
) {
	let block_size = Q_coeff.nrows();
	assert!(all(
		block_size > 0,
		Q_coeff.ncols() == Ord::min(Q_basis.nrows(), Q_basis.ncols())
	));
	let mut end = Q_coeff.ncols();
	while end > 0 {
		let start = (end - 1) / block_size * block_size;
		let first_col = start.saturating_sub(offset).min(out.ncols());
		linalg::householder::apply_block_householder_on_the_left_in_place_with_conj(
			Q_basis.get(start.., start..end),
			Q_coeff.get(..end - start, start..end),
			Conj::No,
			out.rb_mut().get_mut(start.., first_col..),
			par.reborrow(),
			stack.rb_mut(),
		);
		end = start;
	}
}
/// starts `out` as the identity block at rows `offset..offset + q` and applies
/// the $Q$ sequence to it from the left.
///
/// Shared with the LQ module, which passes its transposed view as both the basis
/// and the target.
#[track_caller]
pub(crate) fn apply_to_start_block<T: ComplexField>(
	out: MatMut<'_, T>,
	offset: usize,
	q: usize,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let mut out = out;
	out.rb_mut().fill(zero());
	for j in 0..q {
		out[(offset + j, j)] = one();
	}
	apply_to_identity(out, offset, Q_basis, Q_coeff, par, stack);
}
/// the workspace required by [`unitary`] for an `out` of shape `nrows x out_ncols`.
pub fn unitary_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	out_ncols: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	let _ = ncols;
	let _ = par;
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(nrows, block_size, out_ncols)
}
/// builds the unitary factor $Q$ of the packed factorization in `Q_basis`.
///
/// `Q_basis` is the packed factorization or its first $\min(m, n)$ columns.
///
/// `out` is $m \times \min(m, n)$ for the thin $Q$ or $m \times m$ for the full
/// $Q$. With `positive`, column $j$ of $Q$ is scaled by $d_j / |d_j|$, where
/// $d_j$ is the diagonal of $R$, matching the
/// [`copy_r`](crate::linalg::qr::no_pivoting::factor::copy_r) phase so that the
/// product $QR$ is unchanged and the diagonal of $R$ is real and nonnegative.
#[track_caller]
pub fn unitary<T: ComplexField>(
	out: MatMut<'_, T>,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	positive: bool,
	par: Par,
	stack: &mut MemStack,
) {
	let (m, n) = Q_basis.shape();
	let k = Ord::min(m, n);
	let q = out.ncols();
	let mut out = out;
	assert!(all(out.nrows() == m, q == k || q == m));
	// the sequence is read from the first `k` columns of the packed matrix, as
	// `qr_in_place` leaves it, so a caller that has not truncated `R` away passes
	// the whole matrix.
	let basis = Q_basis.get(.., ..k);
	apply_to_start_block(out.rb_mut(), 0, q, basis, Q_coeff, par, stack);
	if positive {
		for j in 0..k {
			let phase = linalg::positive_phase(Q_basis[(j, j)].clone());
			let col = out.rb_mut().col_mut(j);
			z!(col).for_each(|unzip!(x)| *x *= &phase);
		}
	}
}
/// the workspace required by [`null`].
pub fn null_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	block_size: usize,
	par: ParShape,
) -> StackReq {
	let k = Ord::min(nrows, ncols);
	let _ = par;
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(nrows, block_size, nrows - k)
}
/// builds orthonormal columns orthogonal to the column space determined by the
/// factorization: the columns $\min(m, n) + 1, \dots, m$ of the full $Q$, so
/// that `out` is $m \times (m - \min(m, n))$ with orthonormal columns and
/// $A^H out = 0$.
///
/// these columns are a basis of the left null space of $A$ when
/// $\operatorname{rank}(A) = \min(m, n)$. with rank loss the factorization
/// determines a $\min(m, n)$-dimensional column space, so the complement is that
/// of the determined subspace rather than of the column space of $A$.
#[track_caller]
pub fn null<T: ComplexField>(
	out: MatMut<'_, T>,
	Q_basis: MatRef<'_, T>,
	Q_coeff: MatRef<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	let (m, n) = Q_basis.shape();
	let k = Ord::min(m, n);
	assert!(all(out.nrows() == m, out.ncols() == m - k));
	let basis = Q_basis.get(.., ..k);
	apply_to_start_block(out, k, m - k, basis, Q_coeff, par, stack);
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::assert;
	use crate::linalg::qr::no_pivoting::factor;
	use crate::stats::prelude::*;
	use crate::utils::approx::*;
	use crate::{Mat, c64};
	use dyn_stack::MemBuffer;
	fn packed<T: ComplexField>(
		a: MatRef<'_, T>,
		block_size: usize,
	) -> (Mat<T>, Mat<T>, usize) {
		let (m, n) = a.shape();
		let k = Ord::min(m, n);
		let mut packed = a.cloned();
		let mut coeff = Mat::full(block_size, k, infinity());
		let rank = factor::qr_in_place(
			packed.as_mut(),
			coeff.as_mut(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				factor::qr_in_place_scratch::<T>(
					m,
					n,
					block_size,
					ParShape::Seq,
					default(),
				),
			)),
			default(),
		)
		.rank;
		(packed, coeff, rank)
	}
	fn check(a: MatRef<'_, c64>, block_size: usize) {
		let (m, n) = a.shape();
		let k = Ord::min(m, n);
		let (packed, coeff, rank) = packed(a, block_size);
		let approx_eq = CwiseMat(ApproxEq::eps() * (Ord::max(m, n) as f64));
		let basis = packed.as_ref().get(.., ..k);
		for positive in [false, true] {
			let mut r = Mat::<c64>::zeros(k, n);
			factor::copy_r(r.as_mut(), packed.as_ref(), positive);
			if positive {
				// the phase is exact; the residue is the rounding of $d / |d|$.
				for j in 0..k {
					let d = r[(j, j)];
					assert!(
						d.re >= 0.0 && d.im.abs() <= 1e-14 * d.abs().max(1.0)
					);
				}
			}
			let mut q = Mat::<c64>::zeros(m, k);
			unitary(
				q.as_mut(),
				basis,
				coeff.as_ref(),
				positive,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(unitary_scratch::<c64>(
					m,
					n,
					k,
					block_size,
					ParShape::Seq,
				))),
			);
			assert!((&q * &r) ~ a);
			assert!((q.adjoint() * &q) ~ Mat::<c64>::identity(k, k));
			// the whole packed matrix is accepted too: the sequence is read from its
			// first `k` columns, so a caller that keeps `R` in place need not slice.
			let mut q_full = Mat::<c64>::zeros(m, m);
			unitary(
				q_full.as_mut(),
				packed.as_ref(),
				coeff.as_ref(),
				positive,
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(unitary_scratch::<c64>(
					m,
					n,
					m,
					block_size,
					ParShape::Seq,
				))),
			);
			assert!(q_full.get(.., ..k) ~ q.as_ref());
			assert!((q_full.adjoint() * &q_full) ~ Mat::<c64>::identity(m, m));
			let mut complement = Mat::<c64>::zeros(m, m - k);
			null(
				complement.as_mut(),
				basis,
				coeff.as_ref(),
				Par::Seq,
				MemStack::new(&mut MemBuffer::new(null_scratch::<c64>(
					m,
					n,
					block_size,
					ParShape::Seq,
				))),
			);
			assert!(q_full.get(.., k..) ~ complement.as_ref());
			assert!((complement.adjoint() * &complement) ~ Mat::<c64>::identity(m - k, m - k));
			// $A^H N = R^H (Q^H N) = 0$ for every rank.
			assert!((a.cloned().adjoint() * &complement) ~ Mat::<c64>::zeros(n, m - k));
		}
		assert!(rank <= k);
	}
	fn identity_seed_case<T: ComplexField>(
		make: fn(f64, f64) -> T,
		abs: fn(&T) -> f64,
		tol: f64,
	) {
		for (m, n) in [
			(0, 3),
			(3, 0),
			(1, 1),
			(37, 19),
			(19, 37),
			(21, 21),
			(257, 129),
		] {
			for zero_input in [false, true] {
				let a = Mat::from_fn(m, n, |i, j| {
					if zero_input {
						make(0.0, 0.0)
					} else {
						make(
							((13 * i + 19 * j + 1) as f64).sin(),
							((7 * i + 3 * j + 2) as f64).cos(),
						)
					}
				});
				let k = m.min(n);
				let (packed, coeff, _) = packed(a.as_ref(), 8);
				for (offset, q) in [(0, k), (0, m), (k, m - k)] {
					let mut reference = Mat::from_fn(m, q, |i, j| {
						if i == offset + j { one() } else { zero() }
					});
					let mut mem = MemBuffer::new(unitary_scratch::<T>(
						m,
						n,
						q,
						8,
						ParShape::Seq,
					));
					linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
						packed.as_ref().get(.., ..k), coeff.as_ref(), Conj::No,
						reference.as_mut(), Par::Seq, MemStack::new(&mut mem),
					);
					for row_major in [false, true] {
						let mut storage = Mat::zeros(
							if row_major { q } else { m },
							if row_major { m } else { q },
						);
						let mut output = if row_major {
							storage.transpose_mut()
						} else {
							storage.as_mut()
						};
						apply_to_start_block(
							output.rb_mut(),
							offset,
							q,
							packed.as_ref().get(.., ..k),
							coeff.as_ref(),
							Par::Seq,
							MemStack::new(&mut mem),
						);
						let got = output.rb();
						let diff = got - &reference;
						for j in 0..q {
							for i in 0..m {
								assert!(
									abs(&diff[(i, j)]) <= tol * m.max(1) as f64
								);
							}
						}
						let orth =
							got.adjoint() * got - Mat::<T>::identity(q, q);
						for j in 0..q {
							for i in 0..q {
								assert!(
									abs(&orth[(i, j)]) <= tol * m.max(1) as f64
								);
							}
						}
						if offset == 0 && q == k {
							let mut r = Mat::zeros(k, n);
							factor::copy_r(r.as_mut(), packed.as_ref(), false);
							let diff = got * r.as_ref() - &a;
							let mut err = 0.0f64;
							let mut scale = 0.0f64;
							for j in 0..n {
								for i in 0..m {
									err += abs(&diff[(i, j)]).powi(2);
									scale += abs(&a[(i, j)]).powi(2);
								}
							}
							assert!(
								err.sqrt()
									<= tol
										* m.max(n).max(1) as f64 * scale
										.sqrt()
										.max(1.0)
							);
						}
					}
				}
			}
		}
	}
	#[test]
	fn test_identity_seed_four_scalars() {
		identity_seed_case::<f64>(|r, _| r, |x| x.abs(), 2e-14);
		identity_seed_case::<f32>(|r, _| r as f32, |x| x.abs() as f64, 1e-5);
		identity_seed_case::<crate::c64>(crate::c64::new, |x| x.norm(), 2e-14);
		identity_seed_case::<crate::c32>(
			|r, i| crate::c32::new(r as f32, i as f32),
			|x| x.norm() as f64,
			1e-5,
		);
	}

	#[test]
	fn test_unitary_and_null() {
		let rng = &mut StdRng::seed_from_u64(3);
		for (m, n) in [(7, 4), (4, 7), (5, 5), (0, 3), (3, 0), (0, 0), (96, 64)]
		{
			let a = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			for block_size in [1, Ord::min(m, n).clamp(1, 8)] {
				check(a.as_ref(), block_size);
			}
		}
	}
	#[test]
	fn test_rank_deficient() {
		let rng = &mut StdRng::seed_from_u64(4);
		for (m, n, r) in [(6, 4, 2), (4, 6, 3), (5, 5, 1)] {
			let mut a = CwiseMatDistribution {
				nrows: m,
				ncols: n,
				dist: ComplexDistribution::new(StandardNormal, StandardNormal),
			}
			.rand::<Mat<c64>>(rng);
			for i in r..m {
				for j in 0..n {
					a[(i, j)] = c64::new(0.0, 0.0);
				}
			}
			let (_, _, rank) = packed(a.as_ref(), Ord::min(m, n));
			assert!(rank == r);
			check(a.as_ref(), Ord::min(m, n).clamp(1, 4));
		}
	}
}
