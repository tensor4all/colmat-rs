// Copyright (c) tensor4all contributors
// SPDX-License-Identifier: MIT
//
// Provenance: written for colmat (tensor4all/colmat-rs#60); no code is translated from another
// project.
//
// - The API follows the conventions of MatrixAlgebraKit.jl (`left_polar(A) -> W, P` with
//   `A = W P` for `m >= n`, `right_polar(A) -> P, Wᴴ` with `A = P Wᴴ` for `m <= n`).
// - `PolarMethod::Svd` is the textbook construction $W = U V^H$, $P = V \Sigma V^H$ from the thin
//   SVD $A = U \Sigma V^H$.
// - `PolarMethod::Qdwh` implements the QR-based dynamically weighted Halley iteration from the
//   papers, not from code: Y. Nakatsukasa, Z. Bai and F. Gygi, "Optimizing Halley's iteration for
//   computing the matrix polar decomposition", SIAM J. Matrix Anal. Appl. 31(5), 2700-2720 (2010)
//   (the weights, the QR form of the step and the stopping rule); Y. Nakatsukasa and N. J. Higham,
//   "Backward stability of iterations for computing the polar decomposition", SIAM J. Matrix Anal.
//   Appl. 33(2), 460-479 (2012) (column pivoting in the QR form); and Y. Nakatsukasa and
//   N. J. Higham, "Stable and efficient spectral divide and conquer algorithms for the symmetric
//   eigenvalue decomposition and the SVD", SIAM J. Sci. Comput. 35(3), A1325-A1349 (2013) (the
//   Cholesky form of the step once $c_k \le 100$). MatrixFactorizations.jl's `polar.jl` (MIT,
//   Copyright (c) 2019 JuliaLinearAlgebra, based on PolarFact by Weijian Zhang) is only the test
//   oracle; its QDWH code was not read.
//! the polar decomposition of a matrix $A$ of shape $(m, n)$.
//!
//! - left, $m \ge n$: $A = W P$, where $W$ is $m \times n$ with orthonormal columns and $P$ is
//!   $n \times n$ hermitian positive semidefinite ([`left_polar`]).
//! - right, $m \le n$: $A = P W^H$, where $P$ is $m \times m$ hermitian positive semidefinite and
//!   $W^H$ is $m \times n$ with orthonormal rows ([`right_polar`]).
//!
//! $P$ is unique; $W$ is unique when $A$ has full rank. the right decomposition is the left one of
//! $A^\top$, read transposed.
//!
//! two methods compute it ([`PolarMethod`]): the thin SVD (the default and the reference), and
//! the QDWH iteration, which uses $QR$, Cholesky and matrix products only.
use crate::assert;
use crate::internal_prelude::*;
use linalg::matmul::triangular::BlockStructure;
use linalg::svd::{ComputeSvdVectors, SvdError};

/// the method that computes the polar decomposition
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PolarMethod {
	/// from the thin SVD $A = U \Sigma V^H$: $W = U V^H$ and $P = V \Sigma V^H$.
	///
	/// $W$ has orthonormal columns for every $A$, including a rank-deficient one.
	Svd,
	/// the QR-based dynamically weighted Halley iteration (QDWH, Nakatsukasa, Bai and Gygi 2010),
	/// built from $QR$, Cholesky and matrix products.
	///
	/// a tall $A$ is first reduced to its $n \times n$ triangular factor $R$. $X_0 = R / \|R\|_F$
	/// is iterated with the weights for the lower bound $\ell_0 = u$ (the unit roundoff) on its
	/// smallest singular value, which takes six iterations in `f64` and four in `f32` whatever the
	/// condition of $A$. the step is the $QR$ form (column-pivoted $QR$ of $[\sqrt{c_k} X_k; I]$)
	/// while its weight $c_k > 100$ and the Cholesky form after that; the iteration stops when $\|X_{k+1} - X_k\|_F \le (5u)^{1/3}$ and
	/// $1 - \ell_{k+1} \le 10u$, and $P$ is the hermitian part of $W^H A$.
	///
	/// $W$ has orthonormal columns when the smallest singular value of $A$ is at least
	/// $u \|A\|_F$. below that the iteration does not reach the null space of $A$: $A = W P$
	/// still holds and $P$ is still positive semidefinite, but the columns of $W$ there are not
	/// orthonormal (they are zero for an exactly zero column of $A$).
	Qdwh,
}

/// polar decomposition error
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PolarError {
	/// the SVD did not converge, the QDWH iteration did not meet its stopping rule within its
	/// iteration limit, or $P$ is not representable. reached with non-finite input, or with finite
	/// input whose $P$ is not representable in `T`. the SVD method keeps the singular values at
	/// the SVD's working scale, so a largest singular value beyond the largest finite value is
	/// not an error by itself
	NoConvergence,
}

/// polar decomposition tuning parameters
#[derive(Copy, Clone, Debug)]
pub struct PolarParams {
	/// the method, [`PolarMethod::Svd`] by default
	pub method: PolarMethod,
	#[doc(hidden)]
	pub non_exhaustive: NonExhaustive,
}

impl<T: ComplexField> Auto<T> for PolarParams {
	fn auto() -> Self {
		Self {
			method: PolarMethod::Svd,
			non_exhaustive: NonExhaustive(()),
		}
	}
}

/// QDWH iterations before [`PolarError::NoConvergence`]; finite input stops after at most six
/// (`f64`) or four (`f32`), plus one when the last step still moved $X$ by more than
/// $(5u)^{1/3}$
const QDWH_MAX_ITERATIONS: usize = 20;

/// computes the size and alignment of the workspace required to compute the left polar
/// decomposition of an `nrows x ncols` matrix, `nrows >= ncols`
pub fn left_polar_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	par: ParShape,
	params: Spec<PolarParams, T>,
) -> StackReq {
	let (m, n) = (nrows, ncols);
	assert!(m >= n);
	match params.config.method {
		PolarMethod::Svd => StackReq::all_of(&[
			temp_mat_scratch::<T>(n, 1),
			temp_mat_scratch::<T>(m, n),
			temp_mat_scratch::<T>(n, n),
			linalg::svd::svd_scratch::<T>(
				m,
				n,
				ComputeSvdVectors::Thin,
				ComputeSvdVectors::Thin,
				par,
				default(),
			),
		]),
		PolarMethod::Qdwh => qdwh_scratch::<T>(m, n, par),
	}
}

/// computes the size and alignment of the workspace required to compute the right polar
/// decomposition of an `nrows x ncols` matrix, `nrows <= ncols`
pub fn right_polar_scratch<T: ComplexField>(
	nrows: usize,
	ncols: usize,
	par: ParShape,
	params: Spec<PolarParams, T>,
) -> StackReq {
	assert!(nrows <= ncols);
	left_polar_scratch::<T>(ncols, nrows, par, params)
}

/// computes the left polar decomposition $A = W P$ of an $m \times n$ matrix, $m \ge n$.
///
/// $W$ ($m \times n$) gets the factor with orthonormal columns and $P$ ($n \times n$) the
/// hermitian positive semidefinite factor, stored in full (both triangles, real diagonal, exactly
/// hermitian). the previous contents of $W$ and $P$ are not read. on error their contents are
/// unspecified.
///
/// # panics
///
/// panics if $m < n$, or if $W$ or $P$ does not have the shape above.
#[track_caller]
pub fn left_polar<T: ComplexField>(
	A: MatRef<'_, T>,
	W: MatMut<'_, T>,
	P: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<PolarParams, T>,
) -> Result<(), PolarError> {
	let (m, n) = A.shape();
	assert!(all(
		m >= n,
		W.nrows() == m,
		W.ncols() == n,
		P.nrows() == n,
		P.ncols() == n,
	));
	let mut P = P;
	match params.config.method {
		PolarMethod::Svd => svd_polar(A, W, P.rb_mut(), par, stack),
		PolarMethod::Qdwh => qdwh_polar(A, W, P.rb_mut(), par, stack),
	}?;
	// finite input can still have a $P$ beyond the largest finite value (QDWH scales it back up)
	if !is_finite(&P.norm_max()) {
		return Err(PolarError::NoConvergence);
	}
	Ok(())
}

/// computes the right polar decomposition $A = P W^H$ of an $m \times n$ matrix, $m \le n$.
///
/// $P$ ($m \times m$) gets the hermitian positive semidefinite factor, stored in full, and `Wh`
/// ($m \times n$) the factor $W^H$ with orthonormal rows. the outputs and errors behave as in
/// [`left_polar`], which computes this as the left decomposition of $A^\top$.
///
/// # panics
///
/// panics if $m > n$, or if $P$ or `Wh` does not have the shape above.
#[track_caller]
pub fn right_polar<T: ComplexField>(
	A: MatRef<'_, T>,
	P: MatMut<'_, T>,
	Wh: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
	params: Spec<PolarParams, T>,
) -> Result<(), PolarError> {
	let (m, n) = A.shape();
	assert!(all(
		m <= n,
		P.nrows() == m,
		P.ncols() == m,
		Wh.nrows() == m,
		Wh.ncols() == n,
	));
	// $A^\top = W' P'$ gives $A = P'^\top W'^\top$, and $P'^\top = \bar P'$ is hermitian positive
	// semidefinite
	left_polar(
		A.transpose(),
		Wh.transpose_mut(),
		P.transpose_mut(),
		par,
		stack,
		params,
	)
}

fn svd_polar<T: ComplexField>(
	A: MatRef<'_, T>,
	W: MatMut<'_, T>,
	P: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) -> Result<(), PolarError> {
	let (m, n) = A.shape();
	let mut par = par;
	let mut P = P;
	let (mut s, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, 1, stack) };
	let mut s = s.as_mat_mut();
	let (mut U, stack) = unsafe { temp_mat_uninit::<T, _, _>(m, n, stack) };
	let mut U = U.as_mat_mut();
	let (mut V, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut V = V.as_mat_mut();
	// $\Sigma$ stays at the SVD's working scale: $P$'s entries can be representable when
	// $\sigma_{\max}$ is not (#233), and `left_polar` rejects a $P$ that is not
	let scale = linalg::svd::svd_scaled(
		A,
		s.rb_mut().col_mut(0).as_diagonal_mut(),
		Some(U.rb_mut()),
		Some(V.rb_mut()),
		par.reborrow(),
		stack,
		default(),
	)
	.map_err(|SvdError::NoConvergence| PolarError::NoConvergence)?;
	linalg::matmul::matmul(
		W,
		Accum::Replace,
		U.rb(),
		V.rb().adjoint(),
		one(),
		par.reborrow(),
	);
	// $V \Sigma$ in the leading block of $U$, which is no longer needed
	let mut VS = U.rb_mut().subrows_mut(0, n);
	for j in 0..n {
		let sigma = real(&s[(j, 0)]);
		z!(VS.rb_mut().col_mut(j), V.rb().col(j))
			.for_each(|uz!(dst, src)| *dst = mul_real(src, &sigma));
	}
	linalg::matmul::matmul(
		P.rb_mut(),
		Accum::Replace,
		VS.rb(),
		V.rb().adjoint(),
		one(),
		par,
	);
	hermitian_part_in_place(P.rb_mut());
	if let Some(scale) = scale {
		let inv = scale.recip();
		z!(P).for_each(|uz!(x)| *x = x.mul_pow2(&inv));
	}
	Ok(())
}

/// $P \gets (P + P^H) / 2$, exactly hermitian
fn hermitian_part_in_place<T: ComplexField>(P: MatMut<'_, T>) {
	let mut P = P;
	let half = from_f64::<T::Real>(0.5);
	for j in 0..P.ncols() {
		for i in 0..j {
			// halve before adding: entries near the largest finite value must not overflow
			let value = add(
				&mul_real(&P[(i, j)], &half),
				&mul_real(&conj(&P[(j, i)]), &half),
			);
			P[(j, i)] = conj(&value);
			P[(i, j)] = value;
		}
		P[(j, j)] = from_real(&real(&P[(j, j)]));
	}
}

fn qdwh_scratch<T: ComplexField>(
	m: usize,
	n: usize,
	par: ParShape,
) -> StackReq {
	if n == 0 {
		return StackReq::EMPTY;
	}
	let iteration = qdwh_iteration_scratch::<T>(n, par);
	if m == n {
		return iteration;
	}
	let bs = linalg::qr::no_pivoting::factor::recommended_block_size::<T>(m, n);
	StackReq::all_of(&[
		temp_mat_scratch::<T>(m, n),
		temp_mat_scratch::<T>(bs, n),
		StackReq::any_of(&[
			linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
				m,
				n,
				bs,
				par,
				default(),
			),
			iteration,
			linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
				m, bs, n,
			),
		]),
	])
}

fn qdwh_iteration_scratch<T: ComplexField>(
	n: usize,
	par: ParShape,
) -> StackReq {
	let bs =
		linalg::qr::no_pivoting::factor::recommended_block_size::<T>(2 * n, n);
	let qr_step = StackReq::all_of(&[
		temp_mat_scratch::<T>(2 * n, n),
		temp_mat_scratch::<T>(bs, n),
		StackReq::new::<usize>(n).array(2),
		StackReq::any_of(&[
			linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, T>(
				2 * n,
				n,
				bs,
				par,
				default(),
			),
			StackReq::all_of(&[
				temp_mat_scratch::<T>(2 * n, n),
				linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
					2 * n, bs, n,
				),
			]),
		]),
	]);
	let cholesky_step = StackReq::all_of(&[
		temp_mat_scratch::<T>(n, n),
		StackReq::any_of(&[
			linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T>(
				n,
				par,
				default(),
			),
			linalg::cholesky::llt::solve::solve_in_place_scratch::<T>(
				n, n, par,
			),
		]),
	]);
	StackReq::all_of(&[
		temp_mat_scratch::<T>(n, n),
		StackReq::any_of(&[qr_step, cholesky_step]),
	])
}

fn qdwh_polar<T: ComplexField>(
	A: MatRef<'_, T>,
	W: MatMut<'_, T>,
	P: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) -> Result<(), PolarError> {
	let (m, n) = A.shape();
	if n == 0 {
		return Ok(());
	}
	if m == n {
		return qdwh_iterate(A, BlockStructure::Rectangular, W, P, par, stack);
	}
	// $A = Q R$, so $W = Q W_R$ and $P = P_R$ from the polar decomposition $R = W_R P_R$ of the
	// $n \times n$ factor
	let mut W = W;
	let mut P = P;
	let mut par = par;
	let bs = linalg::qr::no_pivoting::factor::recommended_block_size::<T>(m, n);
	let (mut qr, stack) = unsafe { temp_mat_uninit::<T, _, _>(m, n, stack) };
	let mut qr = qr.as_mat_mut();
	let (mut householder, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(bs, n, stack) };
	let mut householder = householder.as_mat_mut();
	qr.copy_from(A);
	// finite entries near the largest finite value overflow the QR's column norms: factor
	// `eps * A` instead (a power of two, so exact), which leaves $W$ unchanged and scales $P$,
	// and scale $P$ back at the end
	let shrink = !is_finite(&qr.norm_l2()) && is_finite(&qr.norm_max());
	if shrink {
		let down = eps::<T::Real>();
		z!(qr.rb_mut()).for_each(|uz!(x)| *x = mul_real(x, &down));
	}
	linalg::qr::no_pivoting::factor::qr_in_place(
		qr.rb_mut(),
		householder.rb_mut(),
		par.reborrow(),
		stack,
		default(),
	);
	qdwh_iterate(
		qr.rb().subrows(0, n),
		BlockStructure::TriangularUpper,
		W.rb_mut().subrows_mut(0, n),
		P.rb_mut(),
		par.reborrow(),
		stack,
	)?;
	if shrink {
		let up = recip(&eps::<T::Real>());
		z!(P.rb_mut()).for_each(|uz!(x)| *x = mul_real(x, &up));
	}
	W.rb_mut().subrows_mut(n, m - n).fill(zero());
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
		qr.rb(),
		householder.rb(),
		Conj::No,
		W,
		par,
		stack,
	);
	Ok(())
}

/// the weights $(a, b, c)$ of a QDWH step for the lower bound $\ell$ on the smallest singular
/// value (Nakatsukasa, Bai and Gygi 2010, section 3)
fn qdwh_weights(l: f64) -> (f64, f64, f64) {
	let l2 = l * l;
	let gamma = libm::cbrt(4.0 * (1.0 - l2) / (l2 * l2));
	let a = libm::sqrt(1.0 + gamma)
		+ 0.5
			* libm::sqrt(
				8.0 - 4.0 * gamma
					+ 8.0 * (2.0 - l2) / (l2 * libm::sqrt(1.0 + gamma)),
			);
	let b = (a - 1.0) * (a - 1.0) / 4.0;
	(a, b, a + b - 1.0)
}

/// the polar decomposition of the square `base` (read through `base_structure`): $W$ into `X`,
/// $P$ into `P`
fn qdwh_iterate<T: ComplexField>(
	base: MatRef<'_, T>,
	base_structure: BlockStructure,
	X: MatMut<'_, T>,
	P: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) -> Result<(), PolarError> {
	let n = base.nrows();
	let mut X = X;
	let mut P = P;
	let mut par = par;
	if base_structure == BlockStructure::Rectangular {
		X.copy_from(base);
	} else {
		X.fill(zero());
		z!(X.rb_mut(), base).for_each_triangular_upper(
			linalg::zip::Diag::Include,
			|uz!(dst, src)| *dst = copy(src),
		);
	}
	let (mut G, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut G = G.as_mat_mut();
	let mut alpha = X.norm_l2();
	if alpha < min_positive() {
		// subnormal, so its reciprocal overflows, or zero, which `norm_l2` also returns for a
		// nonzero matrix with entries near 1e-320: scale by an exact power of two first
		let up = recip(&eps::<T::Real>());
		z!(X.rb_mut()).for_each(|uz!(x)| *x = mul_real(x, &up));
		alpha = X.norm_l2();
	}
	if !is_finite(&alpha) && is_finite(&X.norm_max()) {
		// the overflow twin: finite entries near the largest finite value make `norm_l2`
		// overflow, and `recip` of infinity would zero $X$ and report a converged $W = 0$.
		// non-finite entries keep the infinite norm and end in `NoConvergence` below.
		let down = eps::<T::Real>();
		z!(X.rb_mut()).for_each(|uz!(x)| *x = mul_real(x, &down));
		alpha = X.norm_l2();
	}
	if alpha == zero() {
		// $A = 0$: every isometry is a polar factor
		X.fill(zero());
		X.rb_mut().diagonal_mut().fill(one());
	} else {
		let alpha_inv = recip(&alpha);
		z!(X.rb_mut()).for_each(|uz!(x)| *x = mul_real(x, &alpha_inv));
		// unit roundoff of `T`, and the lower bound on $\sigma_{\min}(X_0)$ the weights start from
		let u = libm::exp2(-(nbits::<T>() as f64));
		let tol = from_f64::<T::Real>(libm::cbrt(5.0 * u));
		let mut l = u;
		let mut converged = false;
		for _ in 0..QDWH_MAX_ITERATIONS {
			let (a, b, c) = qdwh_weights(l);
			// $X_{k+1} = (b / c) X_k + \delta G$, where $G$ is $X_k (I + c X_k^H X_k)^{-1}$
			// up to the factor folded into $\delta$, and `G_transposed` says $G$ holds it
			// transposed
			let (delta, G_transposed) = if c > 100.0 {
				qdwh_qr_step(X.rb(), c, G.rb_mut(), par.reborrow(), stack);
				((a - b / c) / libm::sqrt(c), false)
			} else {
				qdwh_cholesky_step(
					X.rb(),
					c,
					G.rb_mut(),
					par.reborrow(),
					stack,
				)?;
				(a - b / c, true)
			};
			let G = if G_transposed {
				G.rb().transpose()
			} else {
				G.rb()
			};
			let (b_over_c, delta) =
				(from_f64::<T::Real>(b / c), from_f64::<T::Real>(delta));
			let mut step = zero::<T::Real>();
			z!(X.rb_mut(), G).for_each(|uz!(x, g)| {
				let next = add(&mul_real(x, &b_over_c), &mul_real(g, &delta));
				step = add(&step, &abs2(&sub(&next, x)));
				*x = next;
			});
			l = f64::min(l * (a + b * l * l) / (1.0 + c * l * l), 1.0);
			if sqrt(&step) <= tol && 1.0 - l <= 10.0 * u {
				converged = true;
				break;
			}
		}
		if !converged {
			return Err(PolarError::NoConvergence);
		}
	}
	// $P$ is the hermitian part of $W^H A = W_R^H R$
	linalg::matmul::triangular::matmul(
		G.rb_mut(),
		BlockStructure::Rectangular,
		Accum::Replace,
		X.rb().adjoint(),
		BlockStructure::Rectangular,
		base,
		base_structure,
		one(),
		par,
	);
	P.copy_from(G.rb());
	hermitian_part_in_place(P);
	Ok(())
}

/// $G = Q_1 Q_2^H$ from the thin $QR$ factorization $[\sqrt{c} X; I] = [Q_1; Q_2] R$, with
/// column pivoting.
///
/// $Q_1 Q_2^H$ is a block of the projector $Q Q^H$, so the pivoting does not change it. without
/// it, the first step ($\sqrt{c} \approx 5 \cdot 10^{10}$ in `f64`) left a residual
/// $\|A - W P\|_F / \|A\|_F \approx 5 \cdot 10^{-10}$ on a $33 \times 33$ matrix of condition
/// $5 \cdot 10^9$; with it, $5 \cdot 10^{-16}$, as Nakatsukasa and Higham (2012) prove for the
/// pivoted factorization.
fn qdwh_qr_step<T: ComplexField>(
	X: MatRef<'_, T>,
	c: f64,
	G: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) {
	// shortcut: the factorization ignores the identity block's structure (Nakatsukasa, Bai and
	// Gygi 2010 save flops with it); exploit it if QDWH becomes a hot path
	let n = X.nrows();
	let mut par = par;
	let bs =
		linalg::qr::no_pivoting::factor::recommended_block_size::<T>(2 * n, n);
	let (mut stacked, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(2 * n, n, stack) };
	let mut stacked = stacked.as_mat_mut();
	let (mut householder, stack) =
		unsafe { temp_mat_uninit::<T, _, _>(bs, n, stack) };
	let mut householder = householder.as_mat_mut();
	let (mut col_perm, stack) = stack.make_with(n, |_| 0usize);
	let (mut col_perm_inv, stack) = stack.make_with(n, |_| 0usize);
	let sqrt_c = from_f64::<T::Real>(libm::sqrt(c));
	z!(stacked.rb_mut().subrows_mut(0, n), X)
		.for_each(|uz!(dst, src)| *dst = mul_real(src, &sqrt_c));
	let mut identity = stacked.rb_mut().subrows_mut(n, n);
	identity.fill(zero());
	identity.diagonal_mut().fill(one());
	linalg::qr::col_pivoting::factor::qr_in_place(
		stacked.rb_mut(),
		householder.rb_mut(),
		&mut col_perm,
		&mut col_perm_inv,
		par.reborrow(),
		stack,
		default(),
	);
	let (mut Q, stack) = unsafe { temp_mat_uninit::<T, _, _>(2 * n, n, stack) };
	let mut Q = Q.as_mat_mut();
	Q.fill(zero());
	Q.rb_mut().subrows_mut(0, n).diagonal_mut().fill(one());
	linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
		stacked.rb(),
		householder.rb(),
		Conj::No,
		Q.rb_mut(),
		par.reborrow(),
		stack,
	);
	linalg::matmul::matmul(
		G,
		Accum::Replace,
		Q.rb().subrows(0, n),
		Q.rb().subrows(n, n).adjoint(),
		one(),
		par,
	);
}

/// $G = (X Z^{-1})^\top = \bar Z^{-1} X^\top$ with $Z = I + c X^H X$ from its Cholesky factor
fn qdwh_cholesky_step<T: ComplexField>(
	X: MatRef<'_, T>,
	c: f64,
	G: MatMut<'_, T>,
	par: Par,
	stack: &mut MemStack,
) -> Result<(), PolarError> {
	let n = X.nrows();
	let mut G = G;
	let mut par = par;
	let (mut Z, stack) = unsafe { temp_mat_uninit::<T, _, _>(n, n, stack) };
	let mut Z = Z.as_mat_mut();
	linalg::matmul::triangular::matmul(
		Z.rb_mut(),
		BlockStructure::TriangularLower,
		Accum::Replace,
		X.adjoint(),
		BlockStructure::Rectangular,
		X,
		BlockStructure::Rectangular,
		from_f64::<T>(c),
		par.reborrow(),
	);
	z!(Z.rb_mut().diagonal_mut().column_vector_mut())
		.for_each(|uz!(z)| *z = add(z, &one()));
	// $Z$ has every eigenvalue at least one, so only a non-finite entry fails here
	linalg::cholesky::llt::factor::cholesky_in_place(
		Z.rb_mut(),
		Default::default(),
		par.reborrow(),
		stack,
		default(),
	)
	.map_err(|_| PolarError::NoConvergence)?;
	G.copy_from(X.transpose());
	linalg::cholesky::llt::solve::solve_in_place_with_conj(
		Z.rb(),
		Conj::Yes,
		G.rb_mut(),
		par,
		stack,
	);
	Ok(())
}
