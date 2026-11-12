//! Checked block moves for generalized Schur forms.
use crate::internal_prelude_sp::*;
use equator::assert;

/// A generalized Schur block move was rejected. Outputs retain the state after
/// the last successful adjacent swap.
///
/// # Examples
/// ```
/// use colmat::linalg::gevd::SchurReorderError;
/// let error = SchurReorderError { position: 2 };
/// assert_eq!(error.position, 2); // row at which reordering stopped
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SchurReorderError {
	/// Current first row of the moving block.
	pub position: usize,
}

fn real_swap_scratch<T: ComplexField>(n1: usize, n2: usize) -> StackReq {
	let m = n1 + n2;
	let nz = 2 * n1 * n2;
	StackReq::all_of(&[
		temp_mat_scratch::<T>(m, n2),
		temp_mat_scratch::<T>(n1, m),
		StackReq::any_of(&[
			StackReq::all_of(&[
				temp_mat_scratch::<T>(nz, nz),
				temp_mat_scratch::<T>(nz, 1),
				StackReq::new::<usize>(nz),
				StackReq::new::<usize>(nz),
				StackReq::new::<usize>(nz),
				StackReq::new::<usize>(nz),
				StackReq::any_of(&[
					linalg::lu::full_pivoting::factor::lu_in_place_scratch::<
						usize,
						T,
					>(nz, nz, ParShape::Seq, default()),
					linalg::lu::full_pivoting::solve::solve_in_place_scratch::<
						usize,
						T,
					>(nz, 1, ParShape::Seq),
				]),
			]),
			StackReq::all_of(&[
				temp_mat_scratch::<T>(1, n1),
				temp_mat_scratch::<T>(1, n2),
				StackReq::any_of(&[
					linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
						m,
						n1,
						1,
						ParShape::Seq,
						default(),
					),
					linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T>(
						m,
						n2,
						1,
						ParShape::Seq,
						default(),
					),
					temp_mat_scratch::<T>(m, 1),
				]),
			]),
		]),
	])
}
/// Workspace for either generalized Schur block-move routine, including both
/// pencil factors and Schur-vector backups. The bound covers the largest
/// adjacent blocks possible for the scalar type and dimension; no heap memory
/// is allocated by the block moves. Dimensions below two need no workspace.
///
/// # Examples
/// ```
/// use colmat::linalg::gevd::reorder_schur_scratch;
/// assert_eq!(reorder_schur_scratch::<f64>(0).size_bytes(), 0);
/// assert!(reorder_schur_scratch::<f64>(4).size_bytes() > 0);
/// ```
pub fn reorder_schur_scratch<T: ComplexField>(dim: usize) -> StackReq {
	if dim < 2 {
		return StackReq::EMPTY;
	}
	let w = Ord::min(dim, if T::IS_REAL { 4 } else { 2 });
	let kernel = if T::IS_REAL && dim >= 3 {
		StackReq::any_of(&[
			real_swap_scratch::<T>(1, 2),
			real_swap_scratch::<T>(2, 1),
			if dim >= 4 {
				real_swap_scratch::<T>(2, 2)
			} else {
				StackReq::EMPTY
			},
		])
	} else {
		StackReq::EMPTY
	};
	StackReq::all_of(&[
		temp_mat_scratch::<T>(w, dim),
		temp_mat_scratch::<T>(w, dim),
		temp_mat_scratch::<T>(dim, w),
		temp_mat_scratch::<T>(dim, w),
		temp_mat_scratch::<T>(dim, w),
		temp_mat_scratch::<T>(dim, w),
		temp_mat_scratch::<T>(w, w),
		temp_mat_scratch::<T>(w, w),
		temp_mat_scratch::<T>(w, w),
		temp_mat_scratch::<T>(w, w),
		temp_mat_scratch::<T>(w, w),
		temp_mat_scratch::<T>(w, w),
		kernel,
	])
}
struct Pair<R> {
	ar: R,
	ai: R,
	br: R,
	bi: R,
	beta_zero: bool,
	valid: bool,
}
fn pair<R: RealField>(ar: R, ai: R, br: R, bi: R) -> Pair<R> {
	let beta_zero = br == zero() && bi == zero();
	let scale = ar.abs().fmax(ai.abs()).fmax(br.abs()).fmax(bi.abs());
	if scale == zero() || !scale.is_finite() {
		return Pair {
			ar,
			ai,
			br,
			bi,
			beta_zero,
			valid: false,
		};
	}
	Pair {
		ar: &ar / &scale,
		ai: &ai / &scale,
		br: &br / &scale,
		bi: &bi / &scale,
		beta_zero,
		valid: true,
	}
}
fn scalar_pair<T: ComplexField>(a: &T, b: &T) -> Pair<T::Real> {
	pair(a.real(), a.imag(), b.real(), b.imag())
}
fn same_pair<R: RealField>(a: &Pair<R>, b: &Pair<R>, tol: &R) -> bool {
	if !a.valid || !b.valid || a.beta_zero != b.beta_zero {
		return false;
	}
	let re = (&a.ar * &b.br - &a.ai * &b.bi) - (&b.ar * &a.br - &b.ai * &a.bi);
	let im = (&a.ar * &b.bi + &a.ai * &b.br) - (&b.ar * &a.bi + &b.ai * &a.br);
	re.is_finite() && im.is_finite() && re.abs().fmax(im.abs()) <= *tol
}
fn complex_pairs<T: ComplexField>(
	s: MatRef<'_, T>,
	t: MatRef<'_, T>,
) -> [Pair<T::Real>; 2] {
	[
		scalar_pair(&s[(0, 0)], &t[(0, 0)]),
		pair(zero(), zero(), zero(), zero()),
	]
}
fn real_pairs<T: RealField>(
	s: MatRef<'_, T>,
	t: MatRef<'_, T>,
) -> [Pair<T>; 2] {
	if s.nrows() == 1 {
		return complex_pairs(s, t);
	}
	match super::schur_eigenvalues::real_block(s, t) {
		Some((ar1, ai1, b1, ar2, ai2, b2)) => {
			[pair(ar1, ai1, b1, zero()), pair(ar2, ai2, b2, zero())]
		},
		None => [
			pair(zero(), zero(), zero(), zero()),
			pair(zero(), zero(), zero(), zero()),
		],
	}
}
fn same_block<R: RealField>(
	a: &[Pair<R>; 2],
	b: &[Pair<R>; 2],
	size: usize,
	tol: &R,
) -> bool {
	if size == 1 {
		same_pair(&a[0], &b[0], tol)
	} else {
		// A real conjugate block must remain a conjugate block. A vanishing
		// imaginary homogeneous component cannot silently split the pair.
		if a[0].ai == zero()
			|| a[1].ai == zero()
			|| b[0].ai == zero()
			|| b[1].ai == zero()
		{
			return false;
		}
		(same_pair(&a[0], &b[0], tol) && same_pair(&a[1], &b[1], tol))
			|| (same_pair(&a[0], &b[1], tol) && same_pair(&a[1], &b[0], tol))
	}
}
fn normalized<T: ComplexField>(x: &T, scale: &T::Real, imaginary: &T) -> T {
	let re = &x.real() / scale;
	if T::IS_REAL {
		from_real::<T>(&re)
	} else {
		let im = &x.imag() / scale;
		from_real::<T>(&re) + imaginary.mul_real(&im)
	}
}
fn component_max<T: ComplexField>(x: &T) -> T::Real {
	x.real().abs().fmax(x.imag().abs())
}
fn window_scale<T: ComplexField>(a: MatRef<'_, T>) -> T::Real {
	let mut scale = zero::<T::Real>();
	for j in 0..a.ncols() {
		for i in 0..a.nrows() {
			scale = scale.fmax(component_max(&a[(i, j)]));
		}
	}
	if scale == zero() { one() } else { scale }
}
fn unitary<T: ComplexField>(q: MatRef<'_, T>, tol: &T::Real) -> bool {
	for j in 0..q.ncols() {
		for i in 0..q.ncols() {
			let mut v = zero::<T>();
			for k in 0..q.nrows() {
				v += q[(k, i)].conj() * &q[(k, j)];
			}
			if i == j {
				v -= one::<T>();
			}
			if !v.is_finite() || component_max(&v) > *tol {
				return false;
			}
		}
	}
	true
}
fn reconstruction<T: ComplexField>(
	old: MatRef<'_, T>,
	new: MatRef<'_, T>,
	q: MatRef<'_, T>,
	z: MatRef<'_, T>,
	scale: &T::Real,
	imaginary: &T,
	tol: &T::Real,
) -> bool {
	// INVARIANT: an adjacent QZ window has at most four rows/columns. This
	// bounded check does not add a dimension-dependent dense reconstruction.
	for j in 0..old.ncols() {
		for i in 0..old.nrows() {
			let mut value = zero::<T>();
			for k in 0..old.nrows() {
				for l in 0..old.nrows() {
					value += (&q[(i, k)] * &new[(k, l)]) * z[(j, l)].conj();
				}
			}
			let error = value - normalized(&old[(i, j)], scale, imaginary);
			if !error.is_finite() || component_max(&error) > *tol {
				return false;
			}
		}
	}
	true
}
fn block_size<T: ComplexField>(
	s: MatRef<'_, T>,
	i: usize,
	real: bool,
) -> usize {
	if real && i + 1 < s.nrows() && s[(i + 1, i)] != zero() {
		2
	} else {
		1
	}
}
fn apply_window<T: ComplexField>(
	mut a: MatMut<'_, T>,
	rows: MatRef<'_, T>,
	cols: MatRef<'_, T>,
	new: MatRef<'_, T>,
	q: MatRef<'_, T>,
	z: MatRef<'_, T>,
	start: usize,
) {
	let k = new.nrows();
	let n = a.nrows();
	// INVARIANT: k<=4. These scalar short products update only the affected
	// stripes from their disjoint backups and need no heap/packing workspace.
	for j in start + k..n {
		for i in 0..k {
			let mut value = zero::<T>();
			for r in 0..k {
				value += q[(r, i)].conj() * &rows[(r, j - start)];
			}
			a[(start + i, j)] = value;
		}
	}
	for j in 0..k {
		for i in 0..start {
			let mut value = zero::<T>();
			for r in 0..k {
				value += &cols[(i, r)] * &z[(r, j)];
			}
			a[(i, start + j)] = value;
		}
	}
	for j in 0..k {
		for i in 0..k {
			a[(start + i, start + j)] = new[(i, j)].copy();
		}
	}
}
fn apply_vectors<T: ComplexField>(
	mut a: MatMut<'_, T>,
	old: MatRef<'_, T>,
	rotation: MatRef<'_, T>,
	start: usize,
) {
	for j in 0..rotation.ncols() {
		for i in 0..a.nrows() {
			let mut value = zero::<T>();
			for r in 0..rotation.nrows() {
				value += &old[(i, r)] * &rotation[(r, j)];
			}
			a[(i, start + j)] = value;
		}
	}
}
// INVARIANT: one shared implementation takes the two pencil factors, two
// optional vector accumulators, two indices and the scalar-specific kernels.
#[allow(clippy::too_many_arguments)]
fn move_block<T: ComplexField>(
	mut s: MatMut<'_, T>,
	mut t: MatMut<'_, T>,
	mut q: Option<MatMut<'_, T>>,
	mut z: Option<MatMut<'_, T>>,
	mut ifst: usize,
	mut ilst: usize,
	real: bool,
	pairs: impl Fn(MatRef<'_, T>, MatRef<'_, T>) -> [Pair<T::Real>; 2],
	mut swap: impl FnMut(
		MatMut<'_, T>,
		MatMut<'_, T>,
		MatMut<'_, T>,
		MatMut<'_, T>,
		usize,
		usize,
		&mut MemStack,
	),
	stack: &mut MemStack,
) -> Result<usize, SchurReorderError> {
	let n = s.nrows();
	assert!(s.ncols() == n && t.nrows() == n && t.ncols() == n);
	for a in [q.rb(), z.rb()].into_iter().flatten() {
		assert!(a.nrows() == n && a.ncols() == n);
	}
	if n == 0 {
		assert!(ifst == 0 && ilst == 0);
		return Ok(0);
	}
	assert!(ifst < n && ilst < n);
	if real {
		if ifst > 0 && s[(ifst, ifst - 1)] != zero() {
			ifst -= 1;
		}
		if ilst > 0 && s[(ilst, ilst - 1)] != zero() {
			ilst -= 1;
		}
	}
	if ifst == ilst {
		return Ok(ifst);
	}
	let moving = block_size(s.rb(), ifst, real);
	if ifst < ilst {
		ilst = ilst + block_size(s.rb(), ilst, real) - moving;
	}
	let w = Ord::min(n, if T::IS_REAL { 4 } else { 2 });
	let (mut sr, stack) = temp_mat_zeroed::<T, _, _>(w, n, stack);
	let mut sr = sr.as_mat_mut();
	let (mut tr, stack) = temp_mat_zeroed::<T, _, _>(w, n, stack);
	let mut tr = tr.as_mat_mut();
	let (mut sc, stack) = temp_mat_zeroed::<T, _, _>(n, w, stack);
	let mut sc = sc.as_mat_mut();
	let (mut tc, stack) = temp_mat_zeroed::<T, _, _>(n, w, stack);
	let mut tc = tc.as_mat_mut();
	let (mut qc, stack) = temp_mat_zeroed::<T, _, _>(n, w, stack);
	let mut qc = qc.as_mat_mut();
	let (mut zc, stack) = temp_mat_zeroed::<T, _, _>(n, w, stack);
	let mut zc = zc.as_mat_mut();
	let (mut sw, stack) = temp_mat_zeroed::<T, _, _>(w, w, stack);
	let mut sw = sw.as_mat_mut();
	let (mut tw, stack) = temp_mat_zeroed::<T, _, _>(w, w, stack);
	let mut tw = tw.as_mat_mut();
	let (mut qw, stack) = temp_mat_zeroed::<T, _, _>(w, w, stack);
	let mut qw = qw.as_mat_mut();
	let (mut zw, stack) = temp_mat_zeroed::<T, _, _>(w, w, stack);
	let mut zw = zw.as_mat_mut();
	let (mut sa, stack) = temp_mat_zeroed::<T, _, _>(w, w, stack);
	let mut sa = sa.as_mat_mut();
	let (mut ta, stack) = temp_mat_zeroed::<T, _, _>(w, w, stack);
	let mut ta = ta.as_mat_mut();
	let imaginary = if T::IS_REAL {
		zero::<T>()
	} else {
		(-one::<T>()).sqrt()
	};
	let tol = from_f64::<T::Real>(128.0) * eps::<T::Real>();
	let mut here = ifst;
	while here != ilst {
		let down = here < ilst;
		let next = if down {
			block_size(s.rb(), here + moving, real)
		} else if real && here > 1 && s[(here - 1, here - 2)] != zero() {
			2
		} else {
			1
		};
		let start = if down { here } else { here - next };
		let (n1, n2) = if down { (moving, next) } else { (next, moving) };
		let k = n1 + n2;
		let width = n - start;
		let height = start + k;
		sr.rb_mut()
			.submatrix_mut(0, 0, k, width)
			.copy_from(s.rb().submatrix(start, start, k, width));
		tr.rb_mut()
			.submatrix_mut(0, 0, k, width)
			.copy_from(t.rb().submatrix(start, start, k, width));
		sc.rb_mut()
			.submatrix_mut(0, 0, height, k)
			.copy_from(s.rb().submatrix(0, start, height, k));
		tc.rb_mut()
			.submatrix_mut(0, 0, height, k)
			.copy_from(t.rb().submatrix(0, start, height, k));
		if let Some(a) = q.rb() {
			qc.rb_mut()
				.submatrix_mut(0, 0, n, k)
				.copy_from(a.submatrix(0, start, n, k));
		}
		if let Some(a) = z.rb() {
			zc.rb_mut()
				.submatrix_mut(0, 0, n, k)
				.copy_from(a.submatrix(0, start, n, k));
		}
		let old_s = sr.rb().submatrix(0, 0, k, k);
		let old_t = tr.rb().submatrix(0, 0, k, k);
		if !old_s.is_all_finite() || !old_t.is_all_finite() {
			return Err(SchurReorderError { position: here });
		}
		let scale_s = window_scale(old_s);
		let scale_t = window_scale(old_t);
		let before1 =
			pairs(old_s.submatrix(0, 0, n1, n1), old_t.submatrix(0, 0, n1, n1));
		let before2 = pairs(
			old_s.submatrix(n1, n1, n2, n2),
			old_t.submatrix(n1, n1, n2, n2),
		);
		let mut a = sw.rb_mut().submatrix_mut(0, 0, k, k);
		let mut b = tw.rb_mut().submatrix_mut(0, 0, k, k);
		let mut u = qw.rb_mut().submatrix_mut(0, 0, k, k);
		let mut v = zw.rb_mut().submatrix_mut(0, 0, k, k);
		for j in 0..k {
			for i in 0..k {
				a[(i, j)] = normalized(&old_s[(i, j)], &scale_s, &imaginary);
				b[(i, j)] = normalized(&old_t[(i, j)], &scale_t, &imaginary);
				u[(i, j)] = if i == j { one() } else { zero() };
				v[(i, j)] = if i == j { one() } else { zero() };
			}
		}
		let normalized_before1 = pairs(
			a.rb().submatrix(0, 0, n1, n1),
			b.rb().submatrix(0, 0, n1, n1),
		);
		let normalized_before2 = pairs(
			a.rb().submatrix(n1, n1, n2, n2),
			b.rb().submatrix(n1, n1, n2, n2),
		);
		swap(
			a.rb_mut(),
			b.rb_mut(),
			u.rb_mut(),
			v.rb_mut(),
			n1,
			n2,
			stack,
		);
		// Exact zero beta is part of the incoming regular infinite eigenvalue.
		// Canonicalize its target before the backward-error test.
		if n1 == 1 && before1[0].beta_zero {
			b[(n2, n2)] = zero();
		}
		if n2 == 1 && before2[0].beta_zero {
			b[(0, 0)] = zero();
		}
		let normalized_after2 = pairs(
			a.rb().submatrix(0, 0, n2, n2),
			b.rb().submatrix(0, 0, n2, n2),
		);
		let normalized_after1 = pairs(
			a.rb().submatrix(n2, n2, n1, n1),
			b.rb().submatrix(n2, n2, n1, n1),
		);
		let valid = a.is_all_finite()
			&& b.is_all_finite()
			&& u.is_all_finite()
			&& v.is_all_finite()
			&& same_block(&normalized_before1, &normalized_after1, n1, &tol)
			&& same_block(&normalized_before2, &normalized_after2, n2, &tol)
			&& unitary(u.rb(), &tol)
			&& unitary(v.rb(), &tol)
			&& reconstruction(
				old_s,
				a.rb(),
				u.rb(),
				v.rb(),
				&scale_s,
				&imaginary,
				&tol,
			) && reconstruction(
			old_t,
			b.rb(),
			u.rb(),
			v.rb(),
			&scale_t,
			&imaginary,
			&tol,
		);
		if !valid {
			return Err(SchurReorderError { position: here });
		}
		let mut actual_s = sa.rb_mut().submatrix_mut(0, 0, k, k);
		let mut actual_t = ta.rb_mut().submatrix_mut(0, 0, k, k);
		for j in 0..k {
			for i in 0..k {
				actual_s[(i, j)] = a[(i, j)].mul_real(&scale_s);
				actual_t[(i, j)] = b[(i, j)].mul_real(&scale_t);
			}
		}
		let after2 = pairs(
			actual_s.rb().submatrix(0, 0, n2, n2),
			actual_t.rb().submatrix(0, 0, n2, n2),
		);
		let after1 = pairs(
			actual_s.rb().submatrix(n2, n2, n1, n1),
			actual_t.rb().submatrix(n2, n2, n1, n1),
		);
		if !actual_s.is_all_finite()
			|| !actual_t.is_all_finite()
			|| !same_block(&before1, &after1, n1, &tol)
			|| !same_block(&before2, &after2, n2, &tol)
			|| (real
				&& ((n2 == 2 && actual_s[(1, 0)] == zero())
					|| (n1 == 2 && actual_s[(n2 + 1, n2)] == zero())))
		{
			return Err(SchurReorderError { position: here });
		}
		apply_window(
			s.rb_mut(),
			sr.rb(),
			sc.rb(),
			actual_s.rb(),
			u.rb(),
			v.rb(),
			start,
		);
		apply_window(
			t.rb_mut(),
			tr.rb(),
			tc.rb(),
			actual_t.rb(),
			u.rb(),
			v.rb(),
			start,
		);
		if let Some(a) = q.rb_mut() {
			apply_vectors(a, qc.rb(), u.rb(), start);
		}
		if let Some(a) = z.rb_mut() {
			apply_vectors(a, zc.rb(), v.rb(), start);
		}
		let finite = s.rb().submatrix(start, start, k, width).is_all_finite()
			&& s.rb().submatrix(0, start, height, k).is_all_finite()
			&& t.rb().submatrix(start, start, k, width).is_all_finite()
			&& t.rb().submatrix(0, start, height, k).is_all_finite()
			&& q.rb()
				.is_none_or(|q| q.submatrix(0, start, n, k).is_all_finite())
			&& z.rb()
				.is_none_or(|z| z.submatrix(0, start, n, k).is_all_finite());
		if !finite {
			s.rb_mut()
				.submatrix_mut(0, start, height, k)
				.copy_from(sc.rb().submatrix(0, 0, height, k));
			t.rb_mut()
				.submatrix_mut(0, start, height, k)
				.copy_from(tc.rb().submatrix(0, 0, height, k));
			s.rb_mut()
				.submatrix_mut(start, start, k, width)
				.copy_from(sr.rb().submatrix(0, 0, k, width));
			t.rb_mut()
				.submatrix_mut(start, start, k, width)
				.copy_from(tr.rb().submatrix(0, 0, k, width));
			if let Some(q) = q.rb_mut() {
				q.submatrix_mut(0, start, n, k)
					.copy_from(qc.rb().submatrix(0, 0, n, k));
			}
			if let Some(z) = z.rb_mut() {
				z.submatrix_mut(0, start, n, k)
					.copy_from(zc.rb().submatrix(0, 0, n, k));
			}
			return Err(SchurReorderError { position: here });
		}
		here = if down { here + next } else { here - next };
	}
	Ok(here)
}
/// Moves a real generalized Schur block from `ifst` to `ilst`, updating optional
/// Q/Z. Indices name zero-based rows; either row of a conjugate block names its
/// first row. A downward move adjusts the destination for differing block sizes.
/// Empty factors accept the no-op indices (0,0). Adjacent moves execute
/// sequentially for any supplied parallelism. Optional Q/Z must contain the
/// existing finite orthogonal accumulators; they are postmultiplied by the
/// accepted local transformations.
///
/// # Panics
/// Panics on invalid shapes/indices or insufficient queried workspace. Input
/// factors must be finite, in standardized real generalized Schur form.
///
/// # Errors
/// Rejects unsafe swaps, nonregular diagonal blocks, or a numerically split
/// conjugate block. The last successful state and its moving position remain.
/// Each accepted swap is checked for both pencil reconstructions, local
/// unitarity, projective eigenvalue order and finite updated stripes.
///
/// # Examples
/// ```
/// use colmat::{mat, Mat, Par};
/// use colmat::dyn_stack::{MemBuffer, MemStack};
/// use colmat::linalg::gevd::{reorder_schur_real, reorder_schur_scratch};
/// # fn main() -> Result<(), colmat::linalg::gevd::SchurReorderError> {
/// let mut s = mat![[1.0, 0.0], [0.0, 3.0]];
/// let original = s.clone();
/// let mut t = Mat::<f64>::identity(2, 2);
/// let mut q = t.clone();
/// let mut z = t.clone();
/// let mut memory = MemBuffer::new(reorder_schur_scratch::<f64>(2));
/// let position = reorder_schur_real(s.as_mut(), t.as_mut(), Some(q.as_mut()),
///     Some(z.as_mut()), 1, 0, Par::Seq, MemStack::new(&mut memory))?;
/// assert_eq!(position, 0);
/// assert_eq!(s[(0, 0)] / t[(0, 0)], 3.0);
/// assert!((&q * &s * z.adjoint() - original).norm_l2() < 1e-14);
/// assert!((&q * &t * z.adjoint() - Mat::<f64>::identity(2, 2)).norm_l2() < 1e-14);
/// # Ok(())
/// # }
/// ```
// INVARIANT: follows the ordinary Schur block-move signature, with both
// generalized pencil factors and their separate Q/Z accumulators.
#[allow(clippy::too_many_arguments)]
pub fn reorder_schur_real<T: RealField>(
	s: MatMut<'_, T>,
	t: MatMut<'_, T>,
	q: Option<MatMut<'_, T>>,
	z: Option<MatMut<'_, T>>,
	ifst: usize,
	ilst: usize,
	_par: Par,
	stack: &mut MemStack,
) -> Result<usize, SchurReorderError> {
	move_block(
		s,
		t,
		q,
		z,
		ifst,
		ilst,
		true,
		real_pairs::<T>,
		|s, t, q, z, n1, n2, stack| {
			super::qz_real::swap_qz(s, t, Some(q), Some(z), 0, n1, n2, stack)
		},
		stack,
	)
}
/// Moves a complex triangular generalized Schur entry from `ifst` to `ilst`,
/// updating optional finite unitary Q/Z accumulators by postmultiplication.
/// Empty factors accept (0,0). Regular infinite
/// eigenvalues (zero beta, nonzero alpha) are supported without division.
///
/// # Panics
/// Panics on invalid shapes/indices or insufficient queried workspace. Input
/// factors must be finite and upper triangular.
///
/// # Errors
/// Returns the last valid state/position when a checked adjacent swap fails.
/// Checks and sequential execution follow [`reorder_schur_real`].
///
/// # Examples
/// ```
/// use colmat::{c64, mat, Mat, Par};
/// use colmat::dyn_stack::{MemBuffer, MemStack};
/// use colmat::linalg::gevd::{reorder_schur_cplx, reorder_schur_scratch};
/// # fn main() -> Result<(), colmat::linalg::gevd::SchurReorderError> {
/// let mut s = mat![[c64::new(1.0, 1.0), c64::new(0.0, 0.0)],
///                  [c64::new(0.0, 0.0), c64::new(3.0, -1.0)]];
/// let original = s.clone();
/// let mut t = Mat::<c64>::identity(2, 2);
/// let mut q = t.clone();
/// let mut z = t.clone();
/// let mut memory = MemBuffer::new(reorder_schur_scratch::<c64>(2));
/// reorder_schur_cplx(s.as_mut(), t.as_mut(), Some(q.as_mut()),
///     Some(z.as_mut()), 1, 0, Par::Seq, MemStack::new(&mut memory))?;
/// assert!((s[(0, 0)] / t[(0, 0)] - c64::new(3.0, -1.0)).norm() < 1e-14);
/// assert!((&q * &s * z.adjoint() - original).norm_l2() < 1e-14);
/// # Ok(())
/// # }
/// ```
// INVARIANT: follows the ordinary Schur block-move signature, with both
// generalized pencil factors and their separate Q/Z accumulators.
#[allow(clippy::too_many_arguments)]
pub fn reorder_schur_cplx<T: ComplexField>(
	s: MatMut<'_, T>,
	t: MatMut<'_, T>,
	q: Option<MatMut<'_, T>>,
	z: Option<MatMut<'_, T>>,
	ifst: usize,
	ilst: usize,
	_par: Par,
	stack: &mut MemStack,
) -> Result<usize, SchurReorderError> {
	move_block(
		s,
		t,
		q,
		z,
		ifst,
		ilst,
		false,
		complex_pairs::<T>,
		|s, t, q, z, _, _, _| {
			super::qz_cplx::swap_qz(s, t, Some(q), Some(z), 0)
		},
		stack,
	)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lapack;
