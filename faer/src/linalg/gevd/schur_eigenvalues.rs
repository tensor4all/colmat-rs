//! Homogeneous eigenvalues from saved generalized Schur blocks.
use crate::internal_prelude_sp::*;
use equator::assert;

/// Eigenvalue extraction could not safely scale or evaluate a real conjugate
/// block in the working precision. Earlier entries remain valid; the failed block
/// and later entries are unchanged.
///
/// # Examples
/// ```
/// use colmat::linalg::gevd::SchurEigenvalueError;
/// let error = SchurEigenvalueError { position: 2 };
/// assert_eq!(error.position, 2);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SchurEigenvalueError {
	/// First row of the rejected conjugate block.
	pub position: usize,
}

fn equilibrate<T: RealField>(
	a: &mut [T; 4],
	b: &mut [T; 4],
	indices: [usize; 2],
) {
	let mut small = max_positive::<T>();
	let mut large = zero::<T>();
	for &i in &indices {
		for x in [&a[i], &b[i]] {
			let x = x.abs();
			if x != zero() {
				small = small.fmin(&x);
				large = large.fmax(x);
			}
		}
	}
	if large == zero() {
		return;
	}
	let scale = if small == large {
		large
	} else {
		small.sqrt() * large.sqrt()
	};
	if scale == zero() || !scale.is_finite() {
		return;
	}
	// Scale both pencil factors by the same row/column factor. Preserve every
	// representable nonzero component; a lossy step cannot be used to balance.
	for &i in &indices {
		for x in [&a[i], &b[i]] {
			let next = x / &scale;
			if !next.is_finite() || (*x != zero() && next == zero()) {
				return;
			}
		}
	}
	for &i in &indices {
		a[i] = &a[i] / &scale;
		b[i] = &b[i] / &scale;
	}
}

pub(super) fn real_block<T: RealField>(
	s: MatRef<'_, T>,
	t: MatRef<'_, T>,
) -> Option<(T, T, T, T, T, T)> {
	let mut a = core::array::from_fn::<_, 4, _>(|i| s[(i / 2, i % 2)].copy());
	let mut b = core::array::from_fn::<_, 4, _>(|i| t[(i / 2, i % 2)].copy());
	// INVARIANT: four fixed passes over a 2x2 pencil use only stack scalars.
	// Joint row/column equivalences preserve its eigenvalues and protect small
	// T entries which independent whole-factor normalization can erase.
	for _ in 0..4 {
		for indices in [[0, 2], [1, 3], [0, 1], [2, 3]] {
			equilibrate(&mut a, &mut b, indices);
		}
	}
	let mut sa = zero::<T>();
	let mut tb = zero::<T>();
	for i in 0..4 {
		sa = sa.fmax(a[i].abs());
		tb = tb.fmax(b[i].abs());
	}
	if sa == zero() || tb == zero() || !sa.is_finite() || !tb.is_finite() {
		return None;
	}
	for i in 0..4 {
		let na = &a[i] / &sa;
		let nb = &b[i] / &tb;
		if (a[i] != zero() && na == zero()) || (b[i] != zero() && nb == zero())
		{
			return None;
		}
		a[i] = na;
		b[i] = nb;
	}
	let (b1, b2, a1, a2, ai) = super::qz_real::generalized_eigval_2x2(
		(a[0].copy(), a[1].copy(), a[2].copy(), a[3].copy()),
		(b[0].copy(), b[1].copy(), b[2].copy(), b[3].copy()),
	);
	// Check the returned roots against the balanced pencil's characteristic
	// polynomial. This also rejects a regularization in the private evaluator
	// which changes a very small diagonal of T too much for this block.
	let p = &b[0] * &b[3];
	let q = &a[0] * &b[3] + &a[3] * &b[0] - &a[2] * &b[1];
	let r = &a[0] * &a[3] - &a[1] * &a[2];
	let coefficient_scale = p.abs().fmax(q.abs()).fmax(r.abs());
	if p == zero()
		|| coefficient_scale == zero()
		|| !coefficient_scale.is_finite()
	{
		return None;
	}
	let p = &p / &coefficient_scale;
	let q = &q / &coefficient_scale;
	let r = &r / &coefficient_scale;
	let accurate = |ar: &T, im: &T, beta: &T| {
		let norm = ar.abs().fmax(im.abs()).fmax(beta.abs());
		if norm == zero() || !norm.is_finite() {
			return false;
		}
		let ar = ar / &norm;
		let im = im / &norm;
		let beta = beta / &norm;
		let re = &p * (&ar * &ar - &im * &im) - &q * &ar * &beta
			+ &r * &beta * &beta;
		let imaginary = (from_f64::<T>(2.0) * &p * &ar - &q * &beta) * &im;
		let bound = from_f64::<T>(128.0)
			* eps::<T>()
			* (p.abs() * (&ar * &ar + &im * &im)
				+ q.abs() * ar.abs() * beta.abs()
				+ r.abs() * &beta * &beta);
		re.is_finite()
			&& imaginary.is_finite()
			&& re.abs().fmax(imaginary.abs()) <= bound
	};
	if !accurate(&a1, &ai, &b1) || !accurate(&a2, &(-&ai), &b2) {
		return None;
	}
	let rescale = |ar: T, im: T, b: T| {
		let norm = ar.abs().fmax(im.abs()).fmax(b.abs());
		// Normalize before applying the factor scales, preserving their full
		// representable dynamic range rather than dividing one by the other.
		((&ar / &norm) * &sa, (&im / &norm) * &sa, (&b / &norm) * &tb)
	};
	let (ar1, ai1, b1) = rescale(a1, ai.copy(), b1);
	let (ar2, ai2, b2) = rescale(a2, -ai, b2);
	if [&ar1, &ai1, &b1, &ar2, &ai2, &b2]
		.iter()
		.any(|x| !x.is_finite())
		|| ai1 == zero()
		|| ai2 == zero()
		|| b1 == zero()
		|| b2 == zero()
	{
		return None;
	}
	Some((ar1, ai1, b1, ar2, ai2, b2))
}

/// Writes homogeneous eigenvalues of saved real generalized Schur factors.
/// The eigenvalues are `(alpha_re + i*alpha_im)/beta`, without dividing by
/// beta. A 1x1 block copies its S/T diagonal entries, preserving exact zero
/// beta for an infinite eigenvalue and 0/0 for a nonregular block. A real 2x2
/// block produces its conjugate pair using joint row/column equilibration and
/// independent factor scaling. Homogeneous pair scales are not fixed.
/// Inputs and output columns may have arbitrary strides. No scratch or heap
/// memory is required; neither input factor is changed.
///
/// # Panics
/// Panics if factor shapes are not equal and square or the output lengths
/// differ from the factor dimension. Inputs must be finite and in standardized
/// real generalized Schur form, with nonsingular T blocks for conjugate pairs.
///
/// # Errors
/// Returns [`SchurEigenvalueError`] at the first conjugate block whose scaling
/// would erase a nonzero component or whose computed homogeneous values are
/// nonfinite, no longer a conjugate pair, or fail the characteristic-polynomial
/// residual check. Earlier entries have been written;
/// the rejected block and all later entries are unchanged.
///
/// # Examples
/// ```
/// use colmat::{mat, Col};
/// use colmat::linalg::gevd::schur_eigenvalues_real;
/// # fn main() -> Result<(), colmat::linalg::gevd::SchurEigenvalueError> {
/// let s = mat![[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 3.0]];
/// let t = mat![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
/// let mut re = Col::<f64>::zeros(3);
/// let mut im = re.clone();
/// let mut beta = re.clone();
/// schur_eigenvalues_real(s.as_ref(), t.as_ref(), re.as_mut(), im.as_mut(), beta.as_mut())?;
/// assert!(re[0].abs() < 1e-14 && re[1].abs() < 1e-14);
/// assert!(((im[0] / beta[0]).abs() - 1.0).abs() < 1e-14);
/// assert!((im[0] / beta[0] + im[1] / beta[1]).abs() < 1e-14);
/// assert_eq!((re[2], im[2], beta[2]), (3.0, 0.0, 0.0));
/// # Ok(())
/// # }
/// ```
pub fn schur_eigenvalues_real<T: RealField>(
	s: MatRef<'_, T>,
	t: MatRef<'_, T>,
	mut alpha_re: ColMut<'_, T>,
	mut alpha_im: ColMut<'_, T>,
	mut beta: ColMut<'_, T>,
) -> Result<(), SchurEigenvalueError> {
	let n = s.nrows();
	assert!(s.ncols() == n && t.nrows() == n && t.ncols() == n);
	assert!(
		alpha_re.nrows() == n && alpha_im.nrows() == n && beta.nrows() == n
	);
	let mut i = 0;
	while i < n {
		if i + 1 < n && s[(i + 1, i)] != zero() {
			let (ar1, ai1, b1, ar2, ai2, b2) =
				real_block(s.submatrix(i, i, 2, 2), t.submatrix(i, i, 2, 2))
					.ok_or(SchurEigenvalueError { position: i })?;
			alpha_re[i] = ar1;
			alpha_im[i] = ai1;
			beta[i] = b1;
			alpha_re[i + 1] = ar2;
			alpha_im[i + 1] = ai2;
			beta[i + 1] = b2;
			i += 2;
		} else {
			alpha_re[i] = s[(i, i)].copy();
			alpha_im[i] = zero();
			beta[i] = t[(i, i)].copy();
			i += 1;
		}
	}
	Ok(())
}
