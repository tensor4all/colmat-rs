//! tensor4all/colmat-rs#174 and #173: the dense product of the double-double scalars `fx128` and
//! `cx128`, run as a downstream crate runs it. An integration test is compiled against the library
//! built without `cfg(test)`, so the SIMD micro-kernels are reached through the same gate a user
//! reaches (the in-crate tests took a `cfg!(test)` branch that no user build takes).
//!
//! Every entry has a nonzero low part, so a lane that mixed the high and low parts of different
//! elements (#173) cannot agree with the scalar reference.
#![cfg(feature = "std")]
use colmat::linalg::matmul::{matmul, matmul_with_conj};
use colmat::prelude::{Reborrow, ReborrowMut};
use colmat::traits::ComplexField;
use colmat::traits::math_utils::{abs, add, conj, mul, sub, zero};
use colmat::{Accum, Conj, Mat, MatMut, MatRef, Par, cx128, fx128};

fn q(i: usize) -> fx128 {
	fx128::from_f64(1.0 + i as f64)
		/ fx128::from_f64(3.0 + 7.0 * (i % 29) as f64)
}

fn c(i: usize) -> cx128 {
	cx128 {
		re: q(i),
		im: q(3 * i + 5) - fx128::from_f64(0.25),
	}
}

/// `beta * dst + alpha * op(lhs) * op(rhs)`, one double-double operation at a time.
fn reference<T: ComplexField>(
	dst: MatRef<'_, T>,
	beta: Accum,
	lhs: MatRef<'_, T>,
	conj_lhs: Conj,
	rhs: MatRef<'_, T>,
	conj_rhs: Conj,
	alpha: &T,
) -> Mat<T> {
	Mat::from_fn(dst.nrows(), dst.ncols(), |i, j| {
		let mut acc = zero::<T>();
		for k in 0..lhs.ncols() {
			let mut l = lhs[(i, k)].clone();
			let mut r = rhs[(k, j)].clone();
			if conj_lhs == Conj::Yes {
				l = conj(&l);
			}
			if conj_rhs == Conj::Yes {
				r = conj(&r);
			}
			acc = add(&acc, &mul(&l, &r));
		}
		let acc = mul(alpha, &acc);
		match beta {
			Accum::Add => add(&dst[(i, j)], &acc),
			Accum::Replace => acc,
		}
	})
}

fn max_rel_err<T: ComplexField<Real = fx128>>(
	got: MatRef<'_, T>,
	want: MatRef<'_, T>,
) -> f64 {
	let mut err = 0.0f64;
	for j in 0..want.ncols() {
		for i in 0..want.nrows() {
			let d = abs(&sub(&got[(i, j)], &want[(i, j)]));
			let s = abs(&want[(i, j)]);
			let s = if s.0 == 0.0 { 1.0 } else { s.0 };
			err = err.max(d.0 / s);
		}
	}
	err
}

/// A copy of `src` stored row-major: the same values, transposed storage.
fn row_major<T: ComplexField>(src: MatRef<'_, T>) -> Mat<T> {
	src.transpose().to_owned()
}

#[derive(Clone, Copy, Debug)]
enum Layout {
	/// column-major `dst` and `lhs`: `matmul_vertical`
	ColMajor,
	/// row-major `dst` and `rhs`: `matmul_vertical` on the transposed problem
	RowMajor,
	/// row-major `lhs`, column-major `rhs`, strided `dst`: `matmul_horizontal`
	Horizontal,
}

fn check<T: ComplexField<Real = fx128>>(
	make: impl Fn(usize) -> T,
	mut par: Par,
) {
	// a single element, shapes below and above one register of rows, several micro-kernel rows
	// and columns, and a depth longer than one block
	let shapes = [
		(1, 1, 1),
		(2, 3, 1),
		(13, 17, 11),
		(37, 70, 9),
		(64, 5, 33),
		(5, 300, 7),
	];
	let alpha = make(101);
	for &(m, k, n) in &shapes {
		for layout in [Layout::ColMajor, Layout::RowMajor, Layout::Horizontal] {
			for beta in [Accum::Replace, Accum::Add] {
				for (conj_lhs, conj_rhs) in [
					(Conj::No, Conj::No),
					(Conj::Yes, Conj::No),
					(Conj::No, Conj::Yes),
				] {
					let lhs = Mat::<T>::from_fn(m, k, |i, j| make(i + 3 * j));
					let rhs =
						Mat::<T>::from_fn(k, n, |i, j| make(2 * i + j + 1));
					let dst0 =
						Mat::<T>::from_fn(m, n, |i, j| make(5 * i + 7 * j + 2));
					let want = reference(
						dst0.as_ref(),
						beta,
						lhs.as_ref(),
						conj_lhs,
						rhs.as_ref(),
						conj_rhs,
						&alpha,
					);

					let (lhs_s, rhs_s, mut dst_s);
					let (lhs_v, rhs_v, dst_v): (
						MatRef<'_, T>,
						MatRef<'_, T>,
						MatMut<'_, T>,
					) = match layout {
						Layout::ColMajor => {
							dst_s = dst0.clone();
							(lhs.as_ref(), rhs.as_ref(), dst_s.as_mut())
						},
						Layout::RowMajor => {
							rhs_s = row_major(rhs.as_ref());
							dst_s = row_major(dst0.as_ref());
							(
								lhs.as_ref(),
								rhs_s.transpose(),
								dst_s.as_mut().transpose_mut(),
							)
						},
						Layout::Horizontal => {
							lhs_s = row_major(lhs.as_ref());
							// every other column of a wider matrix: neither row- nor column-major
							dst_s = Mat::<T>::from_fn(m, 2 * n, |i, j| {
								dst0[(i, j / 2)].clone()
							});
							let dst_v =
								dst_s.as_mut().subcols_mut(0, 2 * n - 1);
							let dst_v = unsafe {
								MatMut::from_raw_parts_mut(
									dst_v.as_ptr_mut(),
									m,
									n,
									dst_v.row_stride(),
									2 * dst_v.col_stride(),
								)
							};
							(lhs_s.transpose(), rhs.as_ref(), dst_v)
						},
					};
					let mut dst_v = dst_v;
					matmul_with_conj(
						dst_v.rb_mut(),
						beta,
						lhs_v,
						conj_lhs,
						rhs_v,
						conj_rhs,
						alpha.clone(),
						par.reborrow(),
					);
					let err = max_rel_err(dst_v.rb(), want.as_ref());
					assert!(
						err < 1e-28,
						"{} {m}x{k}x{n} {layout:?} {beta:?} conj=({conj_lhs:?},{conj_rhs:?}) {par:?}: relative error {err:e}",
						core::any::type_name::<T>(),
					);
				}
			}
		}
	}
}

#[test]
fn fx128_product_matches_the_scalar_reference() {
	// the inputs carry low parts: a sum of values with `lo == 0` can hide a lane mix-up (#173)
	assert!((0..64).filter(|&i| q(i).1 != 0.0).count() > 56);
	check::<fx128>(q, Par::Seq);
}

#[test]
fn cx128_product_matches_the_scalar_reference() {
	check::<cx128>(c, Par::Seq);
}

/// The issue's own reproduction, through the operator.
#[test]
fn fx128_operator_product_does_not_panic() {
	let a = Mat::<fx128>::from_fn(13, 17, |i, j| q(i + 3 * j));
	let b = Mat::<fx128>::from_fn(17, 11, |i, j| q(2 * i + j + 1));
	let got = &a * &b;
	let mut via_matmul = Mat::<fx128>::zeros(13, 11);
	matmul(
		via_matmul.as_mut(),
		Accum::Replace,
		a.as_ref(),
		b.as_ref(),
		fx128::from_f64(1.0),
		Par::Seq,
	);
	let want = reference(
		Mat::<fx128>::zeros(13, 11).as_ref(),
		Accum::Replace,
		a.as_ref(),
		Conj::No,
		b.as_ref(),
		Conj::No,
		&fx128::from_f64(1.0),
	);
	assert!(max_rel_err(got.as_ref(), want.as_ref()) < 1e-28);
	assert!(max_rel_err(via_matmul.as_ref(), want.as_ref()) < 1e-28);
}

#[cfg(feature = "threadteam")]
#[test]
fn fx128_and_cx128_products_on_a_team() {
	colmat::team::with_team(3, |mut par| {
		check::<fx128>(q, par.reborrow());
		check::<cx128>(c, par);
	});
}
