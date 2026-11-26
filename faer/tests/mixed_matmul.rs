//! Public mixed-precision products, checked as a downstream crate (colmat-rs#293).
#![cfg(feature = "std")]

use colmat::linalg::matmul::{MixedRealGemmToF64, mixed_matmul};
use colmat::prelude::{Reborrow, ReborrowMut};
use colmat::traits::ComplexField;
use colmat::traits::math_utils::{abs, add, mul, sub, zero};
use colmat::{Accum, Mat, MatMut, MatRef, Par, fx128};

trait MixedScalar: ComplexField + Copy + Send + Sync + 'static {
	fn from_f64(value: f64) -> Self;
	fn to_f64(self) -> f64;
}

impl MixedScalar for f32 {
	fn from_f64(value: f64) -> Self {
		value as f32
	}

	fn to_f64(self) -> f64 {
		f64::from(self)
	}
}

impl MixedScalar for f64 {
	fn from_f64(value: f64) -> Self {
		value
	}

	fn to_f64(self) -> f64 {
		self
	}
}

fn q(value: f64) -> fx128 {
	fx128::from_f64(value)
}

fn value(i: usize, j: usize, seed: usize) -> f64 {
	((i * 7 + j * 11 + seed * 13) % 19) as f64 * 0.25 - 2.0
}

fn reference<Lhs: MixedScalar, Rhs: MixedScalar>(
	old_dst: MatRef<'_, f64>,
	lhs: MatRef<'_, Lhs>,
	rhs: MatRef<'_, Rhs>,
	accum: Accum,
	alpha: f64,
) -> Mat<fx128> {
	Mat::from_fn(old_dst.nrows(), old_dst.ncols(), |i, j| {
		let mut sum = zero::<fx128>();
		for k in 0..lhs.ncols() {
			let a = q(lhs[(i, k)].to_f64());
			let b = q(rhs[(k, j)].to_f64());
			sum = add(&sum, &mul(&a, &b));
		}
		let product = mul(&q(alpha), &sum);
		match accum {
			Accum::Replace => product,
			Accum::Add => add(&q(old_dst[(i, j)]), &product),
		}
	})
}

fn assert_close<Lhs: MixedScalar, Rhs: MixedScalar>(
	got: MatRef<'_, f64>,
	want: MatRef<'_, fx128>,
	old_dst: MatRef<'_, f64>,
	lhs: MatRef<'_, Lhs>,
	rhs: MatRef<'_, Rhs>,
	accum: Accum,
	alpha: f64,
	context: &str,
) {
	for j in 0..want.ncols() {
		for i in 0..want.nrows() {
			let mut scale = zero::<fx128>();
			for k in 0..lhs.ncols() {
				let a = q(lhs[(i, k)].to_f64());
				let b = q(rhs[(k, j)].to_f64());
				scale = add(&scale, &abs(&mul(&a, &b)));
			}
			scale = mul(&q(alpha.abs()), &scale);
			if accum == Accum::Add {
				scale = add(&scale, &q(old_dst[(i, j)].abs()));
			}
			let error = abs(&sub(&q(got[(i, j)]), &want[(i, j)]));
			let tolerance = 256.0
				* lhs.ncols().max(1) as f64
				* f64::EPSILON
				* abs(&scale).0.max(1.0);
			assert!(
				error.0 <= tolerance,
				"{context} at ({i}, {j}): error={} tolerance={tolerance:e}, got={}, want={}",
				error.0,
				got[(i, j)],
				want[(i, j)].0,
			);
		}
	}
}

fn row_major<T: ComplexField>(src: MatRef<'_, T>) -> Mat<T> {
	src.transpose().to_owned()
}

#[derive(Clone, Copy, Debug)]
enum Layout {
	ColMajor,
	RowMajor,
	Horizontal,
	Reversed,
}

fn check_case<Lhs: MixedScalar, Rhs: MixedScalar>(
	m: usize,
	k: usize,
	n: usize,
	layout: Layout,
	accum: Accum,
	alpha: f64,
	mut par: Par,
) where
	(Lhs, Rhs): MixedRealGemmToF64,
{
	let lhs = Mat::<Lhs>::from_fn(m, k, |i, j| Lhs::from_f64(value(i, j, 5)));
	let rhs = Mat::<Rhs>::from_fn(k, n, |i, j| Rhs::from_f64(value(i, j, 7)));
	let dst0 = Mat::<f64>::from_fn(m, n, |i, j| {
		if accum == Accum::Add {
			value(i, j, 3)
		} else {
			f64::NAN
		}
	});

	let (lhs_s, rhs_s, mut dst_s);
	let (lhs_v, rhs_v, dst_v): (
		MatRef<'_, Lhs>,
		MatRef<'_, Rhs>,
		MatMut<'_, f64>,
	) = match layout {
		Layout::ColMajor => {
			dst_s = dst0;
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
			dst_s = Mat::<f64>::from_fn(m, 2 * n, |i, j| {
				if j % 2 == 0 { dst0[(i, j / 2)] } else { -777.0 }
			});
			let dst_v = dst_s.as_mut().subcols_mut(0, 2 * n - 1);
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
		Layout::Reversed => {
			dst_s = dst0;
			(
				lhs.as_ref().reverse_rows().reverse_cols(),
				rhs.as_ref().reverse_rows().reverse_cols(),
				dst_s.as_mut().reverse_rows_mut().reverse_cols_mut(),
			)
		},
	};
	let mut dst_v = dst_v;
	let old_dst = dst_v.rb().to_owned();
	let want = reference(old_dst.as_ref(), lhs_v, rhs_v, accum, alpha);
	mixed_matmul(dst_v.rb_mut(), accum, lhs_v, rhs_v, alpha, par.reborrow());
	assert_close(
		dst_v.rb(),
		want.as_ref(),
		old_dst.as_ref(),
		lhs_v,
		rhs_v,
		accum,
		alpha,
		&format!("{m}x{k}x{n} {layout:?} {accum:?} alpha={alpha} {par:?}"),
	);
}

fn check_pair<Lhs: MixedScalar, Rhs: MixedScalar>(mut par: Par, team: bool)
where
	(Lhs, Rhs): MixedRealGemmToF64,
{
	let ordinary = [
		(1, 0, 1),
		(1, 1, 1),
		(3, 2, 4),
		(1, 17, 4),
		(4, 17, 1),
		(5, 33, 7),
		(7, 1031, 5),
	];
	let team_shape = [(64, 131, 64)];
	let shapes = if team { &team_shape[..] } else { &ordinary[..] };
	let layouts = if team {
		&[Layout::ColMajor, Layout::Horizontal][..]
	} else {
		&[
			Layout::ColMajor,
			Layout::RowMajor,
			Layout::Horizontal,
			Layout::Reversed,
		][..]
	};
	for &(m, k, n) in shapes {
		for &layout in layouts {
			for accum in [Accum::Replace, Accum::Add] {
				for alpha in if team { &[1.25][..] } else { &[0.0, 1.25][..] } {
					check_case::<Lhs, Rhs>(
						m,
						k,
						n,
						layout,
						accum,
						*alpha,
						par.reborrow(),
					);
				}
			}
		}
	}
}

#[test]
fn f32_product_is_computed_after_promotion() {
	let a = Mat::from_fn(1, 1, |_, _| 1.0_f32 + f32::EPSILON);
	let b = Mat::from_fn(1, 1, |_, _| a[(0, 0)]);
	let mut c = Mat::<f64>::zeros(1, 1);
	mixed_matmul(&mut c, Accum::Replace, &a, &b, 1.0, Par::Seq);
	assert_eq!(c[(0, 0)], f64::from(a[(0, 0)]).powi(2));
}

#[test]
fn f32_f64_product_preserves_the_right_operand() {
	let a = Mat::from_fn(1, 1, |_, _| 1.0_f32 + f32::EPSILON);
	let b = Mat::from_fn(1, 1, |_, _| 1.0 + f64::EPSILON);
	let mut c = Mat::<f64>::zeros(1, 1);
	mixed_matmul(&mut c, Accum::Replace, &a, &b, 1.0, Par::Seq);
	assert_eq!(c[(0, 0)], f64::from(a[(0, 0)]) * b[(0, 0)]);
}

#[test]
fn extreme_f32_values_are_converted_before_arithmetic() {
	for value in [f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX] {
		let lhs = Mat::from_fn(1, 1, |_, _| value);
		let rhs_f32 = Mat::from_fn(1, 1, |_, _| value);
		let rhs_f64 = Mat::from_fn(1, 1, |_, _| f64::from(value));
		let expected = f64::from(value) * f64::from(value);
		assert!(expected.is_finite() && expected > 0.0);
		let mut f32_dst = Mat::<f64>::zeros(1, 1);
		mixed_matmul(
			&mut f32_dst,
			Accum::Replace,
			&lhs,
			&rhs_f32,
			1.0,
			Par::Seq,
		);
		assert_eq!(f32_dst[(0, 0)], expected, "f32 x f32 value {value:?}");
		let mut f64_dst = Mat::<f64>::zeros(1, 1);
		mixed_matmul(
			&mut f64_dst,
			Accum::Replace,
			&lhs,
			&rhs_f64,
			1.0,
			Par::Seq,
		);
		assert_eq!(f64_dst[(0, 0)], expected, "f32 x f64 value {value:?}");
	}
}

#[test]
fn empty_output_and_zero_depth_follow_accum() {
	let lhs = Mat::<f32>::zeros(0, 3);
	let rhs = Mat::<f64>::zeros(3, 2);
	let mut empty = Mat::<f64>::zeros(0, 2);
	mixed_matmul(&mut empty, Accum::Replace, &lhs, &rhs, 1.0, Par::Seq);

	let lhs = Mat::<f32>::zeros(2, 0);
	let rhs = Mat::<f64>::zeros(0, 2);
	let mut replace = Mat::<f64>::from_fn(2, 2, |_, _| f64::NAN);
	mixed_matmul(&mut replace, Accum::Replace, &lhs, &rhs, 1.0, Par::Seq);
	assert!(
		replace
			.as_ref()
			.col_iter()
			.all(|col| col.iter().all(|&x| x == 0.0))
	);

	let mut add = Mat::<f64>::from_fn(2, 2, |i, j| value(i, j, 3));
	let before = add.clone();
	mixed_matmul(&mut add, Accum::Add, &lhs, &rhs, 1.0, Par::Seq);
	assert!(
		add.as_ref()
			.col_iter()
			.zip(before.as_ref().col_iter())
			.all(|(a, b)| a.iter().zip(b.iter()).all(|(a, b)| a == b))
	);
}

#[test]
fn mixed_f32_f32_matches_the_fx128_oracle() {
	check_pair::<f32, f32>(Par::Seq, false);
}

#[test]
fn mixed_f32_f64_matches_the_fx128_oracle() {
	check_pair::<f32, f64>(Par::Seq, false);
}

#[cfg(feature = "threadteam")]
#[test]
fn mixed_pairs_match_the_fx128_oracle_on_a_team() {
	colmat::team::with_team(4, |mut par| {
		check_pair::<f32, f32>(par.reborrow(), true);
		check_pair::<f32, f64>(par, true);
	});
}

#[test]
fn shape_panics_happen_before_output_mutation() {
	let lhs = Mat::<f32>::zeros(2, 3);
	let rhs = Mat::<f32>::zeros(4, 5);
	let mut dst = Mat::<f64>::from_fn(2, 5, |_, _| -777.0);
	let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		mixed_matmul(&mut dst, Accum::Replace, &lhs, &rhs, 1.0, Par::Seq);
	}));
	assert!(result.is_err());
	assert!(
		dst.as_ref()
			.col_iter()
			.all(|col| col.iter().all(|&x| x == -777.0))
	);
}

#[test]
fn strided_destination_guards_are_unchanged() {
	let lhs = Mat::<f32>::from_fn(3, 4, |i, j| f32::from_f64(value(i, j, 5)));
	let rhs = Mat::<f32>::from_fn(4, 2, |i, j| f32::from_f64(value(i, j, 7)));
	let mut dst = Mat::<f64>::from_fn(3, 4, |i, j| {
		if j % 2 == 0 {
			value(i, j / 2, 3)
		} else {
			-777.0
		}
	});
	let before = dst.clone();
	let view = dst.as_mut().subcols_mut(0, 3);
	let view = unsafe {
		MatMut::from_raw_parts_mut(
			view.as_ptr_mut(),
			3,
			2,
			view.row_stride(),
			2 * view.col_stride(),
		)
	};
	mixed_matmul(view, Accum::Add, &lhs, &rhs, 1.25, Par::Seq);
	for j in 0..4 {
		for i in 0..3 {
			if j % 2 == 1 {
				assert_eq!(dst[(i, j)], before[(i, j)]);
			}
		}
	}
}
