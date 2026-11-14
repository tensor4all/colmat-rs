//! tensor4all/colmat-rs#173: the SIMD hooks of `fx128` against the scalar operations.
//!
//! `fx128` is stored as `[hi, lo]` per element, and its register type `Quad<f64s>` holds the high
//! parts of `LANES` elements in `.0` and their low parts in `.1`. The conversion between the two is
//! done by `simd_load`/`simd_mask_load` (deinterleave after the raw load) and
//! `simd_store`/`simd_mask_store` (interleave before the raw store); the `_raw` hooks move memory
//! without it, by contract. This test drives the hooks the library calls, with inputs whose low
//! parts are nonzero, on every instruction set this host offers, and requires each lane to equal
//! the scalar result for the element at the same index.
#![cfg(feature = "std")]
use colmat::fx128;
use colmat::traits::ComplexField;
use colmat::traits::math_utils::from_f64;
use colmat::traits::pulp::{self, Simd, WithSimd};

fn input(n: usize, seed: usize) -> Vec<fx128> {
	(0..n)
		.map(|i| {
			let x = from_f64::<fx128>(1.0 + (i + seed) as f64)
				/ from_f64::<fx128>(11.0 + 4.0 * seed as f64);
			if i % 3 == 1 { -x } else { x }
		})
		.collect()
}

struct Check;
impl WithSimd for Check {
	type Output = usize;

	#[allow(clippy::type_complexity)] // faer signature shape, kept explicit to match upstream
	#[allow(clippy::needless_range_loop)]
	// index loop kept as written: the body indexes parallel buffers and strided views
	#[inline(always)]
	fn with_simd<S: Simd>(self, simd: S) -> usize {
		let lanes = core::mem::size_of::<<fx128 as ComplexField>::SimdVec<S>>()
			/ core::mem::size_of::<fx128>();
		let ctx = fx128::simd_ctx(simd);
		let x = input(lanes, 0);
		let y = input(lanes, 1);
		let z = input(lanes, 2);
		assert!(x.iter().all(|v| v.1 != 0.0));

		let load = |v: &[fx128]| {
			fx128::simd_load(&ctx, unsafe {
				&*(v.as_ptr() as *const <fx128 as ComplexField>::SimdVec<S>)
			})
		};
		let store = |v: <fx128 as ComplexField>::SimdVec<S>| {
			let mut out = vec![fx128::NAN; lanes];
			fx128::simd_store(
				&ctx,
				unsafe {
					&mut *(out.as_mut_ptr()
						as *mut <fx128 as ComplexField>::SimdVec<S>)
				},
				v,
			);
			out
		};
		let (vx, vy, vz) = (load(&x), load(&y), load(&z));

		// round trip
		assert_eq!(store(vx), x);

		// element-wise operations, lane i against element i
		let ops: [(
			&str,
			<fx128 as ComplexField>::SimdVec<S>,
			&dyn Fn(usize) -> fx128,
		); 6] = [
			("add", fx128::simd_add(&ctx, vx, vy), &|i| x[i] + y[i]),
			("sub", fx128::simd_sub(&ctx, vx, vy), &|i| x[i] - y[i]),
			("mul", fx128::simd_mul(&ctx, vx, vy), &|i| x[i] * y[i]),
			("mul_add", fx128::simd_mul_add(&ctx, vx, vy, vz), &|i| {
				x[i] * y[i] + z[i]
			}),
			("abs2", fx128::simd_abs2(&ctx, vx), &|i| x[i] * x[i]),
			("neg", fx128::simd_neg(&ctx, vx), &|i| -x[i]),
		];
		for (name, got, want) in ops {
			let got = store(got);
			for i in 0..lanes {
				let w = want(i);
				let err = (got[i] - w).abs().0 / w.abs().0;
				assert!(
					err < 1e-30,
					"{name} lane {i} on {}: {:?} != {:?}",
					core::any::type_name::<S>(),
					got[i],
					w
				);
			}
		}

		// reductions see each element once, with its own low part
		let sum = fx128::simd_reduce_sum(&ctx, fx128::simd_mul(&ctx, vx, vy));
		let want: fx128 = (0..lanes)
			.map(|i| x[i] * y[i])
			.fold(from_f64(0.0), |a, b| a + b);
		assert!(
			(sum - want).abs().0 <= 1e-30 * want.abs().0.max(1.0),
			"reduce_sum: {sum:?} != {want:?}"
		);
		let max = fx128::simd_reduce_max(&ctx, fx128::simd_abs1(&ctx, vx));
		let want = x
			.iter()
			.map(|v| v.abs())
			.fold(fx128::NEG_INFINITY, |a, b| if b > a { b } else { a });
		assert_eq!(max, want);

		// masked loads and stores: element range [start, end) of one register
		for start in 0..=lanes {
			for end in start..=lanes {
				let mask = fx128::simd_mem_mask_between(
					&ctx,
					start as u64,
					end as u64,
				);
				let v = unsafe {
					fx128::simd_mask_load(&ctx, mask, x.as_ptr() as *const _)
				};
				let sq = fx128::simd_mul(&ctx, v, v);
				let mut out = vec![from_f64::<fx128>(-7.0); lanes];
				unsafe {
					fx128::simd_mask_store(
						&ctx,
						mask,
						out.as_mut_ptr() as *mut _,
						sq,
					)
				};
				for i in 0..lanes {
					if (start..end).contains(&i) {
						let w = x[i] * x[i];
						assert!(
							(out[i] - w).abs().0 < 1e-30 * w.abs().0,
							"masked [{start},{end}) lane {i}: {:?} != {w:?}",
							out[i]
						);
					} else {
						assert_eq!(
							out[i],
							from_f64::<fx128>(-7.0),
							"masked [{start},{end}) wrote lane {i}"
						);
					}
				}
			}
		}

		// the index vector names the element each lane holds: a select on `x < 0` keeps the
		// indices of the negative elements, which are the ones at `i % 3 == 1`
		let iota = fx128::simd_iota(&ctx);
		let neg = fx128::simd_less_than(
			&ctx,
			vx,
			fx128::simd_splat(&ctx, &from_f64(0.0)),
		);
		let none = fx128::simd_index_splat(&ctx, u64::MAX);
		let picked = fx128::simd_index_select(&ctx, neg, iota, none);
		let picked: &[u64] =
			bytemuck::cast_slice(core::slice::from_ref(&picked));
		let mut picked: Vec<u64> =
			picked.iter().copied().filter(|&i| i != u64::MAX).collect();
		picked.sort();
		let want: Vec<u64> = (0..lanes as u64).filter(|i| i % 3 == 1).collect();
		assert_eq!(
			picked,
			want,
			"iota/select on {}",
			core::any::type_name::<S>()
		);
		lanes
	}
}

#[test]
fn fx128_hooks_match_the_scalar_operations_on_every_instruction_set() {
	// what `Arch::new()` dispatches to (what the library uses), plus every level this host has
	let lanes = pulp::Arch::new().dispatch(Check);
	assert!(lanes >= 1);
	assert_eq!(Check.with_simd(pulp::Scalar::new()), 1);
	#[cfg(target_arch = "x86_64")]
	{
		if let Some(simd) = pulp::x86::V3::try_new() {
			assert_eq!(simd.vectorize(|| Check.with_simd(simd)), 4);
		}
		#[cfg(feature = "x86-v4")]
		if let Some(simd) = pulp::x86::V4::try_new() {
			assert_eq!(simd.vectorize(|| Check.with_simd(simd)), 8);
		}
	}
}
