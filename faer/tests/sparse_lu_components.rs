use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::sparse::linalg::{SupernodalThreshold, lu::*};
use colmat::sparse::{SparseColMat, Triplet};
use colmat::traits::{ComplexField, math_utils::from_f64};
use colmat::{Mat, Par, ParShape};
fn check<T: ComplexField>(tolerance: f64, value: fn(f64, f64) -> T) {
	for n in [0, 1, 3, 12, 40] {
		let a = Mat::<T>::from_fn(n, n, |i, j| {
			value(
				if i == j {
					0.1
				} else if (i + j) % 3 == 0 || i.abs_diff(j) == 1 {
					((i * 17 + j * 13) % 19) as f64 - 9.0
				} else {
					0.0
				},
				if i == j || !((i + j) % 3 == 0 || i.abs_diff(j) == 1) {
					0.0
				} else {
					((i + j) % 5) as f64 * 0.1
				},
			)
		});
		let entries: Vec<_> = (0..n)
			.flat_map(|j| (0..n).map(move |i| (i, j)))
			.filter(|&(i, j)| i == j || (i + j) % 3 == 0 || i.abs_diff(j) == 1)
			.map(|(i, j)| Triplet {
				row: i,
				col: j,
				val: a[(i, j)].clone(),
			})
			.collect();
		let sp =
			SparseColMat::<usize, T>::try_new_from_triplets(n, n, &entries)
				.unwrap();
		for threshold in [
			SupernodalThreshold::AUTO,
			SupernodalThreshold::FORCE_SUPERNODAL,
			SupernodalThreshold::FORCE_SIMPLICIAL,
		] {
			let sym = factorize_symbolic_lu(
				sp.symbolic(),
				LuSymbolicParams {
					supernodal_flop_ratio_threshold: threshold,
					..Default::default()
				},
			)
			.unwrap();
			let mut num = NumericLu::new();
			let mut mem =
				MemBuffer::new(sym.factorize_numeric_lu_scratch::<T>(
					ParShape::Seq,
					Default::default(),
				));
			let f = sym
				.factorize_numeric_lu(
					&mut num,
					sp.as_ref(),
					Par::Seq,
					MemStack::new(&mut mem),
					Default::default(),
				)
				.unwrap();
			println!("n={n} threshold={threshold:?}");
			let (l, u) = f.factors().unwrap();
			let p = f.row_perm().arrays().0;
			let q = f.col_perm().arrays().0;
			let expected = Mat::from_fn(n, n, |i, j| a[(p[i], q[j])].clone());
			let err = (&l.to_dense() * &u.to_dense() - expected).norm_max();
			let mut x =
				Mat::<T>::from_fn(n, 1, |i, _| from_f64::<T>((i + 1) as f64));
			let b = x.clone();
			let mut work = MemBuffer::new(
				sym.solve_in_place_scratch::<T>(1, ParShape::Seq),
			);
			f.solve_in_place_with_conj(
				colmat::Conj::No,
				x.as_mut(),
				Par::Seq,
				MemStack::new(&mut work),
			);
			let residual = (&a * &x - &b).norm_max();
			assert!(
				residual < from_f64::<T::Real>(tolerance),
				"solve residual={residual:?}"
			);
			assert!(
				err < from_f64::<T::Real>(tolerance),
				"n={n}: error {err:?}"
			);
		}
		println!("n={n} reconstruction passed for all algorithms");
	}
}

#[test]
fn f32_components() {
	check::<f32>(1e-3, |re, _| re as f32);
}
#[test]
fn f64_components() {
	check::<f64>(1e-10, |re, _| re);
}
#[test]
fn c32_components() {
	check::<colmat::c32>(1e-3, |re, im| colmat::c32::new(re as f32, im as f32));
}
#[test]
fn c64_components() {
	check::<colmat::c64>(1e-10, colmat::c64::new);
}
