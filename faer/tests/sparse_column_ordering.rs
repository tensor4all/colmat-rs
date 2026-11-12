use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::perm::Perm;
use colmat::sparse::linalg::{SupernodalThreshold, lu, qr};
use colmat::sparse::{SparseColMat, Triplet};
use colmat::traits::{ComplexField, math_utils::from_f64};
use colmat::{Conj, Mat, Par, ParShape};
fn check<T: ComplexField>(tolerance: f64, value: fn(f64, f64) -> T) {
	for n in [0, 1, 5, 40] {
		for tall in [false, true] {
			let m = if tall { n + 3 } else { n };
			let a = Mat::<T>::from_fn(m, n, |i, j| {
				value(
					if i == j {
						10.0
					} else if i.abs_diff(j) == 1 {
						0.2
					} else {
						0.0
					},
					if i.abs_diff(j) == 1 { 0.1 } else { 0.0 },
				)
			});
			let entries: Vec<_> = (0..n)
				.flat_map(|j| (0..m).map(move |i| (i, j)))
				.filter(|&(i, j)| i == j || i.abs_diff(j) == 1)
				.map(|(i, j)| Triplet {
					row: i,
					col: j,
					val: a[(i, j)].clone(),
				})
				.collect();
			let sp =
				SparseColMat::<usize, T>::try_new_from_triplets(m, n, &entries)
					.unwrap();
			for ordering in 0..3 {
				let p: Vec<_> = (0..n)
					.map(|i| match ordering {
						0 => i,
						1 => n - 1 - i,
						_ => (i + 1) % n,
					})
					.collect();
				let mut inverse = vec![0; n];
				for (i, &j) in p.iter().enumerate() {
					inverse[j] = i;
				}
				let perm = Perm::new_checked(
					p.clone().into_boxed_slice(),
					inverse.into_boxed_slice(),
					n,
				);
				for threshold in [
					SupernodalThreshold::AUTO,
					SupernodalThreshold::FORCE_SIMPLICIAL,
					SupernodalThreshold::FORCE_SUPERNODAL,
				] {
					if !tall {
						let sym =
							lu::factorize_symbolic_lu_with_column_ordering(
								sp.symbolic(),
								lu::LuSymbolicParams {
									supernodal_flop_ratio_threshold: threshold,
									..Default::default()
								},
								Some(perm.as_ref()),
							)
							.unwrap();
						assert_eq!(sym.col_perm().arrays().0, &p);
						let mut numeric = lu::NumericLu::new();
						let mut mem = MemBuffer::new(
							sym.factorize_numeric_lu_scratch::<T>(
								ParShape::Seq,
								Default::default(),
							),
						);
						let f = sym
							.factorize_numeric_lu(
								&mut numeric,
								sp.as_ref(),
								Par::Seq,
								MemStack::new(&mut mem),
								Default::default(),
							)
							.unwrap();
						let (l, u) = f.factors().unwrap();
						let rows = f.row_perm().arrays().0;
						let expected = Mat::from_fn(n, n, |i, j| {
							a[(rows[i], p[j])].clone()
						});
						assert!(
							(&l.to_dense() * &u.to_dense() - expected)
								.norm_max() < from_f64::<T::Real>(tolerance)
						);
					}
					let sym = qr::factorize_symbolic_qr_with_column_ordering(
						sp.symbolic(),
						qr::QrSymbolicParams {
							supernodal_flop_ratio_threshold: threshold,
							..Default::default()
						},
						Some(perm.as_ref()),
					)
					.unwrap();
					assert_eq!(sym.col_perm().arrays().0, &p);
					let mut indices = vec![0; sym.len_idx()];
					let mut values = vec![from_f64::<T>(0.0); sym.len_val()];
					let mut mem =
						MemBuffer::new(sym.factorize_numeric_qr_scratch::<T>(
							ParShape::Seq,
							Default::default(),
						));
					let f = sym.factorize_numeric_qr(
						&mut indices,
						&mut values,
						sp.as_ref(),
						Par::Seq,
						MemStack::new(&mut mem),
						Default::default(),
					);
					let expected = Mat::from_fn(n, 2, |i, j| {
						from_f64::<T>((i + j + 1) as f64)
					});
					let mut rhs = &a * &expected;
					let mut mem = MemBuffer::new(
						sym.solve_in_place_scratch::<T>(2, ParShape::Seq),
					);
					f.solve_in_place_with_conj(
						Conj::No,
						rhs.as_mut(),
						Par::Seq,
						MemStack::new(&mut mem),
					);
					assert!(
						(rhs.as_ref().subrows(0, n) - expected).norm_max()
							< from_f64::<T::Real>(tolerance)
					);
				}
			}
		}
	}
}

#[test]
fn f32_ordering() {
	check::<f32>(1e-3, |r, _| r as f32);
}
#[test]
fn f64_ordering() {
	check::<f64>(1e-10, |r, _| r);
}
#[test]
fn c32_ordering() {
	check::<colmat::c32>(1e-3, |r, i| colmat::c32::new(r as f32, i as f32));
}
#[test]
fn c64_ordering() {
	check::<colmat::c64>(1e-10, colmat::c64::new);
}

#[test]
fn default_and_none_agree() {
	let a = SparseColMat::<usize, f64>::try_new_from_triplets(
		3,
		3,
		&[
			Triplet {
				row: 0,
				col: 0,
				val: 2.0,
			},
			Triplet {
				row: 1,
				col: 1,
				val: 3.0,
			},
			Triplet {
				row: 2,
				col: 2,
				val: 4.0,
			},
			Triplet {
				row: 0,
				col: 2,
				val: 1.0,
			},
		],
	)
	.unwrap();
	let old =
		lu::factorize_symbolic_lu(a.symbolic(), Default::default()).unwrap();
	let new = lu::factorize_symbolic_lu_with_column_ordering(
		a.symbolic(),
		Default::default(),
		None,
	)
	.unwrap();
	assert_eq!(old.col_perm().arrays(), new.col_perm().arrays());
	let old =
		qr::factorize_symbolic_qr(a.symbolic(), Default::default()).unwrap();
	let new = qr::factorize_symbolic_qr_with_column_ordering(
		a.symbolic(),
		Default::default(),
		None,
	)
	.unwrap();
	assert_eq!(old.col_perm().arrays(), new.col_perm().arrays());
}
#[test]
#[should_panic]
fn lu_rejects_wrong_permutation_length() {
	let a = SparseColMat::<usize, f64>::try_new_from_triplets(
		1,
		1,
		&[Triplet {
			row: 0,
			col: 0,
			val: 2.0,
		}],
	)
	.unwrap();
	let p = Perm::new_checked(
		vec![].into_boxed_slice(),
		vec![].into_boxed_slice(),
		0,
	);
	let _ = lu::factorize_symbolic_lu_with_column_ordering(
		a.symbolic(),
		Default::default(),
		Some(p.as_ref()),
	);
}
#[test]
#[should_panic]
fn qr_rejects_wrong_permutation_length() {
	let a = SparseColMat::<usize, f64>::try_new_from_triplets(
		1,
		1,
		&[Triplet {
			row: 0,
			col: 0,
			val: 2.0,
		}],
	)
	.unwrap();
	let p = Perm::new_checked(
		vec![].into_boxed_slice(),
		vec![].into_boxed_slice(),
		0,
	);
	let _ = qr::factorize_symbolic_qr_with_column_ordering(
		a.symbolic(),
		Default::default(),
		Some(p.as_ref()),
	);
}
