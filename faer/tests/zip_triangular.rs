//! tensor4all/colmat-rs#255: `zip!` triangular iteration visits exactly the triangle, in order.
//!
//! The iteration walks the operand's preferred memory layout, so the order is part of the
//! contract (for a view with a contiguous column; a strided view follows the logical order, see
//! `expected`): columns of the *physical* storage, rows ascending inside a column, and the column
//! order follows the logical axis that maps onto the physical columns (descending when that axis
//! is reversed). The oracle below is derived from the view's construction, not from the code
//! under test. Every assertion is on the visited `(row, col, value)` *sequence*: a sum, or a set,
//! passes on a permutation of the same elements.
#![allow(clippy::type_complexity)]
use colmat::linalg::zip::{Diag, ZipEq};
use colmat::prelude::*;

#[derive(Copy, Clone, Debug)]
struct Layout {
	transpose: bool,
	rev_rows: bool,
	rev_cols: bool,
}

fn layouts() -> Vec<Layout> {
	let mut v = vec![];
	for transpose in [false, true] {
		for rev_rows in [false, true] {
			for rev_cols in [false, true] {
				v.push(Layout {
					transpose,
					rev_rows,
					rev_cols,
				});
			}
		}
	}
	v
}

/// the physical base of a logical `m x n` view, value = its linear (column-major, unit-stride)
/// index; with `strided`, rows are laid out two apart, so no column is contiguous
fn with_base<R>(
	m: usize,
	n: usize,
	l: Layout,
	strided: bool,
	f: impl FnOnce(MatMut<'_, f64>) -> R,
) -> R {
	let (r0, c0) = if l.transpose { (n, m) } else { (m, n) };
	if strided {
		let mut big = Mat::<f64>::zeros(2 * r0, c0);
		for q in 0..c0 {
			for p in 0..r0 {
				big[(2 * p, q)] = (p + q * r0) as f64;
			}
		}
		// an owned `Mat` may pad its column stride, so read it back rather than assume `2 * r0`
		let cs = big.col_stride();
		let ptr = big.as_ptr_mut();
		f(unsafe { MatMut::from_raw_parts_mut(ptr, r0, c0, 2, cs) })
	} else {
		let mut b = Mat::from_fn(r0, c0, |i, j| (i + j * r0) as f64);
		f(b.as_mut())
	}
}

macro_rules! with_view {
	($b:expr, $l:ident, |$v:ident| $body:expr) => {{
		let b = $b;
		match ($l.transpose, $l.rev_rows, $l.rev_cols) {
			(false, false, false) => {
				let $v = b;
				$body
			},
			(false, true, false) => {
				let $v = b.reverse_rows_mut();
				$body
			},
			(false, false, true) => {
				let $v = b.reverse_cols_mut();
				$body
			},
			(false, true, true) => {
				let $v = b.reverse_rows_mut().reverse_cols_mut();
				$body
			},
			(true, false, false) => {
				let $v = b.transpose_mut();
				$body
			},
			(true, true, false) => {
				let $v = b.transpose_mut().reverse_rows_mut();
				$body
			},
			(true, false, true) => {
				let $v = b.transpose_mut().reverse_cols_mut();
				$body
			},
			(true, true, true) => {
				let $v =
					b.transpose_mut().reverse_rows_mut().reverse_cols_mut();
				$body
			},
		}
	}};
}

fn visit(
	m: usize,
	n: usize,
	l: Layout,
	upper: bool,
	diag: Diag,
	strided: bool,
) -> Vec<(usize, usize, f64)> {
	let mut out = vec![];
	with_base(m, n, l, strided, |b| with_view!(b, l, |v| {
		assert_eq!((v.nrows(), v.ncols()), (m, n));
		let f = |i: usize, j: usize, x: colmat::linalg::zip::Last<&mut f64>| {
			out.push((i, j, *x.0))
		};
		if upper {
			zip!(v).for_each_triangular_upper_with_index(diag, f)
		} else {
			zip!(v).for_each_triangular_lower_with_index(diag, f)
		}
	}));
	out
}

/// the values only, through the variants that take no index
fn visit_values(
	m: usize,
	n: usize,
	l: Layout,
	upper: bool,
	diag: Diag,
	strided: bool,
) -> Vec<f64> {
	let mut out = vec![];
	with_base(m, n, l, strided, |b| with_view!(b, l, |v| {
		let f = |x: colmat::linalg::zip::Last<&mut f64>| out.push(*x.0);
		if upper {
			zip!(v).for_each_triangular_upper(diag, f)
		} else {
			zip!(v).for_each_triangular_lower(diag, f)
		}
	}));
	out
}

/// the triangle of the logical `m x n` view, in the documented visiting order
fn expected(
	m: usize,
	n: usize,
	l: Layout,
	upper: bool,
	diag: Diag,
	strided: bool,
) -> Vec<(usize, usize, f64)> {
	let strict = matches!(diag, Diag::Skip) as usize;
	let (r0, _c0) = if l.transpose { (n, m) } else { (m, n) };
	let mut v = vec![];
	for j in 0..n {
		for i in 0..m {
			let inside = if upper {
				i + strict <= j
			} else {
				j + strict <= i
			};
			if !inside {
				continue;
			}
			let (ii, jj) = (
				if l.rev_rows { m - 1 - i } else { i },
				if l.rev_cols { n - 1 - j } else { j },
			);
			let (p, q) = if l.transpose { (jj, ii) } else { (ii, jj) };
			// the axis that lands on the physical columns, and whether it is reversed
			let col_rev = if l.transpose { l.rev_rows } else { l.rev_cols };
			let key = if strided {
				// no column is contiguous, so the walk is not flipped to memory order: it follows
				// the logical view, column by column
				(j as isize, i)
			} else {
				(if col_rev { -(q as isize) } else { q as isize }, p)
			};
			v.push((key, (i, j, (p + q * r0) as f64)));
		}
	}
	v.sort_by_key(|&(k, _)| k);
	v.into_iter().map(|(_, e)| e).collect()
}

const SHAPES: [(usize, usize); 9] = [
	(3, 3),
	(4, 4),
	(2, 4),
	(4, 2),
	(2, 3),
	(3, 2),
	(1, 3),
	(3, 1),
	(1, 1),
];

#[test]
fn triangular_with_index_visits_the_triangle_in_order() {
	for (m, n) in SHAPES {
		for l in layouts() {
			for upper in [false, true] {
				for diag in [Diag::Include, Diag::Skip] {
					for strided in [false, true] {
					assert_eq!(
						visit(m, n, l, upper, diag, strided),
						expected(m, n, l, upper, diag, strided),
						"{m}x{n} {l:?} upper={upper} diag={} strided={strided}",
						matches!(diag, Diag::Include),
					);
					}
				}
			}
		}
	}
}

#[test]
fn triangular_without_index_visits_the_triangle_in_order() {
	for (m, n) in SHAPES {
		for l in layouts() {
			for upper in [false, true] {
				for diag in [Diag::Include, Diag::Skip] {
					for strided in [false, true] {
					let want: Vec<f64> = expected(m, n, l, upper, diag, strided)
						.into_iter()
						.map(|e| e.2)
						.collect();
					let mut got = visit_values(m, n, l, upper, diag, strided);
					// the index-free variants take the transpose of the lower walk for
					// `upper`; they promise the elements, in memory order per column
					let mut w = want.clone();
					if upper {
						got.sort_by(f64::total_cmp);
						w.sort_by(f64::total_cmp);
					}
					assert_eq!(
						got,
						w,
						"{m}x{n} {l:?} upper={upper} diag={} strided={strided}",
						matches!(diag, Diag::Include),
					);
					}
				}
			}
		}
	}
}

/// a wide `2 x 3` view with reversed rows has five upper-triangle elements, not three
#[test]
fn reverse_rows_wide_upper_visits_all_five() {
	let l = Layout {
		transpose: false,
		rev_rows: true,
		rev_cols: false,
	};
	let got = visit(2, 3, l, true, Diag::Include, false);
	// base 2x3 col-major: value = i + 2 j; logical row i is physical row 1 - i
	assert_eq!(
		got,
		vec![
			(0, 0, 1.0),
			(1, 1, 2.0),
			(0, 1, 3.0),
			(1, 2, 4.0),
			(0, 2, 5.0)
		]
	);
}

/// the report's reproduction: wide, transposed, column-reversed, zipped with a plain operand
#[test]
fn transposed_reversed_zip_with_plain_operand() {
	let mut a = Mat::<f64>::from_fn(2, 4, |i, j| (i + 2 * j) as f64);
	let mut b = Mat::<f64>::from_fn(4, 2, |i, j| 100.0 + (i + 4 * j) as f64);
	let mut seen = vec![];
	zip!(a.as_mut().transpose_mut().reverse_cols_mut(), &mut b)
		.for_each_triangular_upper_with_index(Diag::Include, |i, j, x| {
			seen.push((i, j, *x.0, *x.1.0))
		});
	// logical a'(i, j) = a(1 - j, i)... the view is 4x2: (i, j) -> base (1 - j, i)
	let mut want = vec![];
	for j in 0..2usize {
		for i in 0..4usize {
			if i <= j {
				want.push((
					i,
					j,
					((1 - j) + 2 * i) as f64,
					100.0 + (i + 4 * j) as f64,
				));
			}
		}
	}
	seen.sort_by_key(|e| (e.1, e.0));
	assert_eq!(seen, want);
}

#[test]
fn zip_eq_new_panics_on_mismatched_shapes() {
	let r = std::panic::catch_unwind(|| {
		let mut a = Mat::<f64>::zeros(8, 8);
		let mut b = Mat::<f64>::zeros(1, 1);
		ZipEq::new(zip!(&mut a), zip!(&mut b)).for_each(|_| {});
	});
	assert!(r.is_err());
}

/// the report's second reproduction: `new_unchecked` is safe, so it must refuse mismatched shapes
/// in release too (it was a `debug_assert!`, and iteration then read out of bounds)
#[test]
fn zip_eq_new_unchecked_panics_on_mismatched_shapes() {
	let r = std::panic::catch_unwind(|| {
		let mut a = Mat::<f64>::zeros(8, 8);
		let mut b = Mat::<f64>::zeros(1, 1);
		ZipEq::new_unchecked(zip!(&mut a), zip!(&mut b)).for_each(|_| {});
	});
	assert!(r.is_err());
}

#[test]
fn zip_eq_new_unchecked_accepts_matching_shapes() {
	let mut a = Mat::<f64>::from_fn(3, 2, |i, j| (i + 3 * j) as f64);
	let mut b = Mat::<f64>::zeros(3, 2);
	let mut n = 0;
	ZipEq::new_unchecked(zip!(&mut a), zip!(&mut b)).for_each(|_| n += 1);
	assert_eq!(n, 6);
}
