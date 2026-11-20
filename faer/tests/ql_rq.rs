//! Public contract of the packed QL and RQ factorizations (colmat-rs#60): runs
//! at the size of the scratch queries, every output mode into reused buffers, the
//! trapezoidal shape and the positive gauge of the triangular factor (and the
//! ungauged factors), the reported rank, rank-deficient and graded inputs, a
//! row-major input factored in place and strided outputs of every routine,
//! `Par::Seq` against a two-thread team, and the MatrixFactorizations.jl
//! reference factors (`fixtures/ql_rq_reference/`).
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::{ql, rq};
use colmat::traits::ComplexField;
use colmat::{Auto, Mat, MatMut, Par, ParShape, c32, c64};
mod common;
use common::*;

fn sample_mat<T: ComplexField>(m: usize, n: usize, seed: f64) -> Mat<T> {
	Mat::from_fn(m, n, |i, j| {
		T::from_f64(sample(i, j) + seed * ((i + 2 * j) as f64).sin())
	})
}

/// `sample_mat` shifted by 16 on the diagonal that ends in the last row and
/// column. The plain samples have condition numbers of 1e9-1e11, so two valid
/// computation orders (a row-major against a column-major input, a team against
/// `Par::Seq`) agree only to `cond * eps`; this keeps the condition number small
/// enough for an entry-by-entry comparison at `200 n eps`.
fn well_conditioned<T: ComplexField>(m: usize, n: usize) -> Mat<T> {
	let k = Ord::min(m, n);
	let mut a = sample_mat::<T>(m, n, 0.5);
	for r in 0..k {
		let (i, j) = (r + m - k, r + n - k);
		a[(i, j)] = a[(i, j)].clone() + T::from_f64(16.0);
	}
	a
}

fn block_size<T: ComplexField>(m: usize, n: usize) -> usize {
	let k = Ord::min(m, n);
	Ord::max(
		ql::no_pivoting::factor::recommended_block_size::<T>(m, n)
			.min(Ord::max(k, 1)),
		1,
	)
}

fn garbage<T: ComplexField>(m: usize, n: usize) -> Mat<T> {
	Mat::full(m, n, T::from_f64(-7.5))
}

fn assert_zero<T: ComplexField>(a: MatRef<'_, T>) {
	assert!(a.nrows() * a.ncols() == 0 || a.norm_l2() == zero());
}

/// The diagonal that ends in the last row and column is real and nonnegative.
fn assert_positive<T: ComplexField>(d: T) {
	assert!(d.real() >= zero());
	let bound = eps::<T::Real>()
		* from_f64::<T::Real>(4.0)
		* d.abs().fmax(one::<T::Real>());
	assert!(d.imag().abs() <= bound);
}

/// Factors `packed` in place as QL; returns the coefficients and the rank.
fn ql_pack_into<T: ComplexField>(
	packed: MatMut<'_, T>,
	block_size: usize,
	par: Par<'_>,
) -> (Mat<T>, usize) {
	use ql::no_pivoting::factor;
	let (m, n) = packed.shape();
	let params = <factor::QlParams as Auto<T>>::auto().into();
	let mut mem = MemBuffer::new(factor::ql_in_place_scratch::<T>(
		m,
		n,
		block_size,
		par.shape(),
		params,
	));
	let mut coeff =
		Mat::<T>::full(block_size, Ord::min(m, n), T::from_f64(f64::INFINITY));
	let info = factor::ql_in_place(
		packed,
		coeff.as_mut(),
		par,
		MemStack::new(&mut mem),
		params,
	);
	(coeff, info.rank)
}

/// Factors `packed` in place as RQ; returns the coefficients and the rank.
fn rq_pack_into<T: ComplexField>(
	packed: MatMut<'_, T>,
	block_size: usize,
	par: Par<'_>,
) -> (Mat<T>, usize) {
	use rq::no_pivoting::factor;
	let (m, n) = packed.shape();
	let params = <factor::RqParams as Auto<T>>::auto().into();
	let mut mem = MemBuffer::new(factor::rq_in_place_scratch::<T>(
		m,
		n,
		block_size,
		par.shape(),
		params,
	));
	let mut coeff =
		Mat::<T>::full(block_size, Ord::min(m, n), T::from_f64(f64::INFINITY));
	let info = factor::rq_in_place(
		packed,
		coeff.as_mut(),
		par,
		MemStack::new(&mut mem),
		params,
	);
	(coeff, info.rank)
}

/// QL packed form, coefficients and rank of `a` (factored in a column-major copy).
fn ql_pack<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	par: Par<'_>,
) -> (Mat<T>, Mat<T>, usize) {
	let mut packed = a.to_owned();
	let (coeff, rank) = ql_pack_into(packed.as_mut(), block_size, par);
	(packed, coeff, rank)
}

/// RQ packed form, coefficients and rank of `a` (factored in a column-major copy).
fn rq_pack<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	par: Par<'_>,
) -> (Mat<T>, Mat<T>, usize) {
	let mut packed = a.to_owned();
	let (coeff, rank) = rq_pack_into(packed.as_mut(), block_size, par);
	(packed, coeff, rank)
}

/// Thin `Q` (`m x k`) and compact `L` (`k x n`) of `a` in the positive gauge.
fn ql_factors<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	mut par: Par<'_>,
) -> (Mat<T>, Mat<T>) {
	let (packed, coeff, _) = ql_pack(a, block_size, par.reborrow());
	ql_extract(packed.as_ref(), coeff.as_ref(), block_size, par, true)
}

/// Thin `Q` and compact `L` from a packed QL factorization.
fn ql_extract<T: ComplexField>(
	packed: MatRef<'_, T>,
	coeff: MatRef<'_, T>,
	block_size: usize,
	par: Par<'_>,
	positive: bool,
) -> (Mat<T>, Mat<T>) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	let mut q = garbage::<T>(m, k);
	let mut l = garbage::<T>(k, n);
	let mut mem =
		MemBuffer::new(ql::no_pivoting::unitary::unitary_scratch::<T>(
			m,
			n,
			k,
			block_size,
			par.shape(),
		));
	ql::no_pivoting::unitary::unitary(
		q.as_mut(),
		packed,
		coeff,
		positive,
		par,
		MemStack::new(&mut mem),
	);
	ql::no_pivoting::factor::copy_l(l.as_mut(), packed, positive);
	(q, l)
}

/// Compact `R` (`m x k`) and thin `Q` (`k x n`) of `a` in the positive gauge.
fn rq_factors<T: ComplexField>(
	a: MatRef<'_, T>,
	block_size: usize,
	mut par: Par<'_>,
) -> (Mat<T>, Mat<T>) {
	let (packed, coeff, _) = rq_pack(a, block_size, par.reborrow());
	rq_extract(packed.as_ref(), coeff.as_ref(), block_size, par, true)
}

/// Compact `R` and thin `Q` from a packed RQ factorization.
fn rq_extract<T: ComplexField>(
	packed: MatRef<'_, T>,
	coeff: MatRef<'_, T>,
	block_size: usize,
	par: Par<'_>,
	positive: bool,
) -> (Mat<T>, Mat<T>) {
	let (m, n) = packed.shape();
	let k = Ord::min(m, n);
	let mut r = garbage::<T>(m, k);
	let mut q = garbage::<T>(k, n);
	let mut mem =
		MemBuffer::new(rq::no_pivoting::unitary::unitary_scratch::<T>(
			m,
			n,
			k,
			block_size,
			par.shape(),
		));
	rq::no_pivoting::unitary::unitary(
		q.as_mut(),
		packed,
		coeff,
		positive,
		par,
		MemStack::new(&mut mem),
	);
	rq::no_pivoting::factor::copy_r(r.as_mut(), packed, positive);
	(r, q)
}

/// Every QL output mode at the exact scratch size, into reused buffers.
fn ql_contract<T: ComplexField>(a: MatRef<'_, T>) {
	use ql::no_pivoting::{factor, reconstruct, unitary};
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let bs = block_size::<T>(m, n);
	let (packed, coeff, _) = ql_pack(a, bs, Par::Seq);
	let (q, l) = ql_factors(a, bs, Par::Seq);
	// without the gauge: still a factorization with orthonormal columns
	let (q0, l0) =
		ql_extract(packed.as_ref(), coeff.as_ref(), bs, Par::Seq, false);
	close((&q0 * &l0).as_ref(), a);
	unitary(q0.as_ref());
	let mut l_full = garbage::<T>(m, n);
	let mut q_full = garbage::<T>(m, m);
	let mut rebuilt = garbage::<T>(m, n);
	factor::copy_l(l_full.as_mut(), packed.as_ref(), true);
	unitary::unitary(
		q_full.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(unitary::unitary_scratch::<T>(
			m,
			n,
			m,
			bs,
			ParShape::Seq,
		))),
	);
	reconstruct::reconstruct(
		rebuilt.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(
			reconstruct::reconstruct_scratch::<T>(m, n, bs, ParShape::Seq),
		)),
	);

	// $A = QL$, thin and full, and the packed rebuild.
	close((&q * &l).as_ref(), a);
	close((&q_full * &l_full).as_ref(), a);
	close(rebuilt.as_ref(), a);
	unitary(q.as_ref());
	unitary(q_full.as_ref());
	// the thin factors are the trailing columns of the full Q and the trailing
	// rows of the full L, whose leading rows are zero.
	close(q_full.as_ref().get(.., m - k..), q.as_ref());
	close(l_full.as_ref().get(m - k.., ..), l.as_ref());
	assert_zero(l_full.as_ref().get(..m - k, ..));
	// lower trapezoidal, ending in the last row and column, with a positive
	// diagonal.
	for r in 0..k {
		assert_zero(l.as_ref().get(r..r + 1, r + n - k + 1..));
		assert_positive(l[(r, r + n - k)].clone());
	}
}

/// Every RQ output mode at the exact scratch size, into reused buffers.
fn rq_contract<T: ComplexField>(a: MatRef<'_, T>) {
	use rq::no_pivoting::{factor, reconstruct, unitary};
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let bs = block_size::<T>(m, n);
	let (packed, coeff, _) = rq_pack(a, bs, Par::Seq);
	let (r, q) = rq_factors(a, bs, Par::Seq);
	// without the gauge: still a factorization with orthonormal rows
	let (r0, q0) =
		rq_extract(packed.as_ref(), coeff.as_ref(), bs, Par::Seq, false);
	close((&r0 * &q0).as_ref(), a);
	unitary(q0.transpose());
	let mut r_full = garbage::<T>(m, n);
	let mut q_full = garbage::<T>(n, n);
	let mut rebuilt = garbage::<T>(m, n);
	factor::copy_r(r_full.as_mut(), packed.as_ref(), true);
	unitary::unitary(
		q_full.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		true,
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(unitary::unitary_scratch::<T>(
			m,
			n,
			n,
			bs,
			ParShape::Seq,
		))),
	);
	reconstruct::reconstruct(
		rebuilt.as_mut(),
		packed.as_ref(),
		coeff.as_ref(),
		Par::Seq,
		MemStack::new(&mut MemBuffer::new(
			reconstruct::reconstruct_scratch::<T>(m, n, bs, ParShape::Seq),
		)),
	);

	// $A = RQ$, thin and full, and the packed rebuild.
	close((&r * &q).as_ref(), a);
	close((&r_full * &q_full).as_ref(), a);
	close(rebuilt.as_ref(), a);
	close(
		(&q * q.adjoint()).as_ref(),
		Mat::<T>::identity(k, k).as_ref(),
	);
	close(
		(&q_full * q_full.adjoint()).as_ref(),
		Mat::<T>::identity(n, n).as_ref(),
	);
	// the thin factors are the trailing rows of the full Q and the trailing
	// columns of the full R, whose leading columns are zero.
	close(q_full.as_ref().get(n - k.., ..), q.as_ref());
	close(r_full.as_ref().get(.., n - k..), r.as_ref());
	assert_zero(r_full.as_ref().get(.., ..n - k));
	// upper trapezoidal, ending in the last row and column, with a positive
	// diagonal.
	for c in 0..k {
		assert_zero(r.as_ref().get(c + m - k + 1.., c..c + 1));
		assert_positive(r[(c + m - k, c)].clone());
	}
}

const SHAPES: [(usize, usize); 9] = [
	(0, 0),
	(0, 3),
	(3, 0),
	(1, 1),
	(8, 8),
	(8, 3),
	(3, 8),
	(100, 33),
	(33, 100),
];

fn contracts<T: ComplexField>() {
	for (m, n) in SHAPES {
		let a = sample_mat::<T>(m, n, 0.0);
		ql_contract(a.as_ref());
		rq_contract(a.as_ref());
	}
	// rank deficient: a repeated column and a zero row
	let mut a = sample_mat::<T>(9, 6, 0.5);
	for i in 0..9 {
		a[(i, 4)] = a[(i, 1)].clone();
		if i < 6 {
			a[(3, i)] = zero();
		}
	}
	for b in [a.as_ref(), a.transpose()] {
		ql_contract(b);
		rq_contract(b);
		// the true rank is 5. The unpivoted kernel's estimate is an upper bound: the
		// repeated column leaves a rounding-level residual, not an exact zero, so it
		// reports 6 here (measured for all four dtypes).
		let bs = block_size::<T>(b.nrows(), b.ncols());
		for rank in [ql_pack(b, bs, Par::Seq).2, rq_pack(b, bs, Par::Seq).2] {
			assert!((5..=6).contains(&rank), "rank estimate {rank}");
		}
	}
	// graded rows, twelve orders of magnitude apart
	let graded = Mat::<T>::from_fn(12, 7, |i, j| {
		T::from_f64(sample(i, j) * 10f64.powi(-(i as i32)))
	});
	for b in [graded.as_ref(), graded.transpose()] {
		ql_contract(b);
		rq_contract(b);
	}
}

#[test]
fn ql_rq_contract() {
	contracts::<f64>();
	contracts::<f32>();
	contracts::<c64>();
	contracts::<c32>();
}

const GUARD: f64 = 2.5;

/// An `m x n` window at `(1, 2)` of a buffer filled with `GUARD`, so the
/// window has a column stride larger than `m`.
fn guarded<T: ComplexField>(m: usize, n: usize) -> Mat<T> {
	Mat::full(m + 3, n + 2, T::from_f64(GUARD))
}

/// The window holds `expected` and every entry outside it is still `GUARD`.
fn check_window<T: ComplexField>(buffer: &Mat<T>, expected: MatRef<'_, T>) {
	let (m, n) = expected.shape();
	close(buffer.as_ref().submatrix(1, 2, m, n), expected);
	for j in 0..buffer.ncols() {
		for i in 0..buffer.nrows() {
			let inside = (1..1 + m).contains(&i) && (2..2 + n).contains(&j);
			if !inside {
				assert_eq!(
					buffer[(i, j)],
					T::from_f64(GUARD),
					"guard ({i}, {j})"
				);
			}
		}
	}
}

/// A row-major input factored in place, and strided outputs of every routine,
/// give the factors of the column-major path.
fn views<T: ComplexField>(m: usize, n: usize) {
	let k = Ord::min(m, n);
	let bs = block_size::<T>(m, n);
	let a = well_conditioned::<T>(m, n);

	// `storage` is A^T column-major, so its transposed view is A in row-major
	// layout, and the factorization runs on that layout.
	let (q, l) = ql_factors(a.as_ref(), bs, Par::Seq);
	let mut storage = a.transpose().to_owned();
	let (coeff, _) =
		ql_pack_into(storage.as_mut().transpose_mut(), bs, Par::Seq);
	let (q_rm, l_rm) =
		ql_extract(storage.transpose(), coeff.as_ref(), bs, Par::Seq, true);
	close(q_rm.as_ref(), q.as_ref());
	close(l_rm.as_ref(), l.as_ref());
	let (r, qr) = rq_factors(a.as_ref(), bs, Par::Seq);
	let mut storage = a.transpose().to_owned();
	let (coeff, _) =
		rq_pack_into(storage.as_mut().transpose_mut(), bs, Par::Seq);
	let (r_rm, qr_rm) =
		rq_extract(storage.transpose(), coeff.as_ref(), bs, Par::Seq, true);
	close(r_rm.as_ref(), r.as_ref());
	close(qr_rm.as_ref(), qr.as_ref());

	// strided outputs inside guarded buffers
	{
		use ql::no_pivoting::{factor, reconstruct, unitary};
		let (packed, coeff, _) = ql_pack(a.as_ref(), bs, Par::Seq);
		let mut q_out = guarded::<T>(m, k);
		unitary::unitary(
			q_out.as_mut().submatrix_mut(1, 2, m, k),
			packed.as_ref(),
			coeff.as_ref(),
			true,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(unitary::unitary_scratch::<T>(
				m,
				n,
				k,
				bs,
				ParShape::Seq,
			))),
		);
		check_window(&q_out, q.as_ref());
		let mut l_out = guarded::<T>(k, n);
		factor::copy_l(
			l_out.as_mut().submatrix_mut(1, 2, k, n),
			packed.as_ref(),
			true,
		);
		check_window(&l_out, l.as_ref());
		let mut rebuilt = guarded::<T>(m, n);
		reconstruct::reconstruct(
			rebuilt.as_mut().submatrix_mut(1, 2, m, n),
			packed.as_ref(),
			coeff.as_ref(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				reconstruct::reconstruct_scratch::<T>(m, n, bs, ParShape::Seq),
			)),
		);
		check_window(&rebuilt, a.as_ref());
	}
	{
		use rq::no_pivoting::{factor, reconstruct, unitary};
		let (packed, coeff, _) = rq_pack(a.as_ref(), bs, Par::Seq);
		let mut q_out = guarded::<T>(k, n);
		unitary::unitary(
			q_out.as_mut().submatrix_mut(1, 2, k, n),
			packed.as_ref(),
			coeff.as_ref(),
			true,
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(unitary::unitary_scratch::<T>(
				m,
				n,
				k,
				bs,
				ParShape::Seq,
			))),
		);
		check_window(&q_out, qr.as_ref());
		let mut r_out = guarded::<T>(m, k);
		factor::copy_r(
			r_out.as_mut().submatrix_mut(1, 2, m, k),
			packed.as_ref(),
			true,
		);
		check_window(&r_out, r.as_ref());
		let mut rebuilt = guarded::<T>(m, n);
		reconstruct::reconstruct(
			rebuilt.as_mut().submatrix_mut(1, 2, m, n),
			packed.as_ref(),
			coeff.as_ref(),
			Par::Seq,
			MemStack::new(&mut MemBuffer::new(
				reconstruct::reconstruct_scratch::<T>(m, n, bs, ParShape::Seq),
			)),
		);
		check_window(&rebuilt, a.as_ref());
	}
}

/// The reported rank: `min(m, n)` for a full-rank input, 0 for the zero matrix.
fn ranks<T: ComplexField>(m: usize, n: usize) {
	let k = Ord::min(m, n);
	let bs = block_size::<T>(m, n);
	let a = sample_mat::<T>(m, n, 0.5);
	assert_eq!(ql_pack(a.as_ref(), bs, Par::Seq).2, k);
	assert_eq!(rq_pack(a.as_ref(), bs, Par::Seq).2, k);
	let zero_mat = Mat::<T>::zeros(m, n);
	assert_eq!(ql_pack(zero_mat.as_ref(), bs, Par::Seq).2, 0);
	assert_eq!(rq_pack(zero_mat.as_ref(), bs, Par::Seq).2, 0);
}

#[test]
fn ql_rq_report_the_rank() {
	for (m, n) in [(8, 3), (3, 8), (8, 8), (100, 33)] {
		ranks::<f64>(m, n);
		ranks::<c64>(m, n);
	}
}

#[test]
fn ql_rq_views() {
	for (m, n) in [(8, 3), (3, 8), (8, 8), (100, 33)] {
		views::<f64>(m, n);
		views::<c64>(m, n);
	}
}

#[cfg(feature = "threadteam")]
#[test]
fn ql_rq_parallel_matches_sequential() {
	fn agree<T: ComplexField>(m: usize, n: usize) {
		let a = well_conditioned::<T>(m, n);
		let bs = Ord::max(block_size::<T>(m, n) / 2, 1);
		let (q_seq, l_seq) = ql_factors(a.as_ref(), bs, Par::Seq);
		let (r_seq, qr_seq) = rq_factors(a.as_ref(), bs, Par::Seq);
		colmat::team::with_team(2, |mut par: Par<'_>| {
			let (q, l) = ql_factors(a.as_ref(), bs, par.reborrow());
			close(q.as_ref(), q_seq.as_ref());
			close(l.as_ref(), l_seq.as_ref());
			let (r, qr) = rq_factors(a.as_ref(), bs, par);
			close(r.as_ref(), r_seq.as_ref());
			close(qr.as_ref(), qr_seq.as_ref());
		});
	}
	for (m, n) in [(257, 129), (129, 257), (257, 257)] {
		agree::<f64>(m, n);
		agree::<c64>(m, n);
	}
}

/// One case of `fixtures/ql_rq_reference/ql_rq.txt`.
struct Case {
	kind: String,
	complex: bool,
	mats: Vec<(String, Mat<f64>)>,
}

fn parse(text: &str) -> Vec<Case> {
	let mut cases: Vec<Case> = Vec::new();
	let mut lines = text.lines().filter(|l| !l.starts_with('#'));
	while let Some(line) = lines.next() {
		let words: Vec<&str> = line.split_whitespace().collect();
		if words[0] == "case" {
			cases.push(Case {
				kind: words[1].to_string(),
				complex: words[2] == "c64",
				mats: Vec::new(),
			});
			continue;
		}
		let (rows, cols): (usize, usize) =
			(words[1].parse().unwrap(), words[2].parse().unwrap());
		let mut mat = Mat::<f64>::zeros(rows, cols);
		for i in 0..rows {
			let values: Vec<f64> = lines
				.next()
				.unwrap()
				.split_whitespace()
				.map(|v| v.parse().unwrap())
				.collect();
			assert_eq!(values.len(), cols);
			for (j, v) in values.into_iter().enumerate() {
				mat[(i, j)] = v;
			}
		}
		cases
			.last_mut()
			.unwrap()
			.mats
			.push((words[0].to_string(), mat));
	}
	cases
}

/// The named matrix of `case`, its entries built by `scalar(re, im)`.
fn get<T: ComplexField>(
	case: &Case,
	name: &str,
	scalar: &impl Fn(f64, f64) -> T,
) -> Mat<T> {
	let find = |key: String| {
		case.mats
			.iter()
			.find(|(n, _)| *n == key)
			.map(|(_, m)| m.clone())
			.unwrap()
	};
	let (re, im) = if case.complex {
		(find(format!("{name}_re")), find(format!("{name}_im")))
	} else {
		let re = find(name.to_string());
		let im = Mat::zeros(re.nrows(), re.ncols());
		(re, im)
	};
	Mat::from_fn(re.nrows(), re.ncols(), |i, j| {
		scalar(re[(i, j)], im[(i, j)])
	})
}

/// With the positive gauge a full-rank factorization is unique, so both factors
/// match MatrixFactorizations.jl entry by entry.
fn matches_reference<T: ComplexField>(
	case: &Case,
	scalar: impl Fn(f64, f64) -> T,
) {
	let a = get(case, "a", &scalar);
	let (q_ref, t_ref) = (get(case, "q", &scalar), get(case, "t", &scalar));
	let bs = block_size::<T>(a.nrows(), a.ncols());
	if case.kind == "ql" {
		let (q, l) = ql_factors(a.as_ref(), bs, Par::Seq);
		close(q.as_ref(), q_ref.as_ref());
		close(l.as_ref(), t_ref.as_ref());
	} else {
		let (r, q) = rq_factors(a.as_ref(), bs, Par::Seq);
		close(q.as_ref(), q_ref.as_ref());
		close(r.as_ref(), t_ref.as_ref());
	}
}

#[test]
fn ql_rq_match_matrixfactorizations_jl() {
	let cases = parse(include_str!("fixtures/ql_rq_reference/ql_rq.txt"));
	assert_eq!(cases.len(), 12);
	for case in &cases {
		if case.complex {
			matches_reference(case, c64::new);
		} else {
			matches_reference(case, |re, _| re);
		}
	}
}
