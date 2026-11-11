//! Where the time of `partial_piv_lu` (the owning route) goes, versus the in-place route the
//! campaign times beside it, on the same square matrix.
//!
//! One cell per process: `--threads T --dtype f64|c64 --order c|f --n N --reps 5 --prime-ms 250`.
//! Every phase is primed and timed with the row's parallelism held around the loop, the way the
//! campaign's harness holds it (`faer::team::with_team`), and each records its median wall time
//! and the minor page faults it took.
//!
//! The phases, all on the same input and the same thread budget (the record's table C prices them
//! in its own order; this is the order they are emitted in):
//!
//!   owning                  `A.partial_piv_lu(par)` - the public owning route, the campaign's row
//!   owning_no_split         `A.to_owned()` then `lu_in_place` on it: `new_imp` before `split_LU`
//!   owning_u_first          `new_imp` + the split, hand-copied, with `U`'s buffer taken first
//!   factor_only             `lu_in_place` on a prepared column-major buffer: the factorization
//!   clone_only              `A.to_owned()` alone
//!   alloc_zeros             `Mat::zeros(n, n)` alone
//!   clone_and_alloc_only    both buffers alive, nothing written: what the split's allocation costs
//!   alloc_then_clone        the same pair in the other order
//!   split_colslices         the `m >= n` branch of `split_LU` with one contiguous span per column
//!   split_colwise           the same, written through `MatIndex`
//!   split_upper_only        `copy_from_triangular_upper` alone, plus `U`'s allocation
//!   split_zero_only         `for_each_triangular_upper` alone, plus `U`'s allocation
//!   split_rowwise           the same split with row-wise copies: NOT a proposal, a row is strided
//!   split_std_threads       a hand-spawned 8-thread split: diagnostic only, the library must use
//!                           its own team, so this is a bound rather than a proposal
//!   copy_upper_slices_only  the split's copies alone, on warm buffers
//!   zero_upper_slices_only  the split's zeroing alone, on warm buffers
//!
//! `owning - owning_no_split` is the split plus the permutation boxes and the per-call scratch;
//! `clone_and_alloc_only` is what it costs before anything is written, which is where the cost is
//! on a host whose allocator hands out fresh pages; `copy_upper_slices_only` and
//! `zero_upper_slices_only` are what the same work costs on warm ones.
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::factor as plu;
use faer::prelude::*;
use faer::{Mat, MatMut, MatRef};
use num_complex::Complex64;

trait Fill: Copy {
    fn value(i: u64) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}
impl Fill for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn value(i: u64) -> Self {
        // a cheap deterministic spread in [-1, 1), no dependency on a rng
        let x = (i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 11)
            as f64
            / (1u64 << 53) as f64;
        x * 2.0 - 1.0
    }
}
impl Fill for Complex64 {
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn value(i: u64) -> Self {
        Complex64::new(f64::value(i), f64::value(i ^ 0x9e3779b9))
    }
}

/// probe-only hand-off of a buffer into `std::thread::scope`
struct Ptr<T>(*mut T);
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}
impl<T> Copy for Ptr<T> {}
impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Ptr<T> {
    /// # Safety
    /// the caller must guarantee the pointee is alive and that no other live view aliases it
    unsafe fn as_mut(&self) -> &mut T {
        unsafe { &mut *self.0 }
    }
}

fn strides(m: usize, n: usize, order: &str) -> (isize, isize) {
    if order == "c" {
        (n as isize, 1)
    } else {
        (1, m as isize)
    }
}

/// minor page faults of this process so far (`/proc/self/stat`, field 10), for saying whether a
/// phase's cost is memory traffic or the mapping of a fresh buffer
fn minor_faults() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
    let tail = stat.rsplit_once(") ").unwrap().1;
    tail.split_whitespace().nth(7).unwrap().parse().unwrap()
}

fn timed(prime_ms: u64, reps: usize, mut f: impl FnMut()) -> (Vec<f64>, Vec<u64>) {
    let until = Instant::now() + Duration::from_millis(prime_ms);
    while Instant::now() < until {
        f();
    }
    let mut samples = Vec::with_capacity(reps);
    let mut faults = Vec::with_capacity(reps);
    for _ in 0..reps {
        let before = minor_faults();
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
        faults.push(minor_faults() - before);
    }
    (samples, faults)
}

fn median(mut s: Vec<f64>) -> f64 {
    s.sort_by(f64::total_cmp);
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[mid - 1] + s[mid]) * 0.5
    } else {
        s[mid]
    }
}

/// `faer/src/linalg/solvers.rs::split_LU`'s `m >= n` branch with row-wise contiguous copies:
/// `L`'s strict upper triangle zeroed and `U`'s upper triangle filled one contiguous row span at
/// a time. Same values in the same buffers: what the proposal would pay.
fn split_rowwise<T: faer::traits::ComplexField + Fill>(mut l: Mat<T>) -> (Mat<T>, Mat<T>) {
    let (m, n) = l.shape();
    let size = Ord::min(m, n);
    let mut u = Mat::zeros(size, size);
    for i in 0..size {
        {
            let mut zero = l.rb_mut().row_mut(i).get_mut(i + 1..);
            zero.fill(T::zero());
        }
        let src = l.rb().row(i).get(i..);
        let mut dst = u.rb_mut().row_mut(i).get_mut(i..);
        dst.copy_from(src);
    }
    l.diagonal_mut().fill(T::one());
    (l, u)
}


/// The proposal, column-wise: an owned `Mat` is column-major (`Mat::row_stride` is 1), so one
/// column's triangle is one contiguous span. `L`'s strict upper is zeroed and `U`'s upper is
/// copied, one column at a time.
fn split_colwise<T: faer::traits::ComplexField + Fill>(mut l: Mat<T>) -> (Mat<T>, Mat<T>) {
    let (m, n) = l.shape();
    let size = Ord::min(m, n);
    let mut u = Mat::zeros(size, size);
    for j in 0..size {
        {
            let src = l.rb().col(j).get(j..);
            let mut dst = u.rb_mut().col_mut(j).get_mut(j..);
            dst.copy_from(src);
        }
        {
            let mut zero = l.rb_mut().col_mut(j).get_mut(j + 1..);
            zero.fill(T::zero());
        }
    }
    l.diagonal_mut().fill(T::one());
    (l, u)
}

/// `copy_from_triangular_upper` alone (plus the allocation of `U`), i.e. the other half of the
/// current split.
fn split_upper_only<T: faer::traits::ComplexField + Fill>(l: Mat<T>) -> (Mat<T>, Mat<T>) {
    let (m, n) = l.shape();
    let size = Ord::min(m, n);
    let mut u = Mat::zeros(size, size);
    u.copy_from_triangular_upper(l.get(..size, ..size));
    (l, u)
}

/// `for_each_triangular_upper(Diag::Skip)` zeroing alone, as `split_LU` writes it today.
fn split_zero_only<T: faer::traits::ComplexField + Fill>(mut l: Mat<T>) -> (Mat<T>, Mat<T>) {
    let (m, n) = l.shape();
    let size = Ord::min(m, n);
    let u = Mat::<T>::zeros(size, size);
    {
        let mut l = l.rb_mut().get_mut(..size, ..size);
        faer::zip!(&mut l).for_each_triangular_upper(
            faer::linalg::zip::Diag::Skip,
            |faer::unzip!(x)| {
                *x = T::zero();
            },
        );
    }
    (l, u)
}


/// The same split through the owned matrix's own column slices (`Mat::col_as_slice`), which are
/// real `&[T]` spans: `copy_from_slice` and `fill` per column, no `MatIndex` per element.
fn split_colslices<T: faer::traits::ComplexField + Fill>(mut l: Mat<T>) -> (Mat<T>, Mat<T>) {
    let (m, n) = l.shape();
    let size = Ord::min(m, n);
    let mut u = Mat::zeros(size, size);
    for j in 0..size {
        {
            let src = &l.col_as_slice(j)[j..size];
            u.col_as_slice_mut(j)[j..size].copy_from_slice(src);
        }
        l.col_as_slice_mut(j)[j + 1..size].fill(T::zero());
    }
    l.diagonal_mut().fill(T::one());
    (l, u)
}

fn run<T: faer::traits::ComplexField + Fill + Send + Sync + 'static>(
    out: &mut String,
    dtype: &str,
    order: &str,
    m: usize,
    threads: usize,
    reps: usize,
    prime_ms: u64,
) {
    let n = m;
    let size = m.min(n);
    let data: Vec<T> = (0..m * n).map(|i| T::value(i as u64)).collect();
    let (rs, cs) = strides(m, n, order);
    // SAFETY: `data` outlives every view made from it, and the strides describe it.
    let a = || unsafe { MatRef::from_raw_parts(data.as_ptr(), m, n, rs, cs) };

    let row = |phase: &str, (samples, faults): (Vec<f64>, Vec<u64>), out: &mut String| {
        let mut f = faults.clone();
        f.sort_unstable();
        let _ = writeln!(
            out,
            "{phase},{dtype},{order},{m},{threads},{:.6},{},\"{samples:?}\",\"{faults:?}\"",
            median(samples.clone()),
            f[f.len() / 2],
        );
    };

    faer::team::with_team(threads, |mut par| {
        let req = plu::lu_in_place_scratch::<usize, T>(m, n, par.shape(), Default::default());
        let params = || Default::default();
        let mut perm = vec![0usize; m];
        let mut perm_inv = vec![0usize; m];

        // one factored buffer (the split phases read it) and one to factor in place
        let mut fact = a().to_owned();
        let mut buf = MemBuffer::new(req);
        plu::lu_in_place(
            fact.as_mut(),
            &mut perm,
            &mut perm_inv,
            par.reborrow(),
            MemStack::new(&mut buf),
            params(),
        );
        let fact_ref = || fact.rb();
        let mut cm: Vec<T> = vec![T::value(0); m * n];
        MatMut::from_column_major_slice_mut(&mut cm, m, n).copy_from(a());
        let mut buf_cm = MemBuffer::new(req);
        let mut l2 = Mat::<T>::zeros(size, size);
        let mut u2 = Mat::<T>::zeros(size, size);

        row(
            "owning",
            timed(prime_ms, reps, || {
                let lu = a().partial_piv_lu(par.reborrow());
                drop(lu);
            }),
            out,
        );
        row(
            "owning_no_split",
            timed(prime_ms, reps, || {
                let mut owned = a().to_owned();
                let mut buf = MemBuffer::new(req);
                plu::lu_in_place(
                    owned.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    par.reborrow(),
                    MemStack::new(&mut buf),
                    params(),
                );
                drop(owned);
            }),
            out,
        );
        row(
            "clone_only",
            timed(prime_ms, reps, || {
                let l = fact_ref().to_owned();
                drop(l);
            }),
            out,
        );
        row(
            "split_rowwise",
            timed(prime_ms, reps, || {
                let l = fact_ref().to_owned();
                let (l, u) = split_rowwise(l);
                debug_assert_eq!((l.shape(), u.shape(), size), ((m, n), (size, size), size));
                drop((l, u));
            }),
            out,
        );
        row(
            "split_colwise",
            timed(prime_ms, reps, || {
                let l = fact_ref().to_owned();
                let (l, u) = split_colwise(l);
                drop((l, u));
            }),
            out,
        );
        row(
            "split_upper_only",
            timed(prime_ms, reps, || {
                let l = fact_ref().to_owned();
                let (l, u) = split_upper_only(l);
                drop((l, u));
            }),
            out,
        );
        row(
            "split_zero_only",
            timed(prime_ms, reps, || {
                let l = fact_ref().to_owned();
                let (l, u) = split_zero_only(l);
                drop((l, u));
            }),
            out,
        );
        row(
            "split_colslices",
            timed(prime_ms, reps, || {
                let l = fact_ref().to_owned();
                let (l, u) = split_colslices(l);
                drop((l, u));
            }),
            out,
        );
        row(
            "alloc_zeros",
            timed(prime_ms, reps, || {
                let u = Mat::<T>::zeros(size, size);
                drop(u);
            }),
            out,
        );
        row(
            "zero_upper_slices_only",
            timed(prime_ms, reps, || {
                for j in 0..size {
                    l2.col_as_slice_mut(j)[j + 1..size].fill(T::zero());
                }
            }),
            out,
        );
        row(
            "copy_upper_slices_only",
            timed(prime_ms, reps, || {
                for j in 0..size {
                    u2.col_as_slice_mut(j)[j..size]
                        .copy_from_slice(&l2.col_as_slice(j)[j..size]);
                }
            }),
            out,
        );
        row(
            "clone_and_alloc_only",
            timed(prime_ms, reps, || {
                let l = fact_ref().to_owned();
                let u = Mat::<T>::zeros(size, size);
                drop((l, u));
            }),
            out,
        );
        row(
            "alloc_then_clone",
            timed(prime_ms, reps, || {
                let u = Mat::<T>::zeros(size, size);
                let l = fact_ref().to_owned();
                drop((l, u));
            }),
            out,
        );
        row(
            "owning_u_first",
            timed(prime_ms, reps, || {
                // a hand copy of `new_imp` + `split_LU`, with `U`'s buffer taken before the
                // factor buffer instead of after the factorization
                let mut u = Mat::<T>::zeros(size, size);
                let mut l = a().to_owned();
                let mut buf = MemBuffer::new(req);
                plu::lu_in_place(
                    l.as_mut(),
                    &mut perm,
                    &mut perm_inv,
                    par.reborrow(),
                    MemStack::new(&mut buf),
                    params(),
                );
                for j in 0..size {
                    {
                        let src = &l.col_as_slice(j)[j..size];
                        u.col_as_slice_mut(j)[j..size].copy_from_slice(src);
                    }
                    l.col_as_slice_mut(j)[j + 1..size].fill(T::zero());
                }
                l.diagonal_mut().fill(T::one());
                drop((l, u));
            }),
            out,
        );
        row(
            "split_std_threads",
            timed(prime_ms, reps, || {
                // diagnostic only: the library must use its own team, not `std::thread`
                let mut l = fact_ref().to_owned();
                let mut u = Mat::<T>::zeros(size, size);
                let k = threads.max(1);
                std::thread::scope(|s| {
                    let chunk = size.div_ceil(k);
                    let lp = Ptr(&mut l as *mut Mat<T>);
                    let up = Ptr(&mut u as *mut Mat<T>);
                    for c in 0..k {
                        let (lo, hi) = (c * chunk, Ord::min((c + 1) * chunk, size));
                        s.spawn(move || {
                            // SAFETY: the column ranges are disjoint, so the two views never
                            // alias; `l` and `u` outlive the scope.
                            let l: &mut Mat<T> = unsafe { lp.as_mut() };
                            let u: &mut Mat<T> = unsafe { up.as_mut() };
                            for j in lo..hi {
                                {
                                    let src = &l.col_as_slice(j)[j..size];
                                    u.col_as_slice_mut(j)[j..size].copy_from_slice(src);
                                }
                                l.col_as_slice_mut(j)[j + 1..size].fill(T::zero());
                            }
                        });
                    }
                });
                l.diagonal_mut().fill(T::one());
                drop((l, u));
            }),
            out,
        );
        row(
            "factor_only",
            timed(prime_ms, reps, || {
                plu::lu_in_place(
                    MatMut::from_column_major_slice_mut(&mut cm, m, n),
                    &mut perm,
                    &mut perm_inv,
                    par.reborrow(),
                    MemStack::new(&mut buf_cm),
                    params(),
                );
            }),
            out,
        );
    });
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut threads = 1usize;
    let mut dtype = "f64".to_owned();
    let mut order = "c".to_owned();
    let mut n = 512usize;
    let mut reps = 5usize;
    let mut prime_ms = 250u64;
    let mut csv = None;
    while let Some(flag) = args.next() {
        let mut value = || args.next().expect("missing value");
        match flag.as_str() {
            "--threads" => threads = value().parse().unwrap(),
            "--dtype" => dtype = value(),
            "--order" => order = value(),
            "--n" => n = value().parse().unwrap(),
            "--reps" => reps = value().parse().unwrap(),
            "--prime-ms" => prime_ms = value().parse().unwrap(),
            "--csv" => csv = Some(value()),
            other => panic!("unknown flag {other}"),
        }
    }
    let path = csv.expect("--csv is required");
    let mut out = String::new();
    out.push_str(
        "phase,dtype,order,n,threads,median_ms,median_minor_faults,\"samples_ms\",\"faults\"\n",
    );
    let mut s = String::new();
    if dtype == "f64" {
        run::<f64>(&mut s, &dtype, &order, n, threads, reps, prime_ms);
    } else {
        run::<Complex64>(&mut s, &dtype, &order, n, threads, reps, prime_ms);
    }
    out.push_str(&s);
    std::fs::write(&path, out).unwrap();
    println!("wrote {path}");
}
