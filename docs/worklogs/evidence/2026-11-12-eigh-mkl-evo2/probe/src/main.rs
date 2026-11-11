//! tensor4all/faer#391 increment 4 probe: one cell per process, the fork and MKL in one binary.
//!
//! usage: u391 <lib> <mode> <dtype> <n> <threads> <thr> [prime_ms] [window_ms]
//!   lib    fork | mkl
//!   mode   call   - the whole `eigh` call: fork `self_adjoint_evd` (in place, includes the
//!                   tridiagonalization and the reflector apply) against MKL `?syevd`
//!          mstage - MKL `?sytrd` and `?stedc` timed apart on the same tridiagonal (f64 only)
//!   dtype  f64 | c64
//!   thr    default | <usize>   (fork only: `SelfAdjointEvdParams::recursion_threshold`)
//!
//! Prints CSV rows `key,level,phase,value` on stdout. Timing values are per call: the window's
//! calls accumulated and divided by their count.
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{
	ComputeEigenvectors, SelfAdjointEvdParams, self_adjoint_evd, self_adjoint_evd_scratch,
};
use faer::traits::ComplexField;
use faer::{Mat, MatRef, Par, ParShape, Spec, c64};
use std::time::{Duration, Instant};

// ------------------------------------------------------------------ MKL (Link: see .cargo/config.toml)

#[link(name = "mkl_rt")]
extern "C" {
	fn LAPACKE_dsyevd(layout: i32, jobz: i8, uplo: i8, n: i32, a: *mut f64, lda: i32, w: *mut f64)
	-> i32;
	fn LAPACKE_zheevd(layout: i32, jobz: i8, uplo: i8, n: i32, a: *mut c64, lda: i32, w: *mut f64)
	-> i32;
	fn LAPACKE_dsytrd(
		layout: i32,
		uplo: i8,
		n: i32,
		a: *mut f64,
		lda: i32,
		d: *mut f64,
		e: *mut f64,
		tau: *mut f64,
	) -> i32;
	fn LAPACKE_dstedc(
		layout: i32,
		compz: i8,
		n: i32,
		d: *mut f64,
		e: *mut f64,
		z: *mut f64,
		ldz: i32,
	) -> i32;
	fn mkl_get_max_threads() -> i32;
	fn omp_get_max_threads() -> i32;
}
// the Intel OpenMP runtime `libmkl_intel_thread` runs on; linked so its symbols are resident
#[link(name = "iomp5")]
extern "C" {}
// `libmkl_core` resolves `log` from libm at load time, and an `--as-needed` link of a binary that
// never calls libm drops it, so the load fails with `undefined symbol: log`; one real reference
// keeps it in the NEEDED list.
#[link(name = "m")]
extern "C" {
	fn log(x: f64) -> f64;
}
const COL_MAJOR: i32 = 102;
const JOBZ_V: i8 = b'V' as i8;
const UPLO_L: i8 = b'L' as i8;
const COMPZ_I: i8 = b'I' as i8;

// ------------------------------------------------------------------ scalars

trait Scalar: ComplexField<Real = f64> + Copy {
	fn parts(self) -> (f64, f64);
	fn sample(st: &mut u64) -> Self;
	fn herm(self) -> Self;
	fn re_only(self) -> Self;
	fn zero_of() -> Self;
	fn syevd(buf: &mut [Self], n: usize, w: &mut [f64]) -> i32;
}
fn unif(st: &mut u64) -> f64 {
	*st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
	((*st >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}
impl Scalar for f64 {
	fn parts(self) -> (f64, f64) {
		(self, 0.0)
	}
	fn sample(st: &mut u64) -> Self {
		unif(st)
	}
	fn herm(self) -> Self {
		self
	}
	fn re_only(self) -> Self {
		self
	}
	fn zero_of() -> Self {
		0.0
	}
	fn syevd(buf: &mut [Self], n: usize, w: &mut [f64]) -> i32 {
		unsafe { LAPACKE_dsyevd(COL_MAJOR, JOBZ_V, UPLO_L, n as i32, buf.as_mut_ptr(), n as i32, w.as_mut_ptr()) }
	}
}
impl Scalar for c64 {
	fn parts(self) -> (f64, f64) {
		(self.re, self.im)
	}
	fn sample(st: &mut u64) -> Self {
		c64::new(unif(st), unif(st))
	}
	fn herm(self) -> Self {
		self.conj()
	}
	fn re_only(self) -> Self {
		c64::new(self.re, 0.0)
	}
	fn zero_of() -> Self {
		c64::new(0.0, 0.0)
	}
	fn syevd(buf: &mut [Self], n: usize, w: &mut [f64]) -> i32 {
		unsafe { LAPACKE_zheevd(COL_MAJOR, JOBZ_V, UPLO_L, n as i32, buf.as_mut_ptr(), n as i32, w.as_mut_ptr()) }
	}
}

/// a Hermitian matrix, column-major lower triangle, from a fixed seed (the record's `0x391`)
fn hermitian<T: Scalar>(n: usize) -> Vec<T> {
	let mut st = 0x391u64;
	let mut buf = vec![T::zero_of(); n * n];
	for j in 0..n {
		for i in j..n {
			let mut x = T::sample(&mut st);
			if i == j {
				x = x.re_only();
			}
			buf[i + j * n] = x;
			buf[j + i * n] = x.herm();
		}
	}
	buf
}

// ------------------------------------------------------------------ timing

/// primes for `prime`, then times every call of a `window`-long window and returns the per-call
/// durations in ns. `after_prime` runs once, between the two, so counters reset after the priming.
fn window_of(
	prime: Duration,
	window: Duration,
	mut call: impl FnMut(),
	after_prime: impl FnOnce(),
) -> Vec<u64> {
	let t0 = Instant::now();
	while t0.elapsed() < prime {
		call();
	}
	after_prime();
	let mut times = Vec::with_capacity(1 << 22); // no push reallocates inside a window this long
	let t0 = Instant::now();
	while t0.elapsed() < window {
		let c = Instant::now();
		call();
		times.push(c.elapsed().as_nanos() as u64);
	}
	times
}

fn report(times: &[u64]) {
	let calls = times.len() as f64;
	let mean = times.iter().sum::<u64>() as f64 / calls;
	let mut t = times.to_vec();
	t.sort_unstable();
	println!("calls,,,{}", times.len());
	println!("call_mean_ns,,,{mean:.1}");
	println!("call_median_ns,,,{:.1}", t[t.len() / 2] as f64);
	println!("call_min_ns,,,{:.1}", t[0] as f64);
	println!("call_max_ns,,,{:.1}", t[t.len() - 1] as f64);
}

// ------------------------------------------------------------------ fork

fn fork_call<T: Scalar>(
	n: usize,
	threads: usize,
	thr: Option<usize>,
	prime: Duration,
	window: Duration,
) {
	let a_buf = hermitian::<T>(n);
	let a = MatRef::from_column_major_slice(&a_buf, n, n);
	let shape = if threads == 1 { ParShape::Seq } else { ParShape::team(threads) };
	let mut params: Spec<SelfAdjointEvdParams, T> = Default::default();
	if let Some(t) = thr {
		params.recursion_threshold = t;
	}
	println!("thr,,,{}", params.recursion_threshold);
	let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<T>(
		n,
		ComputeEigenvectors::Yes,
		shape,
		params,
	));
	let mut s = Diag::<T>::zeros(n);
	let mut u = Mat::<T>::zeros(n, n);
	let mut run_once = |mut par: Par| {
		let mut call = |mut par: Par| {
			self_adjoint_evd(
				a,
				s.as_mut(),
				Some(u.as_mut()),
				par.reborrow(),
				MemStack::new(&mut mem),
				params,
			)
			.unwrap();
		};
		window_of(prime, window, || call(par.reborrow()), || {
			#[cfg(feature = "instr")]
			faer::dc_probe::reset();
			#[cfg(feature = "count")]
			faer::dc_probe::reset_counts();
		})
	};
	let times = if threads == 1 {
		run_once(Par::Seq)
	} else {
		faer::team::with_team(threads, |par| run_once(par))
	};
	report(&times);
	// accuracy of the last call, outside the timed window
	let eps = f64::EPSILON;
	let tnorm = a.norm_max();
	let lam = Mat::<T>::from_fn(n, n, |i, j| if i == j { s[i] } else { T::zero_of() });
	let res = (a * u.as_ref() - u.as_ref() * lam.as_ref()).norm_max() / (n as f64 * eps * tnorm);
	let orth = (u.as_ref().adjoint() * u.as_ref() - Mat::<T>::identity(n, n)).norm_max()
		/ (n as f64 * eps);
	println!("residual_neps,,,{res:.4}");
	println!("orth_neps,,,{orth:.4}");
	let mut h: u64 = 0xcbf29ce484222325;
	let mut eat = |x: f64| {
		for b in x.to_bits().to_le_bytes() {
			h ^= b as u64;
			h = h.wrapping_mul(0x100000001b3);
		}
	};
	for i in 0..n {
		let (re, im) = s[i].parts();
		eat(re);
		eat(im);
	}
	for j in 0..n {
		for i in 0..n {
			let (re, im) = u[(i, j)].parts();
			eat(re);
			eat(im);
		}
	}
	println!("digest,,,{h:016x}");
	#[cfg(feature = "count")]
	{
		use faer::dc_probe as P;
		for l in 0..P::MAX_LEVEL {
			for (key, arr) in [
				("sec_roots", &P::SEC_ROOTS),
				("sec_batch", &P::SEC_BATCH),
				("sec_terms", &P::SEC_TERMS),
				("sec_evals", &P::SEC_EVALS),
				("sec_bisect_roots", &P::SEC_BISECT_ROOTS),
				("sec_bisect_evals", &P::SEC_BISECT_EVALS),
			] {
				let v = arr[l].load(std::sync::atomic::Ordering::Relaxed);
				if v != 0 {
					println!("cnt,{l},{key},{:.4}", v as f64 / times.len() as f64);
				}
			}
		}
	}
	#[cfg(feature = "instr")]
	{
		use faer::dc_probe as P;
		for l in 0..P::MAX_LEVEL {
			for p in 0..P::N_PHASES {
				let c = P::CALLS[l][p].load(std::sync::atomic::Ordering::Relaxed);
				if c == 0 {
					continue;
				}
				println!(
					"ns,{l},{},{:.1}",
					P::PHASES[p],
					P::NS[l][p].load(std::sync::atomic::Ordering::Relaxed) as f64
						/ times.len() as f64
				);
				println!(
					"nodes,{l},{},{:.4}",
					P::PHASES[p],
					c as f64 / times.len() as f64
				);
			}
			// joins and non-deflated sizes: what a level's thread-time has to be divided by
			for (key, arr) in [
				("halves_par", &P::HALVES_PAR),
				("halves_seq", &P::HALVES_SEQ),
				("gemm_par", &P::GEMM_PAR),
				("gemm_seq", &P::GEMM_SEQ),
				("k_sum", &P::K_SUM),
			] {
				let v = arr[l].load(std::sync::atomic::Ordering::Relaxed);
				if v != 0 {
					println!("stat,{l},{key},{:.4}", v as f64 / times.len() as f64);
				}
			}
		}
	}
}

// ------------------------------------------------------------------ MKL

fn mkl_call<T: Scalar>(n: usize, prime: Duration, window: Duration) {
	let a0 = hermitian::<T>(n);
	let mut buf = a0.clone();
	let mut w = vec![0.0f64; n];
	let info = T::syevd(&mut buf, n, &mut w);
	assert_eq!(info, 0, "MKL ?syevd returned {info}");
	// the input is overwritten, so it is restored outside the clock
	let times = window_of(
		prime,
		window,
		|| {
			buf.copy_from_slice(&a0);
			let info = T::syevd(&mut buf, n, &mut w);
			assert_eq!(info, 0);
		},
		|| {},
	);
	report(&times);
}

/// MKL's own stages on the tridiagonal the fork's stage 1 hands its stage 2 (f64).
fn mkl_stage(n: usize, prime: Duration, window: Duration) {
	let a0 = hermitian::<f64>(n);
	let mut buf = a0.clone();
	let mut d = vec![0.0f64; n];
	let mut e = vec![0.0f64; n - 1];
	let mut tau = vec![0.0f64; n - 1];
	let mut z = vec![0.0f64; n * n];
	unsafe {
		assert_eq!(
			LAPACKE_dsytrd(COL_MAJOR, UPLO_L, n as i32, buf.as_mut_ptr(), n as i32, d.as_mut_ptr(), e.as_mut_ptr(), tau.as_mut_ptr()),
			0
		);
	}
	// ?stedc consumes (d, e); keep the tridiagonal as ?sytrd left it
	let (d0, e0) = (d.clone(), e.clone());
	let t_trd = window_of(
		prime,
		window,
		|| {
			buf.copy_from_slice(&a0);
			unsafe {
				assert_eq!(
					LAPACKE_dsytrd(COL_MAJOR, UPLO_L, n as i32, buf.as_mut_ptr(), n as i32, d.as_mut_ptr(), e.as_mut_ptr(), tau.as_mut_ptr()),
					0
				);
			}
		},
		|| {},
	);
	print_means("sytrd", &t_trd);
	// ?stedc with eigenvectors: compz='I' starts from the identity
	let t_stedc = window_of(
		prime,
		window,
		|| {
			d.copy_from_slice(&d0);
			e.copy_from_slice(&e0);
			unsafe {
				assert_eq!(LAPACKE_dstedc(COL_MAJOR, COMPZ_I, n as i32, d.as_mut_ptr(), e.as_mut_ptr(), z.as_mut_ptr(), n as i32), 0);
			}
		},
		|| {},
	);
	print_means("stedc", &t_stedc);
}

fn print_means(key: &str, times: &[u64]) {
	let mean = times.iter().sum::<u64>() as f64 / times.len() as f64;
	let mut t = times.to_vec();
	t.sort_unstable();
	println!("stage_mean_ns,,{key},{mean:.1}");
	println!("stage_median_ns,,{key},{:.1}", t[t.len() / 2] as f64);
}

fn main() {
	let a: Vec<String> = std::env::args().collect();
	let lib = a[1].as_str();
	let mode = a[2].as_str();
	let dtype = a[3].as_str();
	let n: usize = a[4].parse().unwrap();
	let threads: usize = a[5].parse().unwrap();
	// `default` (and the vendor arm's ignored `-`) leave the crate's own threshold
	let thr: Option<usize> = a[6].parse().ok();
	unsafe {
		assert_eq!(log(1.0), 0.0);
	}
	let prime = Duration::from_millis(a.get(7).map_or(250, |x| x.parse().unwrap()));
	let window = Duration::from_millis(a.get(8).map_or(1000, |x| x.parse().unwrap()));
	// the vendor budget is the process's OpenMP budget; the runner sets `OMP_NUM_THREADS` too
	unsafe {
		println!("mkl_max_threads,,,{}", mkl_get_max_threads());
		println!("omp_max_threads,,,{}", omp_get_max_threads());
	}
	match (lib, mode, dtype) {
		("fork", "call", "f64") => fork_call::<f64>(n, threads, thr, prime, window),
		("fork", "call", "c64") => fork_call::<c64>(n, threads, thr, prime, window),
		("mkl", "call", "f64") => mkl_call::<f64>(n, prime, window),
		("mkl", "call", "c64") => mkl_call::<c64>(n, prime, window),
		("mkl", "mstage", "f64") => mkl_stage(n, prime, window),
		_ => panic!("usage: u391 <fork|mkl> <call|mstage> <f64|c64> <n> <threads> <thr|default> [prime_ms] [window_ms]"),
	}
}
