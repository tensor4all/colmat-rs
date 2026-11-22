//! Which kernel does `colmat::linalg::matmul::matmul` reach on this host? (tensor4all/colmat-rs#180)
//!
//! Runs one product shape through the public `matmul` entry, at `Par::Seq` or on a caller-owned
//! team, and prints the per-call time as median and interquartile range, the number of `matmul`
//! calls it made (`calls`), and what the linked `t4a-gemm-common`'s `sme_thresholds` gate says for
//! the whole shape as one sequential call (`gate_seq`). It does not observe the route itself:
//! `scripts/sme-route-probe.sh` runs it under `lldb` and counts the hits on the `t4a-gemm` SME and
//! NEON `gemm_basic` symbols, which is what says whether the Apple SME kernel was reached.
//!
//! ```sh
//! cargo run --release -p colmat --example sme_route -- <f32|f64|c32|c64> <m> <n> <k> <threads> [samples]
//! ```
//!
//! Each sample is at least 50 ms of back-to-back calls after two untimed priming calls. The result
//! of the last call is compared with a direct sum at 64 entries, as a smoke check against gross
//! errors (a wrong tile, a missing `k` block), with the worst-case bound `4 k eps sum |a| |b|`. It
//! is not a test of the kernel's accuracy: that belongs to `t4a-gemm` and to colmat's own tests.

use colmat::prelude::*;
use colmat::traits::ComplexField;
use colmat::{Accum, Par};
use gemm_common::sme_thresholds::{self, SmeThresholds};
use std::time::Instant;

trait Elem: ComplexField + Copy + Send + Sync + 'static {
	const NAME: &'static str;
	const FLOPS_PER_MAC: f64;
	/// the `t4a-gemm` SME size gate of this element type
	const GATE: SmeThresholds;
	fn from_seed(i: usize) -> Self;
	fn dist(a: Self, b: Self) -> f64;
	fn mag(a: Self) -> f64;
	fn mul_add(acc: Self, a: Self, b: Self) -> Self;
	fn zero_() -> Self;
	fn eps() -> f64;
}

fn seed(i: usize) -> f64 {
	// a cheap deterministic value in [-1, 1)
	let x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
	(x as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

macro_rules! real {
	($t:ty, $name:literal, $gate:expr) => {
		impl Elem for $t {
			const FLOPS_PER_MAC: f64 = 2.0;
			const GATE: SmeThresholds = $gate;
			const NAME: &'static str = $name;

			fn from_seed(i: usize) -> Self {
				seed(i) as $t
			}

			fn dist(a: Self, b: Self) -> f64 {
				((a - b) as f64).abs()
			}

			fn mag(a: Self) -> f64 {
				(a as f64).abs()
			}

			fn mul_add(acc: Self, a: Self, b: Self) -> Self {
				acc + a * b
			}

			fn zero_() -> Self {
				0.0
			}

			fn eps() -> f64 {
				<$t>::EPSILON as f64
			}
		}
	};
}
macro_rules! cplx {
	($t:ty, $r:ty, $name:literal, $gate:expr) => {
		impl Elem for $t {
			const FLOPS_PER_MAC: f64 = 8.0;
			const GATE: SmeThresholds = $gate;
			const NAME: &'static str = $name;

			fn from_seed(i: usize) -> Self {
				<$t>::new(seed(2 * i) as $r, seed(2 * i + 1) as $r)
			}

			fn dist(a: Self, b: Self) -> f64 {
				(a - b).norm() as f64
			}

			fn mag(a: Self) -> f64 {
				a.norm() as f64
			}

			fn mul_add(acc: Self, a: Self, b: Self) -> Self {
				acc + a * b
			}

			fn zero_() -> Self {
				<$t>::new(0.0, 0.0)
			}

			fn eps() -> f64 {
				<$r>::EPSILON as f64
			}
		}
	};
}
real!(f32, "f32", sme_thresholds::F32);
real!(f64, "f64", sme_thresholds::F64);
cplx!(c32, f32, "c32", sme_thresholds::C32);
cplx!(c64, f64, "c64", sme_thresholds::C64);

fn run<T: Elem>(m: usize, n: usize, k: usize, threads: usize, samples: usize) {
	let a = Mat::<T>::from_fn(m, k, |i, j| T::from_seed(i + m * j));
	let b = Mat::<T>::from_fn(k, n, |i, j| T::from_seed(7 + i + k * j + m * k));
	let mut c = Mat::<T>::zeros(m, n);

	let call = |par: Par<'_>, c: &mut Mat<T>| {
		colmat::linalg::matmul::matmul(
			c.as_mut(),
			Accum::Replace,
			a.as_ref(),
			b.as_ref(),
			T::one_impl(),
			par,
		);
	};

	let mut times = Vec::with_capacity(samples);
	let mut measure = |mut par: Par<'_>| {
		call(par.reborrow(), &mut c);
		call(par.reborrow(), &mut c);
		let once = {
			let t = Instant::now();
			call(par.reborrow(), &mut c);
			t.elapsed().as_secs_f64()
		};
		let batch = ((0.05 / once.max(1e-9)).ceil() as usize).max(1);
		for _ in 0..samples {
			let t = Instant::now();
			for _ in 0..batch {
				call(par.reborrow(), &mut c);
			}
			times.push(t.elapsed().as_secs_f64() / batch as f64);
		}
		batch
	};

	let batch = if threads <= 1 {
		measure(Par::Seq)
	} else {
		let pool = threadteam::Pool::builder()
			.threads(threads)
			.acquire_timeout(core::time::Duration::from_secs(30))
			.build()
			.expect("a thread pool");
		let team = pool
			.into_team(threads)
			.unwrap_or_else(|_| panic!("an idle pool"));
		let mut exec = team.exec(threads);
		exec.team(|t| measure(Par::Team(t)))
			.unwrap_or_else(|_| panic!("a team token"))
	};

	// 64 entries against a direct sum: the corners, then a spread over the whole output
	let picks = (0..64usize).map(|t| match t {
		0 => (0, 0),
		1 => (m - 1, n - 1),
		2 => (m - 1, 0),
		3 => (0, n - 1),
		t => ((t * 7919) % m, (t * 104_729) % n),
	});
	for (i, j) in picks {
		let mut s = T::zero_();
		let mut scale = 0.0f64;
		for p in 0..k {
			s = T::mul_add(s, a[(i, p)], b[(p, j)]);
			scale += T::mag(a[(i, p)]) * T::mag(b[(p, j)]);
		}
		let err = T::dist(c[(i, j)], s);
		assert!(
			err <= 4.0 * (k as f64) * T::eps() * scale.max(1.0),
			"entry ({i}, {j}): error {err:e}"
		);
	}

	times.sort_by(|x, y| x.partial_cmp(y).unwrap());
	let q = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
	let (q1, med, q3) = (q(0.25), q(0.5), q(0.75));
	let gflops = T::FLOPS_PER_MAC * (m * n * k) as f64 / med / 1e9;
	println!(
		"dtype={} m={m} n={n} k={k} threads={threads} samples={samples} batch={batch} calls={} gate_seq={} median_ms={:.4} q1_ms={:.4} q3_ms={:.4} iqr_ms={:.4} gflops={gflops:.1}",
		T::NAME,
		3 + samples * batch,
		if T::GATE.wants_sme(m, n, k) {
			"sme"
		} else {
			"fallback"
		},
		med * 1e3,
		q1 * 1e3,
		q3 * 1e3,
		(q3 - q1) * 1e3,
	);
}

fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	if args.len() < 5 {
		eprintln!(
			"usage: sme_route <f32|f64|c32|c64> <m> <n> <k> <threads> [samples]"
		);
		std::process::exit(2);
	}
	let p = |i: usize| args[i].parse::<usize>().expect("a size");
	let (m, n, k, threads) = (p(1), p(2), p(3), p(4));
	let samples = args
		.get(5)
		.map(|s| s.parse().expect("a count"))
		.unwrap_or(11);
	if m == 0 || n == 0 || k == 0 || threads == 0 || samples == 0 {
		eprintln!("sme_route: m, n, k, threads and samples must be at least 1");
		std::process::exit(2);
	}
	match args[0].as_str() {
		"f32" => run::<f32>(m, n, k, threads, samples),
		"f64" => run::<f64>(m, n, k, threads, samples),
		"c32" => run::<c32>(m, n, k, threads, samples),
		"c64" => run::<c64>(m, n, k, threads, samples),
		other => panic!("unknown dtype {other}"),
	}
}
