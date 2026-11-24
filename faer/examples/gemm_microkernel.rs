//! Microkernel-level benchmark of colmat's x86 f64 GEMM kernels
//! (tensor4all/colmat-rs#253, E0).
//!
//! On x86_64, f64 `matmul` above the nano-gemm threshold runs the assembly
//! microkernels that `t4a-private-gemm-x86`'s `build.rs` generates. This probe
//! calls **one kernel** of those tables through the crate's public
//! `call_microkernel`, on A and B that were packed before the clock starts, so
//! what it times is the kernel alone: no packing, no blocking loops, no team,
//! no `matmul` dispatch. Kernels and their packed layouts are the ones the
//! linked crate generates; the version the binary was built against is read
//! from `Cargo.lock` and printed.
//!
//! ```sh
//! cargo build --release -p colmat --example gemm_microkernel
//! target/release/examples/gemm_microkernel --cpu 2 [--samples 31] [--sample-ms 10]
//!     [--prime-ms 200] [--filter <substring>] [--kc <depth>]
//! ```
//!
//! `scripts/gemm-microkernel-bench.sh` builds it, records the build and host
//! pins and checks that the measured core stayed idle; use that for a
//! measurement that is meant to be kept.
//!
//! # What one row is
//!
//! A *case* is (instruction set, kernel table, `mr` x `n` tile, rows,
//! residency, B layout). A *sample* is a batch of back-to-back kernel calls
//! lasting about `--sample-ms`; each case takes `--samples` samples after a
//! time-based priming spin of `--prime-ms`, and the row reports the median and
//! the interquartile range [Q1, Q3] of the per-sample FLOP/cycle and ns/call.
//! One call is `C[0..rows, 0..n] += A[0..rows, 0..kc] * B[0..kc, 0..n]`
//! (`Accum::Add`, alpha = 1, `2 rows n kc` flops), with C column-major.
//!
//! Tiles: every entry of `F64_SIMD512x4` (rows 48..16 by 8 with n = 1..4, rows
//! 8 with n = 1..8), the entries of `F64_SIMD512x8` that only that table
//! generates (rows 24 and 16 with n = 5..8), and every entry of `F64_SIMD256`
//! (rows 12 and 8 with n = 1..4, rows 4 with n = 1..8). The production tiles
//! are 48x4 (AVX-512, m > 48), 24x8 (AVX-512, m <= 48) and 12x4 (AVX2). Each of
//! them is also run with one row fewer (`rows = mr - 1`), which takes the
//! masked load/store path.
//!
//! Residency (sizes from the pinned CPU's
//! `/sys/devices/system/cpu/cpuN/cache`):
//!
//! - `l1`: one A panel, one B panel and one C tile, reused; `kc` is the largest
//!   multiple of 4 with `(mr + n) kc 8 <= L1d / 2`. The kernel's compute
//!   ceiling, prologue and epilogue included.
//! - `l2`: `kc = 512` (the depth block `gemm` uses), A rotates over as many
//!   packed panels as fit in half of L2 (at least two), B one panel, C one
//!   tile.
//! - `cstream`: as `l2`, but every call writes the next tile of a column-major
//!   C of at least 4 x L3 bytes (leading dimension `ldc`, default 4000),
//!   walking tiles down a column block and then to the next one. The hardware
//!   prefetcher can follow it. (Not verified to be the order the gemm driver
//!   visits C in; its 1T path can traverse row-major.)
//! - `ccold`: as `cstream`, but the tiles are visited with a stride coprime to
//!   their count, so C comes from memory and no stream prefetcher can predict
//!   it (the issue's E0(c)).
//!
//! B layout: `packed` is the kernel's packed-B layout (per depth step, `n`
//! contiguous values). `direct:<ldb>` passes B unpacked, column-major with
//! leading dimension `ldb`, which is what `gemm` hands the kernel whenever it
//! does not pack B (`pack_rhs = tall`); it is run for the production tiles at
//! `ldb` = 4000 and 4096 (the H2 aliasing question). A is always packed here.
//!
//! # Cycles
//!
//! FLOP/cycle needs core cycles, not TSC ticks. The probe opens a user-mode
//! `PERF_COUNT_HW_CPU_CYCLES` counter, read with its enabled and running
//! times: a sample during which the event was not on the PMU the whole time
//! (multiplexed with other events) or that counted zero cycles is rejected, the
//! row prints `-` for FLOP/cycle, its `check` column names the rejection and
//! the largest enabled/running multiplier, and the probe exits 4 at the end. A
//! rate is never extrapolated from a partial count. When the kernel refuses
//! (`perf_event_paranoid` > 2 without privileges) it falls back to
//! wall time times the core clock, which it measures before and after every
//! case with a chain of dependent register `add`s (one cycle each); the `clock`
//! header says which source was used, and a case whose two clock readings
//! differ by more than 1% is marked `unstable`.
//!
//! The FMA ceiling rows (`peak/zmm`, `peak/ymm`) time twelve independent
//! `vfmadd231pd` chains, so the percentage of peak can be read against the
//! measured ceiling of this core (`pct_fma`) as well as against the nominal one
//! (`pct_nom`: 32 FLOP/cycle for AVX-512 and 16 for AVX2, as the issue
//! states for Zen5 and Xeon; a core with a 256-bit datapath, mobile Zen5 among
//! them, measures 16 for both).
//!
//! # Correctness smoke
//!
//! Before timing, each case runs one call into a C of zeros and compares it
//! with a direct sum, bound `4 kc eps sum_k |a| |b|` per entry, and checks that
//! the rows past `rows` were not written (masked store). It guards against a
//! wrong packed layout or table index in this probe, and is no test of the
//! kernels' accuracy.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod imp {
	use private_gemm_x86::{
		F64_SIMD256, F64_SIMD512x4, F64_SIMD512x8, MicrokernelInfo, Position,
		call_microkernel,
	};
	use std::alloc::{Layout, alloc_zeroed, dealloc};
	use std::fs::File;
	use std::hint::black_box;
	use std::io::Read;
	use std::os::fd::FromRawFd;
	use std::ptr::null;
	use std::time::{Duration, Instant};

	/// `FLAGS_ACCUM` of `private_gemm_x86` (private there): accumulate into C.
	const FLAGS_ACCUM: usize = 1;
	/// the depth block `private_gemm_x86::gemm` uses (`kc = min(k, 512)`)
	const GEMM_KC: usize = 512;

	type Kernel = unsafe extern "C" fn();

	#[derive(Copy, Clone, PartialEq, Eq)]
	enum Isa {
		Avx512,
		Avx2,
	}

	impl Isa {
		fn name(self) -> &'static str {
			match self {
				Isa::Avx512 => "avx512",
				Isa::Avx2 => "avx2",
			}
		}

		fn available(self) -> bool {
			match self {
				Isa::Avx512 => is_x86_feature_detected!("avx512f"),
				Isa::Avx2 => {
					is_x86_feature_detected!("avx2")
						&& is_x86_feature_detected!("fma")
				},
			}
		}
	}

	#[derive(Copy, Clone)]
	struct Shape {
		isa: Isa,
		table: &'static str,
		kernel: Kernel,
		mr: usize,
		n: usize,
	}

	/// Every kernel of the f64 tables, in the order `build.rs` generates them,
	/// so a table index is the same kernel the gemm driver would pick. A table
	/// whose length does not match is a generator change this probe has not
	/// followed: stop rather than time the wrong kernel.
	fn shapes() -> Vec<Shape> {
		let mut v = Vec::new();

		let mut i = 0;
		for m in (1..=6usize).rev() {
			let last = if m == 1 { 8 } else { 4 };
			for n in 1..=last {
				v.push(Shape {
					isa: Isa::Avx512,
					table: "512x4",
					kernel: F64_SIMD512x4[i],
					mr: 8 * m,
					n,
				});
				i += 1;
			}
		}
		assert_eq!(i, F64_SIMD512x4.len(), "F64_SIMD512x4 layout changed");

		// the x8 table repeats the x4 kernels for m = 1 or n <= 4 (same
		// symbols); keep the rest
		let mut i = 0;
		for m in (1..=3usize).rev() {
			for n in 1..=8 {
				if m > 1 && n > 4 {
					v.push(Shape {
						isa: Isa::Avx512,
						table: "512x8",
						kernel: F64_SIMD512x8[i],
						mr: 8 * m,
						n,
					});
				}
				i += 1;
			}
		}
		assert_eq!(i, F64_SIMD512x8.len(), "F64_SIMD512x8 layout changed");

		let mut i = 0;
		for m in (1..=3usize).rev() {
			let last = if m == 1 { 8 } else { 4 };
			for n in 1..=last {
				v.push(Shape {
					isa: Isa::Avx2,
					table: "256",
					kernel: F64_SIMD256[i],
					mr: 4 * m,
					n,
				});
				i += 1;
			}
		}
		assert_eq!(i, F64_SIMD256.len(), "F64_SIMD256 layout changed");
		v
	}

	#[derive(Copy, Clone, PartialEq, Eq)]
	enum Res {
		L1,
		L2,
		CStream,
		CCold,
	}

	impl Res {
		fn name(self) -> &'static str {
			match self {
				Res::L1 => "l1",
				Res::L2 => "l2",
				Res::CStream => "cstream",
				Res::CCold => "ccold",
			}
		}
	}

	#[derive(Copy, Clone)]
	struct Case {
		shape: Shape,
		rows: usize,
		res: Res,
		/// `None`: packed B; `Some(ldb)`: column-major B with that leading
		/// dimension
		ldb: Option<usize>,
	}

	impl Case {
		fn label(&self) -> String {
			let b = match self.ldb {
				None => "packed".to_string(),
				Some(ldb) => format!("direct:{ldb}"),
			};
			format!(
				"{}/{}/{}x{}/rows={}/{}/{}",
				self.shape.isa.name(),
				self.shape.table,
				self.shape.mr,
				self.shape.n,
				self.rows,
				self.res.name(),
				b
			)
		}
	}

	fn suite() -> Vec<Case> {
		let all = shapes();
		let mut v = Vec::new();
		for &shape in &all {
			for res in [Res::L1, Res::L2, Res::CStream, Res::CCold] {
				v.push(Case {
					shape,
					rows: shape.mr,
					res,
					ldb: None,
				});
			}
		}
		let production = |table: &str, mr: usize, n: usize| {
			*all.iter()
				.find(|s| s.table == table && s.mr == mr && s.n == n)
				.unwrap()
		};
		let prod = [
			production("512x4", 48, 4),
			production("512x8", 24, 8),
			production("256", 12, 4),
		];
		for &shape in &prod {
			v.push(Case {
				shape,
				rows: shape.mr - 1,
				res: Res::L2,
				ldb: None,
			});
		}
		for &shape in &prod {
			for res in [Res::L2, Res::CStream] {
				for ldb in [4000, 4096] {
					v.push(Case {
						shape,
						rows: shape.mr,
						res,
						ldb: Some(ldb),
					});
				}
			}
		}
		v
	}

	/// A zeroed, page-aligned buffer of f64, with a tail the kernels may read
	/// past the end of a packed panel.
	struct Buf {
		ptr: *mut f64,
		layout: Layout,
		len: usize,
	}

	impl Buf {
		fn new(len: usize) -> Self {
			let bytes = (len * 8 + 4096).next_multiple_of(4096);
			let layout = Layout::from_size_align(bytes, 4096).unwrap();
			// SAFETY: nonzero size
			let ptr = unsafe { alloc_zeroed(layout) } as *mut f64;
			assert!(!ptr.is_null(), "allocation of {bytes} bytes failed");
			Buf { ptr, layout, len }
		}

		fn slice(&mut self) -> &mut [f64] {
			// SAFETY: `len` zero-initialized f64 owned by `self`
			unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
		}
	}

	impl Drop for Buf {
		fn drop(&mut self) {
			// SAFETY: allocated with this layout in `new`
			unsafe { dealloc(self.ptr as *mut u8, self.layout) }
		}
	}

	fn seed(i: usize) -> f64 {
		// a cheap deterministic value in [-1, 1)
		let x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
		(x as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
	}

	struct Caches {
		l1d: usize,
		l2: usize,
		l3: usize,
	}

	fn sysfs(path: &str) -> Option<String> {
		std::fs::read_to_string(path)
			.ok()
			.map(|s| s.trim().to_string())
	}

	fn cache_bytes(cpu: usize, level: u32) -> Option<usize> {
		for index in 0..8 {
			let dir =
				format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{index}");
			let Some(lv) = sysfs(&format!("{dir}/level")) else {
				break;
			};
			let ty = sysfs(&format!("{dir}/type")).unwrap_or_default();
			if lv == level.to_string() && ty != "Instruction" {
				let size = sysfs(&format!("{dir}/size"))?;
				let (num, mul) = match size.strip_suffix('K') {
					Some(n) => (n, 1024),
					None => match size.strip_suffix('M') {
						Some(n) => (n, 1024 * 1024),
						None => (size.as_str(), 1),
					},
				};
				return num.parse::<usize>().ok().map(|n| n * mul);
			}
		}
		None
	}

	// --- clocks
	// ----------------------------------------------------------------------------

	/// Core cycles by a user-mode hardware counter, when the kernel allows one.
	fn perf_cycles_counter() -> Result<File, String> {
		// struct perf_event_attr, PERF_ATTR_SIZE_VER7 (128 bytes); only the
		// leading fields are set
		let mut attr = [0u64; 16];
		attr[0] = 128 << 32; // type = PERF_TYPE_HARDWARE (0), size = 128
		attr[1] = 0; // config = PERF_COUNT_HW_CPU_CYCLES
		// read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | ..._RUNNING: a read
		// returns { value, time_enabled, time_running }
		attr[4] = 1 | 2;
		attr[5] = (1 << 5) | (1 << 6); // exclude_kernel | exclude_hv
		let (pid, cpu, group, flags): (
			libc::pid_t,
			libc::c_int,
			libc::c_int,
			libc::c_ulong,
		) = (0, -1, -1, 0);
		// SAFETY: perf_event_open(attr, pid = 0 (self), cpu = -1 (any),
		// group_fd = -1, flags = 0)
		let fd = unsafe {
			libc::syscall(
				libc::SYS_perf_event_open,
				attr.as_ptr(),
				pid,
				cpu,
				group,
				flags,
			)
		};
		if fd < 0 {
			return Err(std::io::Error::last_os_error().to_string());
		}
		// SAFETY: a fresh fd owned by nobody else
		Ok(unsafe { File::from_raw_fd(fd as i32) })
	}

	/// One read of the counter: `{ value, time_enabled, time_running }`
	/// (`read_format` = TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING).
	#[derive(Clone, Copy, Debug, PartialEq)]
	struct PerfSample {
		count: u64,
		enabled_ns: u64,
		running_ns: u64,
	}

	impl PerfSample {
		fn from_bytes(b: &[u8; 24]) -> Self {
			let w = |i: usize| {
				u64::from_ne_bytes(b[8 * i..8 * i + 8].try_into().unwrap())
			};
			PerfSample {
				count: w(0),
				enabled_ns: w(1),
				running_ns: w(2),
			}
		}
	}

	fn read_counter(f: &mut File) -> PerfSample {
		let mut b = [0u8; 24];
		f.read_exact(&mut b).expect("read perf counter");
		PerfSample::from_bytes(&b)
	}

	/// Why a counter interval gives no cycle count.
	#[derive(Clone, Copy, Debug, PartialEq)]
	enum CounterReject {
		/// the event was on the PMU for only part of the interval (or never):
		/// `multiplier` = enabled / running, infinite when it never ran
		Multiplexed { multiplier: f64 },
		/// the event ran the whole interval but counted nothing
		ZeroCount,
		/// a field went backwards (not a pair of reads of one event)
		NotMonotonic,
	}

	/// Core cycles between two reads, only when the event counted the whole
	/// interval. A multiplexed event's raw delta covers only the periods it was
	/// scheduled, while the caller's FLOP numerator covers the whole interval,
	/// so a partial delta would inflate FLOP/cycle by enabled/running; it is
	/// rejected rather than scaled.
	fn cycles_between(
		s0: PerfSample,
		s1: PerfSample,
	) -> Result<u64, CounterReject> {
		let (Some(count), Some(enabled), Some(running)) = (
			s1.count.checked_sub(s0.count),
			s1.enabled_ns.checked_sub(s0.enabled_ns),
			s1.running_ns.checked_sub(s0.running_ns),
		) else {
			return Err(CounterReject::NotMonotonic);
		};
		if running > enabled {
			return Err(CounterReject::NotMonotonic);
		}
		if enabled == 0 || running < enabled {
			let multiplier = if running == 0 {
				f64::INFINITY
			} else {
				enabled as f64 / running as f64
			};
			return Err(CounterReject::Multiplexed { multiplier });
		}
		if count == 0 {
			return Err(CounterReject::ZeroCount);
		}
		Ok(count)
	}

	/// Run `iters * 64` dependent register adds: `iters * 64` core cycles at
	/// one cycle each.
	#[inline(never)]
	fn add_chain(iters: u64) -> u64 {
		let mut x: u64 = 1;
		let y: u64 = black_box(3);
		let mut n = iters;
		// SAFETY: register-only arithmetic
		unsafe {
			core::arch::asm!(
				"2:",
				".rept 64",
				"add {x}, {y}",
				".endr",
				"dec {n}",
				"jnz 2b",
				x = inout(reg) x,
				y = in(reg) y,
				n = inout(reg) n,
				options(nomem, nostack),
			);
		}
		black_box(n);
		x
	}

	/// Core clock in cycles per ns, from the dependent-add chain (about 5 ms).
	fn core_ghz() -> f64 {
		let iters = 150_000u64;
		add_chain(iters / 10);
		let t = Instant::now();
		black_box(add_chain(iters));
		let ns = t.elapsed().as_nanos() as f64;
		(iters * 64) as f64 / ns
	}

	enum Clock {
		Perf(File),
		Calibrated,
	}

	impl Clock {
		fn name(&self) -> &'static str {
			match self {
				Clock::Perf(_) => "perf:cpu-cycles(user)",
				Clock::Calibrated => "wall*add-chain-clock",
			}
		}
	}

	/// Core cycles of one timed interval.
	enum Cycles {
		Counted(f64),
		Rejected(CounterReject),
		/// calibrated clock: the caller derives them from the clock readings
		FromWall,
	}

	/// Time `f` and return (core cycles, ns).
	fn timed(clock: &mut Clock, f: impl FnOnce()) -> (Cycles, f64) {
		match clock {
			Clock::Perf(file) => {
				let s0 = read_counter(file);
				let t = Instant::now();
				f();
				let ns = t.elapsed().as_nanos() as f64;
				let s1 = read_counter(file);
				let cy = match cycles_between(s0, s1) {
					Ok(c) => Cycles::Counted(c as f64),
					Err(r) => Cycles::Rejected(r),
				};
				(cy, ns)
			},
			Clock::Calibrated => {
				let t = Instant::now();
				f();
				(Cycles::FromWall, t.elapsed().as_nanos() as f64)
			},
		}
	}

	fn spin(d: Duration) {
		let t = Instant::now();
		while t.elapsed() < d {
			black_box(());
		}
	}

	// --- FMA ceiling
	// ------------------------------------------------------------------------

	#[inline(never)]
	fn fma_zmm(iters: u64) {
		let mut n = iters;
		// SAFETY: register-only arithmetic on zeroed registers; runs only when
		// avx512f is detected, and every instruction here is AVX-512F: the
		// zeroing is `vpxord` (EVEX.512 `vxorpd` would need AVX-512DQ)
		unsafe {
			core::arch::asm!(
				"vpxord zmm12, zmm12, zmm12",
				"vpxord zmm13, zmm13, zmm13",
				"2:",
				"vfmadd231pd zmm0, zmm12, zmm13",
				"vfmadd231pd zmm1, zmm12, zmm13",
				"vfmadd231pd zmm2, zmm12, zmm13",
				"vfmadd231pd zmm3, zmm12, zmm13",
				"vfmadd231pd zmm4, zmm12, zmm13",
				"vfmadd231pd zmm5, zmm12, zmm13",
				"vfmadd231pd zmm6, zmm12, zmm13",
				"vfmadd231pd zmm7, zmm12, zmm13",
				"vfmadd231pd zmm8, zmm12, zmm13",
				"vfmadd231pd zmm9, zmm12, zmm13",
				"vfmadd231pd zmm10, zmm12, zmm13",
				"vfmadd231pd zmm11, zmm12, zmm13",
				"dec {n}",
				"jnz 2b",
				"vzeroupper",
				n = inout(reg) n,
				out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
				out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
				out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
				out("zmm12") _, out("zmm13") _,
				options(nomem, nostack),
			);
		}
		black_box(n);
	}

	#[inline(never)]
	fn fma_ymm(iters: u64) {
		let mut n = iters;
		// SAFETY: register-only arithmetic on zeroed registers; runs only when
		// avx2+fma is detected
		unsafe {
			core::arch::asm!(
				"vxorpd ymm12, ymm12, ymm12",
				"vxorpd ymm13, ymm13, ymm13",
				"2:",
				"vfmadd231pd ymm0, ymm12, ymm13",
				"vfmadd231pd ymm1, ymm12, ymm13",
				"vfmadd231pd ymm2, ymm12, ymm13",
				"vfmadd231pd ymm3, ymm12, ymm13",
				"vfmadd231pd ymm4, ymm12, ymm13",
				"vfmadd231pd ymm5, ymm12, ymm13",
				"vfmadd231pd ymm6, ymm12, ymm13",
				"vfmadd231pd ymm7, ymm12, ymm13",
				"vfmadd231pd ymm8, ymm12, ymm13",
				"vfmadd231pd ymm9, ymm12, ymm13",
				"vfmadd231pd ymm10, ymm12, ymm13",
				"vfmadd231pd ymm11, ymm12, ymm13",
				"dec {n}",
				"jnz 2b",
				"vzeroupper",
				n = inout(reg) n,
				out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
				out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
				out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
				out("ymm12") _, out("ymm13") _,
				options(nomem, nostack),
			);
		}
		black_box(n);
	}

	// --- statistics
	// -------------------------------------------------------------------------

	/// (Q1, median, Q3), linear interpolation between order statistics
	fn quartiles(xs: &[f64]) -> (f64, f64, f64) {
		let mut v = xs.to_vec();
		v.sort_by(|a, b| a.partial_cmp(b).unwrap());
		let q = |p: f64| {
			let h = p * (v.len() - 1) as f64;
			let lo = h.floor() as usize;
			let hi = h.ceil() as usize;
			v[lo] + (h - lo as f64) * (v[hi] - v[lo])
		};
		(q(0.25), q(0.5), q(0.75))
	}

	struct Opts {
		cpu: usize,
		samples: usize,
		sample_ms: u64,
		prime_ms: u64,
		filter: Option<String>,
		kc: Option<usize>,
		ldc: usize,
		peak512: f64,
		peak256: f64,
	}

	const USAGE: &str = "usage: gemm_microkernel --cpu N [--samples S] \
	                     [--sample-ms MS] [--prime-ms MS] [--filter SUBSTR] \
	                     [--kc K] [--ldc LDC] [--peak512 F] [--peak256 F]";

	/// Parse the probe's arguments. Every flag may be given once: a repeated
	/// `--cpu` would make the probe pin a core other than the one a wrapper
	/// asked for (and watches the SMT siblings of), so it is refused.
	fn parse_args(args: &[String]) -> Result<Opts, String> {
		let mut o = Opts {
			cpu: usize::MAX,
			samples: 31,
			sample_ms: 10,
			prime_ms: 200,
			filter: None,
			kc: None,
			ldc: 4000,
			peak512: 32.0,
			peak256: 16.0,
		};
		let mut seen: Vec<&str> = Vec::new();
		let mut i = 0;
		while i < args.len() {
			let flag = args[i].as_str();
			let val = args.get(i + 1).ok_or_else(|| USAGE.to_string())?;
			let num = || -> Result<usize, String> {
				val.parse()
					.map_err(|_| format!("{flag}: not a number: {val}"))
			};
			let flt = || -> Result<f64, String> {
				val.parse()
					.map_err(|_| format!("{flag}: not a number: {val}"))
			};
			if seen.contains(&flag) {
				return Err(format!("{flag} given twice\n{USAGE}"));
			}
			match flag {
				"--cpu" => o.cpu = num()?,
				"--samples" => o.samples = num()?,
				"--sample-ms" => o.sample_ms = num()? as u64,
				"--prime-ms" => o.prime_ms = num()? as u64,
				"--filter" => o.filter = Some(val.clone()),
				"--kc" => o.kc = Some(num()?),
				"--ldc" => o.ldc = num()?,
				"--peak512" => o.peak512 = flt()?,
				"--peak256" => o.peak256 = flt()?,
				_ => return Err(USAGE.to_string()),
			}
			seen.push(flag);
			i += 2;
		}
		if o.cpu == usize::MAX {
			return Err(format!(
				"--cpu is required: the probe pins itself to one core\n{USAGE}"
			));
		}
		if o.samples < 4 || o.kc == Some(0) {
			return Err("--samples must be >= 4 and --kc >= 1".into());
		}
		Ok(o)
	}

	fn parse() -> Opts {
		let args: Vec<String> = std::env::args().skip(1).collect();
		parse_args(&args).unwrap_or_else(|e| {
			eprintln!("{e}");
			std::process::exit(2)
		})
	}

	fn pin(cpu: usize) {
		// SAFETY: plain libc calls on a zeroed cpu_set_t
		unsafe {
			let mut set: libc::cpu_set_t = std::mem::zeroed();
			libc::CPU_SET(cpu, &mut set);
			if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set)
				!= 0
			{
				eprintln!(
					"sched_setaffinity({cpu}): {}",
					std::io::Error::last_os_error()
				);
				std::process::exit(1);
			}
			let now = libc::sched_getcpu();
			if now != cpu as i32 {
				eprintln!("pinned to {cpu} but running on {now}");
				std::process::exit(1);
			}
		}
	}

	/// The `t4a-private-gemm-x86` entry of the `Cargo.lock` this binary was
	/// built from.
	fn private_gemm_pin() -> String {
		let lock =
			include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../Cargo.lock"));
		let mut out = Vec::new();
		let mut it = lock.split("\n\n");
		for block in &mut it {
			if block.contains("name = \"t4a-private-gemm-x86\"") {
				let version = block
					.lines()
					.find_map(|l| l.strip_prefix("version = "))
					.unwrap_or("?")
					.trim_matches('"');
				let source = block
					.lines()
					.find_map(|l| l.strip_prefix("source = "))
					.map(|s| s.trim_matches('"').to_string())
					.unwrap_or_else(|| "path (patched)".to_string());
				let checksum = block
					.lines()
					.find_map(|l| l.strip_prefix("checksum = "))
					.unwrap_or("-")
					.trim_matches('"');
				out.push(format!("{version} {source} {checksum}"));
			}
		}
		if out.is_empty() {
			"not in Cargo.lock".to_string()
		} else {
			out.join("; ")
		}
	}

	struct Prepared {
		a: Buf,
		a_panels: usize,
		b: Buf,
		c: Buf,
		/// C tile origins (element offsets), in visiting order
		tiles: Vec<usize>,
		ldc: usize,
		kc: usize,
	}

	fn prepare(case: &Case, caches: &Caches, opts: &Opts) -> Prepared {
		let Shape { mr, n, .. } = case.shape;
		let kc = opts.kc.unwrap_or(match case.res {
			Res::L1 => ((caches.l1d / 2) / ((mr + n) * 8)) / 4 * 4,
			_ => GEMM_KC,
		});
		let kc = kc.max(4);
		let panel = mr * kc;
		let a_panels = match case.res {
			Res::L1 => 1,
			_ => ((caches.l2 / 2) / (panel * 8)).max(2),
		};
		let mut a = Buf::new(a_panels * panel);
		for (i, x) in a.slice().iter_mut().enumerate() {
			*x = seed(i);
		}
		let b_len = match case.ldb {
			None => kc * n,
			Some(ldb) => {
				assert!(ldb >= kc, "ldb {ldb} < kc {kc}");
				ldb * n
			},
		};
		let mut b = Buf::new(b_len);
		for (i, x) in b.slice().iter_mut().enumerate() {
			*x = seed(i + 1_000_003);
		}
		let (c, tiles, ldc) = match case.res {
			Res::L1 | Res::L2 => (Buf::new(mr * n), vec![0], mr),
			Res::CStream | Res::CCold => {
				let ldc = opts.ldc.max(mr);
				let cols =
					((4 * caches.l3).div_ceil(ldc * 8)).next_multiple_of(n);
				let per_col = ldc / mr;
				let blocks = cols / n;
				let count = per_col * blocks;
				let mut tiles = Vec::with_capacity(count);
				for cb in 0..blocks {
					for r in 0..per_col {
						tiles.push(r * mr + cb * n * ldc);
					}
				}
				if case.res == Res::CCold {
					// a stride near count / golden ratio, made coprime to count
					let mut s = (count as f64 * 0.618) as usize | 1;
					while gcd(s, count) != 1 {
						s += 2;
					}
					let seq = tiles.clone();
					for (j, t) in tiles.iter_mut().enumerate() {
						*t = seq[(j * s) % count];
					}
				}
				(Buf::new(ldc * cols), tiles, ldc)
			},
		};
		Prepared {
			a,
			a_panels,
			b,
			c,
			tiles,
			ldc,
			kc,
		}
	}

	fn gcd(a: usize, b: usize) -> usize {
		if b == 0 { a } else { gcd(b, a % b) }
	}

	fn info(
		case: &Case,
		kc: usize,
		c: *mut f64,
		ldc: usize,
		alpha: &f64,
	) -> MicrokernelInfo {
		let (rhs_rs, rhs_cs) = match case.ldb {
			// ignored by the kernel when it is handed rhs = null (packed B)
			None => (0, 0),
			Some(ldb) => (8, (ldb * 8) as isize),
		};
		MicrokernelInfo {
			flags: FLAGS_ACCUM,
			depth: kc,
			// ignored for packed A (lhs = null): the kernel uses (8, mr * 8)
			lhs_rs: 8,
			lhs_cs: (case.shape.mr * 8) as isize,
			rhs_rs,
			rhs_cs,
			alpha: alpha as *const f64 as *const (),
			ptr: c as *mut (),
			rs: 8,
			cs: (ldc * 8) as isize,
			row_idx: null(),
			col_idx: null(),
			diag_ptr: null(),
			diag_stride: 0,
		}
	}

	/// One kernel call: C(tile) += A(panel) B, with A packed and B packed or
	/// direct.
	///
	/// # Safety
	/// `a` holds a packed `mr x kc` panel, `b` the B of `case`, `info.ptr` a C
	/// tile with `info.cs`.
	#[inline(always)]
	unsafe fn call(
		case: &Case,
		a: *const f64,
		b: *const f64,
		info: &MicrokernelInfo,
	) {
		let rhs = match case.ldb {
			None => null(),
			Some(_) => b as *const (),
		};
		let mut pos = Position { row: 0, col: 0 };
		unsafe {
			call_microkernel(
				case.shape.kernel,
				null(),
				a as *mut (),
				rhs,
				b as *mut (),
				case.rows,
				case.shape.n,
				info,
				&mut pos,
			);
		}
	}

	/// One call into a C of zeros (sentinel 7 in the rows past `rows`), checked
	/// against a direct sum.
	fn check(case: &Case, p: &mut Prepared) -> Result<(), String> {
		let Shape { mr, n, .. } = case.shape;
		let kc = p.kc;
		let mut c = Buf::new(mr * n);
		for j in 0..n {
			for i in case.rows..mr {
				c.slice()[i + j * mr] = 7.0;
			}
		}
		let alpha = 1.0f64;
		let inf = info(case, kc, c.ptr, mr, &alpha);
		// SAFETY: buffers sized by `prepare`
		unsafe { call(case, p.a.ptr, p.b.ptr, &inf) };
		let a = &p.a.slice()[..mr * kc];
		let b = p.b.slice();
		let bv = |k: usize, j: usize| match case.ldb {
			None => b[k * n + j],
			Some(ldb) => b[k + j * ldb],
		};
		let c = c.slice();
		for j in 0..n {
			for i in 0..mr {
				let got = c[i + j * mr];
				if i >= case.rows {
					if got != 7.0 {
						return Err(format!(
							"masked row {i} col {j} written: {got}"
						));
					}
					continue;
				}
				let mut want = 0.0;
				let mut mag = 0.0;
				for k in 0..kc {
					want += a[i + k * mr] * bv(k, j);
					mag += (a[i + k * mr] * bv(k, j)).abs();
				}
				let tol = 4.0 * kc as f64 * f64::EPSILON * mag;
				if (got - want).abs() > tol || got.is_nan() {
					return Err(format!(
						"C[{i},{j}] = {got}, direct sum {want}, bound {tol:e}"
					));
				}
			}
		}
		Ok(())
	}

	/// `perf_reject:<rejected>/<samples>:<reasons>` for a row with rejected
	/// counter samples, where the reasons are `x<largest enabled/running>` for
	/// multiplexed samples, `zero` and `nonmonotonic`; `None` when all counted.
	fn reject_summary(
		rejects: &[CounterReject],
		samples: usize,
	) -> Option<String> {
		if rejects.is_empty() {
			return None;
		}
		let mut worst: Option<f64> = None;
		let (mut zero, mut nonmono) = (false, false);
		for r in rejects {
			match *r {
				CounterReject::Multiplexed { multiplier } => {
					worst =
						Some(worst.map_or(multiplier, |w| w.max(multiplier)))
				},
				CounterReject::ZeroCount => zero = true,
				CounterReject::NotMonotonic => nonmono = true,
			}
		}
		let mut s = format!("perf_reject:{}/{samples}", rejects.len());
		if let Some(w) = worst {
			s += &format!(":x{w:.3}");
		}
		if zero {
			s += ":zero";
		}
		if nonmono {
			s += ":nonmonotonic";
		}
		Some(s)
	}

	/// a FLOP/cycle figure, `-` when there is none
	fn fmt_rate(x: Option<f64>, digits: usize) -> String {
		match x {
			Some(x) if x.is_finite() => format!("{x:.digits$}"),
			_ => "-".into(),
		}
	}

	struct Row {
		/// `None` when any sample's cycle count was rejected
		fpc: Option<(f64, f64, f64)>,
		/// the `check` column suffix naming the rejected samples
		perf_reject: Option<String>,
		ns: (f64, f64, f64),
		ghz: (f64, f64),
		reps: usize,
	}

	fn measure(
		clock: &mut Clock,
		opts: &Opts,
		flops_per_call: f64,
		mut batch: impl FnMut(usize),
	) -> Row {
		spin(Duration::from_millis(opts.prime_ms));
		// size the batch to about `sample_ms`
		batch(1);
		let mut reps = 1usize;
		loop {
			let t = Instant::now();
			batch(reps);
			let ms = t.elapsed().as_secs_f64() * 1e3;
			if ms >= opts.sample_ms as f64 / 4.0 || reps >= 1 << 30 {
				reps = ((reps as f64) * opts.sample_ms as f64 / ms.max(1e-3))
					.ceil()
					.max(1.0) as usize;
				break;
			}
			reps *= 4;
		}
		let ghz0 = core_ghz();
		let mut cycles = Vec::with_capacity(opts.samples);
		let mut nss = Vec::with_capacity(opts.samples);
		for _ in 0..opts.samples {
			let (cy, ns) = timed(clock, || batch(reps));
			cycles.push(cy);
			nss.push(ns);
		}
		let ghz1 = core_ghz();
		let ghz = 0.5 * (ghz0 + ghz1);
		let rejects: Vec<CounterReject> = cycles
			.iter()
			.filter_map(|cy| match cy {
				Cycles::Rejected(r) => Some(*r),
				_ => None,
			})
			.collect();
		let perf_reject = reject_summary(&rejects, opts.samples);
		let fpc = perf_reject.is_none().then(|| {
			let fpc: Vec<f64> = cycles
				.iter()
				.zip(&nss)
				.map(|(cy, ns)| {
					let cy = match cy {
						Cycles::Counted(c) => *c,
						Cycles::FromWall => ns * ghz,
						Cycles::Rejected(_) => unreachable!(),
					};
					flops_per_call * reps as f64 / cy
				})
				.collect();
			quartiles(&fpc)
		});
		let per_call: Vec<f64> =
			nss.iter().map(|ns| ns / reps as f64).collect();
		Row {
			fpc,
			perf_reject,
			ns: quartiles(&per_call),
			ghz: (ghz0, ghz1),
			reps,
		}
	}

	pub fn main() {
		let opts = parse();
		pin(opts.cpu);
		let caches = Caches {
			l1d: cache_bytes(opts.cpu, 1).unwrap_or(32 << 10),
			l2: cache_bytes(opts.cpu, 2).unwrap_or(1 << 20),
			l3: cache_bytes(opts.cpu, 3).unwrap_or(32 << 20),
		};
		let mut clock = match perf_cycles_counter() {
			Ok(f) => Clock::Perf(f),
			Err(e) => {
				eprintln!(
					"gemm_microkernel: no cycle counter ({e}); using wall \
					 time x add-chain clock"
				);
				Clock::Calibrated
			},
		};
		let model = std::fs::read_to_string("/proc/cpuinfo")
			.ok()
			.and_then(|s| {
				s.lines()
					.find_map(|l| l.strip_prefix("model name"))
					.map(|l| l.trim_start_matches([' ', '\t', ':']).to_string())
			})
			.unwrap_or_default();
		let siblings = sysfs(&format!(
			"/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
			opts.cpu
		))
		.unwrap_or_default();

		println!("# gemm_microkernel (tensor4all/colmat-rs#253 E0)");
		println!("# cpu_model: {model}");
		println!("# pinned_cpu: {} (smt siblings {siblings})", opts.cpu);
		println!(
			"# caches: l1d={} l2={} l3={} bytes (cpu{} sysfs)",
			caches.l1d, caches.l2, caches.l3, opts.cpu
		);
		println!("# private_gemm_x86: {}", private_gemm_pin());
		println!("# clock: {}", clock.name());
		println!(
			"# samples={} sample_ms={} prime_ms={} kc_override={:?} ldc={} \
			 peak_nominal: avx512={} avx2={}",
			opts.samples,
			opts.sample_ms,
			opts.prime_ms,
			opts.kc,
			opts.ldc,
			opts.peak512,
			opts.peak256
		);

		let cols = [
			"case",
			"isa",
			"table",
			"mr",
			"n",
			"rows",
			"res",
			"b",
			"kc",
			"a_panels",
			"c_tiles",
			"ldc",
			"reps",
			"fpc_med",
			"fpc_q1",
			"fpc_q3",
			"pct_nom",
			"pct_fma",
			"ns_med",
			"ns_q1",
			"ns_q3",
			"gflops_med",
			"ghz_pre",
			"ghz_post",
			"check",
		];
		println!("{}", cols.join("\t"));

		let want = |label: &str| {
			opts.filter.as_deref().is_none_or(|f| label.contains(f))
		};

		// rows whose cycle counter samples were rejected
		let mut perf_rejected = 0usize;

		// FMA ceilings
		let mut fma_peak = [f64::NAN, f64::NAN];
		for (k, (isa, name, lanes)) in
			[(Isa::Avx512, "peak/zmm", 8.0), (Isa::Avx2, "peak/ymm", 4.0)]
				.into_iter()
				.enumerate()
		{
			if !isa.available() {
				continue;
			}
			let iters = 1000u64;
			let flops = iters as f64 * 12.0 * lanes * 2.0;
			let row = measure(&mut clock, &opts, flops, |reps| {
				for _ in 0..reps {
					match isa {
						Isa::Avx512 => fma_zmm(iters),
						Isa::Avx2 => fma_ymm(iters),
					}
				}
			});
			fma_peak[k] = row.fpc.map_or(f64::NAN, |q| q.1);
			let nom = match isa {
				Isa::Avx512 => opts.peak512,
				Isa::Avx2 => opts.peak256,
			};
			let mut f: Vec<String> = vec![name.into(), isa.name().into()];
			f.extend(std::iter::repeat_n("-".to_string(), 10));
			f.extend([
				row.reps.to_string(),
				fmt_rate(row.fpc.map(|q| q.1), 3),
				fmt_rate(row.fpc.map(|q| q.0), 3),
				fmt_rate(row.fpc.map(|q| q.2), 3),
				fmt_rate(row.fpc.map(|q| 100.0 * q.1 / nom), 1),
				fmt_rate(row.fpc.map(|_| 100.0), 1),
				format!("{:.1}", row.ns.1),
				format!("{:.1}", row.ns.0),
				format!("{:.1}", row.ns.2),
				format!("{:.2}", flops / row.ns.1),
				format!("{:.4}", row.ghz.0),
				format!("{:.4}", row.ghz.1),
				row.perf_reject.clone().unwrap_or_else(|| "-".into()),
			]);
			perf_rejected += row.perf_reject.is_some() as usize;
			assert_eq!(f.len(), cols.len());
			println!("{}", f.join("\t"));
		}

		let mut failed = 0;
		for case in suite() {
			let label = case.label();
			if !want(&label) || !case.shape.isa.available() {
				continue;
			}
			let mut p = prepare(&case, &caches, &opts);
			let status = match check(&case, &mut p) {
				Ok(()) => "ok".to_string(),
				Err(e) => {
					failed += 1;
					eprintln!("{label}: CHECK FAILED: {e}");
					"FAIL".to_string()
				},
			};
			let Prepared {
				a,
				a_panels,
				b,
				c,
				tiles,
				ldc,
				kc,
			} = &p;
			let alpha = 1.0f64;
			let mut inf = info(&case, *kc, c.ptr, *ldc, &alpha);
			let panel = case.shape.mr * kc;
			let flops = 2.0 * (case.rows * case.shape.n * kc) as f64;
			let mut next_a = 0usize;
			let mut next_c = 0usize;
			let row = measure(&mut clock, &opts, flops, |reps| {
				for _ in 0..reps {
					// SAFETY: offsets stay inside the buffers `prepare` sized
					unsafe {
						inf.ptr = c.ptr.add(tiles[next_c]) as *mut ();
						call(&case, a.ptr.add(next_a * panel), b.ptr, &inf);
					}
					next_a += 1;
					if next_a == *a_panels {
						next_a = 0;
					}
					next_c += 1;
					if next_c == tiles.len() {
						next_c = 0;
					}
				}
			});
			let (nom, fma) = match case.shape.isa {
				Isa::Avx512 => (opts.peak512, fma_peak[0]),
				Isa::Avx2 => (opts.peak256, fma_peak[1]),
			};
			let unstable = (row.ghz.0 - row.ghz.1).abs() > 0.01 * row.ghz.0;
			let b = match case.ldb {
				None => "packed".to_string(),
				Some(ldb) => format!("direct:{ldb}"),
			};
			let f: Vec<String> = vec![
				label,
				case.shape.isa.name().into(),
				case.shape.table.into(),
				case.shape.mr.to_string(),
				case.shape.n.to_string(),
				case.rows.to_string(),
				case.res.name().into(),
				b,
				kc.to_string(),
				a_panels.to_string(),
				tiles.len().to_string(),
				ldc.to_string(),
				row.reps.to_string(),
				fmt_rate(row.fpc.map(|q| q.1), 3),
				fmt_rate(row.fpc.map(|q| q.0), 3),
				fmt_rate(row.fpc.map(|q| q.2), 3),
				fmt_rate(row.fpc.map(|q| 100.0 * q.1 / nom), 1),
				fmt_rate(row.fpc.map(|q| 100.0 * q.1 / fma), 1),
				format!("{:.1}", row.ns.1),
				format!("{:.1}", row.ns.0),
				format!("{:.1}", row.ns.2),
				format!("{:.2}", flops / row.ns.1),
				format!("{:.4}", row.ghz.0),
				format!("{:.4}", row.ghz.1),
				format!(
					"{status}{}{}",
					if unstable { ",unstable" } else { "" },
					row.perf_reject
						.as_deref()
						.map(|r| format!(",{r}"))
						.unwrap_or_default()
				),
			];
			perf_rejected += row.perf_reject.is_some() as usize;
			assert_eq!(f.len(), cols.len());
			println!("{}", f.join("\t"));
		}
		if failed > 0 {
			eprintln!(
				"gemm_microkernel: {failed} case(s) failed the correctness \
				 smoke"
			);
			std::process::exit(1);
		}
		if perf_rejected > 0 {
			eprintln!(
				"gemm_microkernel: {perf_rejected} row(s) had cycle-counter \
				 samples rejected (multiplexed or zero count); their \
				 FLOP/cycle is `-`"
			);
			std::process::exit(4);
		}
	}

	#[cfg(test)]
	mod tests {
		use super::*;

		fn sample(count: u64, enabled_ns: u64, running_ns: u64) -> PerfSample {
			PerfSample {
				count,
				enabled_ns,
				running_ns,
			}
		}

		#[test]
		fn perf_sample_parses_value_enabled_running_in_order() {
			let mut b = [0u8; 24];
			b[..8].copy_from_slice(&7u64.to_ne_bytes());
			b[8..16].copy_from_slice(&11u64.to_ne_bytes());
			b[16..].copy_from_slice(&13u64.to_ne_bytes());
			assert_eq!(PerfSample::from_bytes(&b), sample(7, 11, 13));
		}

		#[test]
		fn full_interval_counts() {
			let s0 = sample(1_000, 5_000, 5_000);
			let s1 = sample(41_000, 15_000, 15_000);
			assert_eq!(cycles_between(s0, s1), Ok(40_000));
		}

		#[test]
		fn half_scheduled_interval_is_rejected_with_its_multiplier() {
			// the event ran 5 of the 10 us it was enabled and counted half the
			// cycles: the raw delta would double FLOP/cycle
			let s0 = sample(0, 0, 0);
			let s1 = sample(20_000, 10_000, 5_000);
			assert_eq!(
				cycles_between(s0, s1),
				Err(CounterReject::Multiplexed { multiplier: 2.0 })
			);
		}

		#[test]
		fn never_scheduled_and_zero_count_are_rejected() {
			let s0 = sample(5, 100, 100);
			assert_eq!(
				cycles_between(s0, sample(5, 900, 100)),
				Err(CounterReject::Multiplexed {
					multiplier: f64::INFINITY
				})
			);
			assert_eq!(
				cycles_between(s0, sample(5, 100, 100)),
				Err(CounterReject::Multiplexed {
					multiplier: f64::INFINITY
				})
			);
			assert_eq!(
				cycles_between(s0, sample(5, 900, 900)),
				Err(CounterReject::ZeroCount)
			);
			assert_eq!(
				cycles_between(s0, sample(4, 900, 900)),
				Err(CounterReject::NotMonotonic)
			);
		}

		#[test]
		fn any_rejected_sample_withholds_the_rate() {
			assert_eq!(reject_summary(&[], 31), None);
			let r = [
				CounterReject::Multiplexed { multiplier: 1.25 },
				CounterReject::Multiplexed { multiplier: 2.0 },
				CounterReject::ZeroCount,
			];
			assert_eq!(
				reject_summary(&r, 31).as_deref(),
				Some("perf_reject:3/31:x2.000:zero")
			);
			assert_eq!(fmt_rate(None, 3), "-");
			assert_eq!(fmt_rate(Some(f64::NAN), 1), "-");
			assert_eq!(fmt_rate(Some(15.98765), 3), "15.988");
		}

		fn args(s: &str) -> Vec<String> {
			s.split_whitespace().map(String::from).collect()
		}

		#[test]
		fn a_second_cpu_is_refused() {
			let e = parse_args(&args("--cpu 2 --cpu 3")).err().unwrap();
			assert!(e.starts_with("--cpu given twice"), "{e}");
			assert_eq!(
				parse_args(&args("--cpu 2 --samples 5")).unwrap().cpu,
				2
			);
			assert!(parse_args(&args("--samples 5")).is_err());
			assert!(parse_args(&args("--cpu 2 --samples 3")).is_err());
		}
	}
}

fn main() {
	#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
	imp::main();
	#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
	{
		eprintln!("gemm_microkernel: x86_64 Linux only");
		std::process::exit(2);
	}
}
