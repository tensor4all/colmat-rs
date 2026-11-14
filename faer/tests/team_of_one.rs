//! tensor4all/colmat-rs#168: a `Par::Team` holding a one-participant team runs sequentially.
//!
//! A caller often holds a team whose size comes from configuration, and 1 is a legitimate size.
//! `utils::thread::with_width` returned such a `par` unchanged (its `width >= par.degree()` early
//! return came before the `width < 2` one), so a kernel took its parallel branch with one thread:
//! `matvec` on an `n x 1` column-major product allocated an `n`-element partial buffer per call and
//! added it back, where `Par::Seq` allocates nothing.
#![cfg(all(feature = "std", feature = "threadteam"))]
use colmat::utils::thread::with_width;
use colmat::{Accum, Mat, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;
static COUNTING: AtomicBool = AtomicBool::new(false);
thread_local! {
	// only the measuring thread counts: the harness, the other test and a pool's starting
	// workers allocate on their own threads while the counted call runs
	static MINE: Cell<bool> = const { Cell::new(false) };
}
fn counting() -> bool {
	COUNTING.load(Ordering::Relaxed)
		&& MINE.try_with(|m| m.get()).unwrap_or(false)
}
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if counting() {
			BYTES.fetch_add(layout.size(), Ordering::Relaxed);
		}
		unsafe { System.alloc(layout) }
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if counting() {
			BYTES.fetch_add(layout.size(), Ordering::Relaxed);
		}
		unsafe { System.alloc_zeroed(layout) }
	}
	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		if counting() {
			BYTES.fetch_add(new_size, Ordering::Relaxed);
		}
		unsafe { System.realloc(ptr, layout, new_size) }
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Runs `f` with the authority over a held team of one participant.
fn with_team_of_one<R>(f: impl FnOnce(Par<'_>) -> R) -> R {
	let team = threadteam::Pool::builder()
		.threads(1)
		.acquire_timeout(std::time::Duration::from_secs(30))
		.build()
		.unwrap()
		.into_team(1)
		.expect("an idle pool");
	let mut exec = team.exec(1);
	exec.team(|t| {
		let par = Par::Team(t);
		assert_eq!(par.degree(), 1);
		f(par)
	})
	.expect("a team token for an idle team")
}

#[test]
fn with_width_runs_a_team_of_one_sequentially() {
	for width in [0, 1, 2, 8] {
		with_team_of_one(|par| {
			let got = with_width(par, width, |par| matches!(par, Par::Seq));
			assert!(
				got,
				"with_width(team of 1, {width}) passed the team on instead of Par::Seq"
			);
		});
	}
	// the sequential case is unchanged
	for width in [0, 1, 2] {
		assert!(with_width(Par::Seq, width, |par| matches!(par, Par::Seq)));
	}
}

/// The kernel's parallel branch keeps an `n`-element partial buffer per thread, allocated on the
/// thread's first call and reused after. `Par::Seq` needs none, so on a fresh thread a team of one
/// must allocate nothing either: the first call is the one that shows which branch ran.
#[test]
fn matvec_on_a_team_of_one_takes_the_sequential_branch() {
	let n = 1 << 16;
	let k = 4;
	let lhs = Mat::<f64>::from_fn(n, k, |i, j| ((i + 3 * j) % 17) as f64 - 8.0);
	let rhs = Mat::<f64>::from_fn(k, 1, |j, _| 0.5 + j as f64);
	let mut want = Mat::<f64>::from_fn(n, 1, |i, _| i as f64);
	colmat::linalg::matmul::matmul(
		want.as_mut(),
		Accum::Add,
		lhs.as_ref(),
		rhs.as_ref(),
		2.0,
		Par::Seq,
	);

	// a fresh thread: no partial buffer from an earlier call
	std::thread::scope(|s| {
		s.spawn(|| {
			with_team_of_one(|par| {
				let mut dst = Mat::<f64>::from_fn(n, 1, |i, _| i as f64);
				MINE.with(|m| m.set(true));
				BYTES.store(0, Ordering::Relaxed);
				COUNTING.store(true, Ordering::Relaxed);
				colmat::linalg::matmul::matmul(
					dst.as_mut(),
					Accum::Add,
					lhs.as_ref(),
					rhs.as_ref(),
					2.0,
					par,
				);
				COUNTING.store(false, Ordering::Relaxed);
				let bytes = BYTES.load(Ordering::Relaxed);
				assert_eq!(
					bytes, 0,
					"an n x 1 product on a team of one allocated {bytes} bytes (Par::Seq allocates none)"
				);
				assert_eq!(dst, want);
			});
		});
	});
}
