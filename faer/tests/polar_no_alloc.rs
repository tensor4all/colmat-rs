//! The polar decomposition (`linalg::polar`, colmat-rs#60) allocates nothing on the heap once the
//! caller's `MemStack` is sized by `left_polar_scratch` / `right_polar_scratch`: both methods, left
//! and right, real and complex. The allocator counts every allocation **of the counting thread**
//! while the call runs (the pattern of `svd_values_no_alloc.rs`); the matrices and the scratch are
//! built outside the counted interval.
#![cfg(feature = "std")]
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::polar::{
	PolarMethod, PolarParams, left_polar, left_polar_scratch, right_polar,
	right_polar_scratch,
};
use colmat::traits::ComplexField;
use colmat::{Auto, Mat, Par, ParShape, c64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn count() {
	if COUNTING.with(Cell::get) {
		ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
	}
}
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		count();
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		count();
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		count();
		unsafe { System.realloc(ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations of one `Par::Seq` polar call on `a` after a warm-up call.
fn allocations<T: ComplexField>(a: &Mat<T>, method: PolarMethod) -> usize {
	let (m, n) = a.shape();
	let k = Ord::min(m, n);
	let mut params = <PolarParams as Auto<T>>::auto();
	params.method = method;
	let params = params.into();
	let mut mem = MemBuffer::new(if m >= n {
		left_polar_scratch::<T>(m, n, ParShape::Seq, params)
	} else {
		right_polar_scratch::<T>(m, n, ParShape::Seq, params)
	});
	let mut w = Mat::<T>::zeros(m, n);
	let mut p = Mat::<T>::zeros(k, k);
	let mut run = || {
		let stack = MemStack::new(&mut mem);
		if m >= n {
			left_polar(
				a.as_ref(),
				w.as_mut(),
				p.as_mut(),
				Par::Seq,
				stack,
				params,
			)
		} else {
			right_polar(
				a.as_ref(),
				p.as_mut(),
				w.as_mut(),
				Par::Seq,
				stack,
				params,
			)
		}
		.unwrap()
	};
	run(); // warm-up: first-use allocations (lazy statics, feature detection) are not the call's
	ALLOCATIONS.store(0, Ordering::SeqCst);
	COUNTING.with(|c| c.set(true));
	run();
	COUNTING.with(|c| c.set(false));
	ALLOCATIONS.load(Ordering::SeqCst)
}

#[test]
fn polar_allocates_nothing() {
	for (m, n) in [(40, 40), (90, 30), (30, 90)] {
		let re = Mat::<f64>::from_fn(m, n, |i, j| {
			(((i * 37 + j * 11) % 23) as f64 - 11.0) / 7.0
				+ if i == j { 3.0 } else { 0.0 }
		});
		let complex = Mat::<c64>::from_fn(m, n, |i, j| {
			c64::new(re[(i, j)], re[((i + 1) % m, j)])
		});
		for method in [PolarMethod::Svd, PolarMethod::Qdwh] {
			let counts =
				[allocations(&re, method), allocations(&complex, method)];
			assert_eq!(
				counts,
				[0, 0],
				"{m}x{n} {method:?}: allocations (f64, c64)"
			);
		}
	}
}
