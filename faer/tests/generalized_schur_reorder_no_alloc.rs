//! Counting allocator follows the existing svd_values_no_alloc test pattern.
#![cfg(feature = "std")]
mod common;
use common::*;
use colmat::MatMut;
use colmat::linalg::gevd;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
// The sequential call allocates only on this thread. Ignore allocations made
// by libtest's watchdog/reporting threads while the measured region is active.
thread_local! {
	static COUNTING: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn counting() -> bool {
	COUNTING.with(Cell::get)
}
// SAFETY: Every method forwards the original allocation contract to System.
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if counting() {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		// SAFETY: GlobalAlloc forwards the caller-valid allocation contract unchanged.
		unsafe { System.alloc(layout) }
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if counting() {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		// SAFETY: GlobalAlloc forwards the caller-valid allocation contract unchanged.
		unsafe { System.alloc_zeroed(layout) }
	}

	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		if counting() {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
		}
		// SAFETY: GlobalAlloc forwards the caller-valid allocation contract unchanged.
		unsafe { System.realloc(ptr, layout, new_size) }
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		// SAFETY: GlobalAlloc forwards the caller-valid allocation contract unchanged.
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn check<T: ComplexField>(
	blocks: bool,
	mut reorder: impl FnMut(
		MatMut<'_, T>,
		MatMut<'_, T>,
		Option<MatMut<'_, T>>,
		Option<MatMut<'_, T>>,
		usize,
		usize,
		&mut MemStack,
	) -> Result<usize, gevd::SchurReorderError>,
) {
	let mut s = Mat::<T>::from_fn(4, 4, |i, j| {
		if i == j {
			from_f64((i + 1) as f64)
		} else if i < j {
			from_f64(0.1)
		} else {
			zero()
		}
	});
	let mut t = Mat::<T>::identity(4, 4);
	let mut q = t.clone();
	let mut z = t.clone();
	if blocks {
		s[(0, 0)] = from_f64(0.1);
		s[(1, 1)] = from_f64(0.1);
		s[(0, 1)] = one();
		s[(1, 0)] = -one::<T>();
		s[(2, 2)] = from_f64(0.5);
		s[(3, 3)] = from_f64(0.5);
		s[(2, 3)] = one();
		s[(3, 2)] = -one::<T>();
	}
	let last = if blocks { 2 } else { 3 };
	let mut mem = MemBuffer::new(gevd::reorder_schur_scratch::<T>(4));
	reorder(
		s.as_mut(),
		t.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		last,
		0,
		MemStack::new(&mut mem),
	)
	.unwrap();
	ALLOCATIONS.store(0, Ordering::Relaxed);
	COUNTING.with(|flag| flag.set(true));
	let result = reorder(
		s.as_mut(),
		t.as_mut(),
		Some(q.as_mut()),
		Some(z.as_mut()),
		0,
		last,
		MemStack::new(&mut mem),
	);
	COUNTING.with(|flag| flag.set(false));
	result.unwrap();
	assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}
#[test]
fn all_native_scalars_allocate_nothing() {
	check::<f32>(false, |s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
	check::<f64>(false, |s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
	check::<c32>(false, |s, t, q, z, i, j, stack| {
		gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
	});
	check::<c64>(false, |s, t, q, z, i, j, stack| {
		gevd::reorder_schur_cplx(s, t, q, z, i, j, Par::Seq, stack)
	});
	real_conjugate_blocks_allocate_nothing();
	eigenvalues_no_alloc::<f32>();
	eigenvalues_no_alloc::<f64>();
}

fn real_conjugate_blocks_allocate_nothing() {
	check::<f32>(true, |s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
	check::<f64>(true, |s, t, q, z, i, j, stack| {
		gevd::reorder_schur_real(s, t, q, z, i, j, Par::Seq, stack)
	});
}

fn eigenvalues_no_alloc<T: RealField>() {
	let mut s = Mat::<T>::zeros(2, 2);
	s[(0, 1)] = one();
	s[(1, 0)] = -one::<T>();
	let t = Mat::<T>::identity(2, 2);
	let mut ar = Col::<T>::zeros(2);
	let mut ai = ar.clone();
	let mut be = ar.clone();
	gevd::schur_eigenvalues_real(
		s.as_ref(),
		t.as_ref(),
		ar.as_mut(),
		ai.as_mut(),
		be.as_mut(),
	)
	.unwrap();
	ALLOCATIONS.store(0, Ordering::Relaxed);
	COUNTING.with(|flag| flag.set(true));
	gevd::schur_eigenvalues_real(
		s.as_ref(),
		t.as_ref(),
		ar.as_mut(),
		ai.as_mut(),
		be.as_mut(),
	)
	.unwrap();
	COUNTING.with(|flag| flag.set(false));
	assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
	assert!(
		((&ai[0] / &be[0]).abs() - one::<T>()).abs()
			<= eps::<T>() * from_f64::<T>(10.0)
	);
}
