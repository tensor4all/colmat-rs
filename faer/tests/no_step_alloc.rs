//! #111: a parallel region allocates nothing on the heap. Each counted call below runs on a held
//! team after a warm-up. Existing configurable gates are forced with `ParThreshold::Fixed(0)`;
//! Pfaffian uses its derived gate on a larger input. The allocator counts every allocation
//! of the process (any thread) while the call runs. The team,
//! the matrices and the scratch are built outside the counted interval, and the check that the
//! regions did dispatch reads the team counters after it.
#![cfg(all(feature = "std", feature = "threadteam"))]
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::utils::cost::ParThreshold;
use colmat::utils::team::stats;
use colmat::{Auto, Mat, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
/// the largest allocation made while counting: a call that recycles its output buffer still
/// allocates small things, but nothing the size of what it recycles
static LARGEST: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
			LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
		}
		unsafe { System.alloc(layout) }
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
			LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
		}
		unsafe { System.alloc_zeroed(layout) }
	}
	unsafe fn realloc(
		&self,
		ptr: *mut u8,
		layout: Layout,
		new_size: usize,
	) -> *mut u8 {
		if COUNTING.load(Ordering::Relaxed) {
			ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
			LARGEST.fetch_max(new_size, Ordering::Relaxed);
		}
		unsafe { System.realloc(ptr, layout, new_size) }
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations during `f`, the largest of them, and the number of parallel sections and splits it
/// dispatched.
fn counted_bytes(mut f: impl FnMut()) -> (usize, usize, usize) {
	f(); // warm-up: first-use allocations (thread locals, lazy statics) are not the region's
	stats::enable(true);
	let _ = stats::take();
	ALLOCATIONS.store(0, Ordering::SeqCst);
	LARGEST.store(0, Ordering::SeqCst);
	COUNTING.store(true, Ordering::SeqCst);
	f();
	COUNTING.store(false, Ordering::SeqCst);
	let n = ALLOCATIONS.load(Ordering::SeqCst);
	let largest = LARGEST.load(Ordering::SeqCst);
	stats::enable(false);
	let s = stats::take();
	let parallel = s.run.iter().skip(2).sum::<usize>()
		+ s.split.iter().skip(2).sum::<usize>();
	(n, largest, parallel)
}

/// Allocations during `f`, and the number of parallel sections and splits it dispatched.
fn counted(f: impl FnMut()) -> (usize, usize) {
	let (n, _, parallel) = counted_bytes(f);
	(n, parallel)
}

fn sample(m: usize, n: usize, seed: u64) -> Mat<f64> {
	let mut x = seed
		.wrapping_mul(6364136223846793005)
		.wrapping_add(1442695040888963407);
	Mat::from_fn(m, n, |_, _| {
		x = x
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
	})
}

// one test function: the counter is process wide, so nothing else may run while it counts
#[test]
fn parallel_regions_allocate_nothing() {
	use colmat::linalg::cholesky::lblt::factor as lblt;
	use colmat::linalg::lu::full_pivoting::factor as flu;
	for width in [4usize, 8, 70] {
		colmat::team::with_team(width, |mut par: Par<'_>| {
			// Identity-seeded thin Q on a held team: output and scratch are reused.
			if width <= 8 {
				use colmat::linalg::qr::no_pivoting::{factor, unitary};
				let (m, n, block) = (1024, 257, 16);
				let a0 = sample(m, n, 84);
				let mut packed = a0.clone();
				let mut coeff = Mat::<f64>::zeros(block, n);
				let params = colmat::Spec::new(factor::QrParams {
					recursion_leaf_size: 8,
					..<factor::QrParams as Auto<f64>>::auto()
				});
				let mut factor_mem =
					MemBuffer::new(factor::qr_in_place_scratch::<f64>(
						m,
						n,
						block,
						par.shape(),
						params,
					));
				let (allocs, parallel) = counted(|| {
					packed.copy_from(&a0);
					factor::qr_in_place(
						packed.as_mut(),
						coeff.as_mut(),
						par.reborrow(),
						MemStack::new(&mut factor_mem),
						params,
					);
				});
				assert!(
					parallel > 0,
					"QR factor, width {width}: no parallel region"
				);
				assert_eq!(allocs, 0, "QR factor, width {width}");
				let mut q = Mat::<f64>::zeros(m, n);
				let mut mem = MemBuffer::new(unitary::unitary_scratch::<f64>(
					m,
					n,
					n,
					block,
					par.shape(),
				));
				let (allocs, parallel) = counted(|| {
					unitary::unitary(
						q.as_mut(),
						packed.as_ref(),
						coeff.as_ref(),
						false,
						par.reborrow(),
						MemStack::new(&mut mem),
					);
				});
				assert!(
					parallel > 0,
					"QR thin Q, width {width}: no parallel region"
				);
				assert_eq!(allocs, 0, "QR thin Q, width {width}");
				// The small paths (tensor4all/colmat-rs#269): the factor, the thin Q and the
				// sequence apply below their gates, on the held team and sequentially. They run on
				// the calling thread, so only the allocation count is asserted.
				use colmat::linalg::householder;
				for (m, n) in [(32usize, 32usize), (96, 96), (256, 64)] {
					let block = factor::recommended_block_size::<f64>(m, n);
					let k = Ord::min(m, n);
					let a0 = sample(m, n, 269);
					let mut packed = a0.clone();
					let mut coeff = Mat::<f64>::zeros(block, k);
					let mut q = Mat::<f64>::zeros(m, k);
					let mut mem = MemBuffer::new(colmat::dyn_stack::StackReq::all_of(&[
						factor::qr_in_place_scratch::<f64>(m, n, block, par.shape(), Default::default()),
						unitary::unitary_scratch::<f64>(m, n, k, block, par.shape()),
						householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(m, block, k),
					]));
					for seq in [false, true] {
						let (allocs, _) = counted(|| {
							let mut p =
								if seq { Par::Seq } else { par.reborrow() };
							packed.copy_from(&a0);
							factor::qr_in_place(
								packed.as_mut(),
								coeff.as_mut(),
								p.reborrow(),
								MemStack::new(&mut mem),
								Default::default(),
							);
							unitary::unitary(
								q.as_mut(),
								packed.as_ref(),
								coeff.as_ref(),
								false,
								p.reborrow(),
								MemStack::new(&mut mem),
							);
							q.fill(0.0);
							for i in 0..k {
								q[(i, i)] = 1.0;
							}
							householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
								packed.get(.., ..k),
								coeff.as_ref(),
								colmat::Conj::No,
								q.as_mut(),
								p,
								MemStack::new(&mut mem),
							);
						});
						assert_eq!(
							allocs, 0,
							"small QR {m}x{n}, width {width}, seq {seq}"
						);
					}
				}
				// The tridiagonal EVD solver, including its independent merge repairs.
				use colmat::linalg::evd::{
					ComputeEigenvectors, tridiagonal_self_adjoint_evd,
					tridiagonal_self_adjoint_evd_scratch,
				};
				let n = 256;
				let diag =
					colmat::Col::<f64>::from_fn(n, |i| (i % 19) as f64 / 19.0)
						.into_diagonal();
				let sub = colmat::Col::<f64>::from_fn(n - 1, |i| {
					0.2 + (i % 7) as f64 / 13.0
				})
				.into_diagonal();
				let mut values = colmat::diag::Diag::<f64>::zeros(n);
				let mut vectors = Mat::<f64>::zeros(n, n);
				let params = Default::default();
				let mut mem = MemBuffer::new(
					tridiagonal_self_adjoint_evd_scratch::<f64>(
						n,
						ComputeEigenvectors::Yes,
						par.shape(),
						params,
					),
				);
				let (allocs, parallel) = counted(|| {
					tridiagonal_self_adjoint_evd(
						diag.as_ref(),
						sub.as_ref(),
						values.as_mut(),
						Some(vectors.as_mut()),
						par.reborrow(),
						MemStack::new(&mut mem),
						params,
					)
					.unwrap();
				});
				assert!(
					parallel > 0,
					"tridiagonal EVD, width {width}: no parallel region"
				);
				assert_eq!(allocs, 0, "tridiagonal EVD, width {width}");
			}

			// full pivoting LU: the per-step pivot search (`for_each` over column blocks)
			let n = 96;
			let a0 = sample(n, n, 1);
			let mut a = a0.clone();
			let (mut p, mut pi, mut q, mut qi) = (
				vec![0usize; n],
				vec![0usize; n],
				vec![0usize; n],
				vec![0usize; n],
			);
			let params = flu::FullPivLuParams {
				par_threshold: ParThreshold::Fixed(0),
				..<flu::FullPivLuParams as Auto<f64>>::auto()
			};
			let mut mem =
				MemBuffer::new(flu::lu_in_place_scratch::<usize, f64>(
					n,
					n,
					par.shape(),
					params.into(),
				));
			let (allocs, parallel) = counted(|| {
				a.copy_from(&a0);
				flu::lu_in_place(
					a.as_mut(),
					&mut p,
					&mut pi,
					&mut q,
					&mut qi,
					par.reborrow(),
					MemStack::new(&mut mem),
					params.into(),
				);
			});
			assert!(
				parallel > 0,
				"full pivoting LU, width {width}: no parallel region"
			);
			assert_eq!(allocs, 0, "full pivoting LU, width {width}");

			// partial pivoting LU at a 2 KiB column stride: the `U` panel solves stage their right
			// hand sides in a per-thread buffer, which grows during the warm-up and is reused after
			{
				use colmat::linalg::lu::partial_pivoting::factor as plu;
				let n = 256;
				let a0 = sample(n, n, 4);
				let mut a = a0.clone();
				assert_eq!(
					a.col_stride() as usize * core::mem::size_of::<f64>()
						% 1024,
					0
				);
				let (mut p, mut pi) = (vec![0usize; n], vec![0usize; n]);
				let params = plu::PartialPivLuParams {
					par_threshold: ParThreshold::Fixed(0),
					..<plu::PartialPivLuParams as Auto<f64>>::auto()
				};
				let mut mem =
					MemBuffer::new(plu::lu_in_place_scratch::<usize, f64>(
						n,
						n,
						par.shape(),
						params.into(),
					));
				let (allocs, parallel) = counted(|| {
					a.copy_from(&a0);
					plu::lu_in_place(
						a.as_mut(),
						&mut p,
						&mut pi,
						par.reborrow(),
						MemStack::new(&mut mem),
						params.into(),
					);
				});
				assert!(
					parallel > 0,
					"partial pivoting LU, width {width}: no parallel region"
				);
				assert_eq!(allocs, 0, "partial pivoting LU, width {width}");
			}

			// `PartialPivLu::factor_into` (#191): a factorization into a factor buffer that is
			// already resident. The two factor blocks and the two permutation arrays are reused, so
			// what a repeat call still allocates is the `min(m, n) * size_of::<usize>()`-byte
			// transposition scratch and nothing else: the factor buffer's own bytes must never
			// appear, which is the 4.4-4.8 ms first touch this path exists to keep off the repeat
			// calls. The permutation arrays cannot make the count grow either.
			{
				use colmat::linalg::solvers::PartialPivLu;
				let n = 256;
				let a0 = sample(n, n, 9);
				let a1 = sample(n, n, 11);
				let mut lu = PartialPivLu::new(a0.as_ref(), par.reborrow());
				let factor_bytes = 2 * n * n * core::mem::size_of::<f64>();
				let factors = lu.L().as_ptr() as usize;
				lu.factor_into(a1.as_ref(), par.reborrow());
				let (allocs, largest, parallel) = counted_bytes(|| {
					lu.factor_into(a1.as_ref(), par.reborrow());
				});
				assert!(
					parallel > 0,
					"PartialPivLu factor_into, width {width}: no parallel region"
				);
				assert!(
					largest < factor_bytes,
					"PartialPivLu factor_into, width {width}: allocated {largest} bytes, the factor buffer is {factor_bytes}"
				);
				assert_eq!(
					allocs, 1,
					"PartialPivLu factor_into, width {width}: {allocs} allocations, and the transposition scratch is the only per-call one"
				);
				let (again, _, _) = counted_bytes(|| {
					lu.factor_into(a1.as_ref(), par.reborrow());
				});
				assert_eq!(
					allocs, again,
					"PartialPivLu factor_into, width {width}: the count grew on the next call"
				);
				assert!(
					lu.L().as_ptr() as usize == factors,
					"PartialPivLu factor_into, width {width}: the factor buffer moved"
				);
			}

			// Bunch-Kaufman with full pivoting: rank-one and rank-two steps
			let x = sample(n, n, 2);
			let s0 = &x + x.transpose();
			let mut s = s0.clone();
			let mut sub = colmat::diag::Diag::<f64>::zeros(n);
			let params = lblt::LbltParams {
				pivoting: lblt::PivotingStrategy::Full,
				par_threshold: ParThreshold::Fixed(0),
				..<lblt::LbltParams as Auto<f64>>::auto()
			};
			let mut mem = MemBuffer::new(lblt::cholesky_in_place_scratch::<
				usize,
				f64,
			>(n, par.shape(), params.into()));
			let (allocs, parallel) = counted(|| {
				s.copy_from(&s0);
				lblt::cholesky_in_place(
					s.as_mut(),
					sub.as_mut(),
					&mut p,
					&mut pi,
					par.reborrow(),
					MemStack::new(&mut mem),
					params.into(),
				);
			});
			assert!(
				parallel > 0,
				"Bunch-Kaufman, width {width}: no parallel region"
			);
			assert_eq!(allocs, 0, "Bunch-Kaufman, width {width}");
			// both update ranks ran: a 2x2 pivot leaves a nonzero subdiagonal entry, a 1x1 a zero
			let subs: Vec<f64> = (0..n).map(|i| sub[i]).collect();
			assert!(
				subs.iter().any(|&x| x != 0.0) && subs.contains(&0.0),
				"Bunch-Kaufman, width {width}: a pivot size never occurred"
			);

			// Pfaffian: the complete call, including its internally partitioned column updates.
			let x = sample(258, 258, 171);
			let a0 = &x - x.transpose();
			let mut mem = MemBuffer::new(
				colmat::linalg::pfaffian::pfaffian_scratch::<f64>(258),
			);
			let (allocs, parallel) = counted(|| {
				let value = colmat::linalg::pfaffian::pfaffian(
					a0.as_ref(),
					par.reborrow(),
					MemStack::new(&mut mem),
				)
				.unwrap();
				assert!(value.is_finite());
			});
			assert!(
				parallel > 0,
				"Pfaffian, width {width}: no parallel region"
			);
			assert_eq!(allocs, 0, "Pfaffian, width {width}");

			// a plain product on the team: the x86 kernel's parallel job tables come from the
			// calling thread's workspace (faer#223), so the same shape again allocates nothing,
			// and so does a larger shape once the workspace has grown to it
			#[cfg(target_arch = "x86_64")]
			if std::arch::is_x86_feature_detected!("avx2")
				&& std::arch::is_x86_feature_detected!("fma")
			{
				// the same shape again, a larger one (the workspace grows), then one below the
				// grown size
				for &(m, n, k) in &[
					(384usize, 320usize, 448usize),
					(768, 320, 448),
					(384, 320, 448),
				] {
					let a = sample(m, k, 31);
					let b = sample(k, n, 32);
					let mut c = sample(m, n, 33);
					let (allocs, parallel) = counted(|| {
						colmat::linalg::matmul::matmul(
							c.as_mut(),
							colmat::Accum::Replace,
							a.as_ref(),
							b.as_ref(),
							1.0,
							par.reborrow(),
						);
					});
					assert!(
						parallel > 0,
						"matmul {m}x{n}x{k}, width {width}: no parallel region"
					);
					assert_eq!(allocs, 0, "matmul {m}x{n}x{k}, width {width}");
				}
			}

			// `join_sized`, the LU swap stage's split, with closures that capture
			let mut left = [0usize; 4];
			let mut right = [0usize; 4];
			let (allocs, parallel) = counted(|| {
				colmat::utils::thread::join_sized(
					par.reborrow(),
					width / 2,
					|_| left.iter_mut().for_each(|x| *x += 1),
					|_| right.iter_mut().for_each(|x| *x += 2),
				);
			});
			assert!(parallel > 0, "join_sized, width {width}: no split");
			assert_eq!(allocs, 0, "join_sized, width {width}");
		});
	}
}
