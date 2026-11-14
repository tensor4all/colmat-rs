#![allow(non_snake_case)]
use colmat::mat::AsMatMut;
use colmat::prelude::*;
use colmat::{Accum, dyn_stack};
use dyn_stack::{MemBuffer, MemStack, StackReq};
fn foo_scratch(n: usize) -> StackReq {
	let tmp = colmat::linalg::temp_mat_scratch::<f64>(n, n);
	let llt = colmat::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<
		f64,
	>(n, ParShape::Seq, default());
	tmp.and(StackReq::any_of(&[tmp.and(llt), tmp]))
}
// A <- A * B
fn foo(mut A: MatMut<'_, f64>, B: MatRef<'_, f64>, stack: &mut MemStack) {
	let n = A.nrows();
	let (mut tmp0, stack) =
		unsafe { colmat::linalg::temp_mat_uninit(n, n, stack) };
	let mut tmp0: MatMut<'_, f64> = tmp0.as_mat_mut();
	{
		let (mut tmp1, stack) =
			unsafe { colmat::linalg::temp_mat_uninit(n, n, stack) };
		let _tmp1: MatMut<'_, f64> = tmp1.as_mat_mut();
		colmat::linalg::matmul::matmul(
			&mut tmp0,
			Accum::Replace,
			&A,
			&B,
			1.0,
			Par::Seq,
		);
		A.copy_from(&tmp0);
		let _ = colmat::linalg::cholesky::llt::factor::cholesky_in_place(
			A.rb_mut(),
			default(),
			Par::Seq,
			stack,
			default(),
		);
	}
	let (_tmp1, _) =
		unsafe { colmat::linalg::temp_mat_uninit::<f64, _, _>(n, n, stack) };
}
fn main() {
	let n = 128;
	let nthreads = 100;
	let scratch = foo_scratch(n);
	let mut workspace = MemBuffer::new(scratch);
	let stack = MemStack::new(&mut workspace);
	//          stack
	// [....................]
	// let (mut stacks, mut stack) = stack.make_with(nthreads, |_|
	// MemStack::new(&mut [])); stacks    stack
	// [...][.................]
	// for i in 0..nthreads {
	// 	let scratch = foo_scratch(n);
	// 	let new;
	// 	(new, stack) = stack.make_aligned_uninit(scratch.size_bytes(),
	// scratch.align_bytes()); 	// stacks foo_scratch  stack
	// 	// [...][..][..][..][....]
	// 	stacks[i] = MemStack::new(new);
	// }
	// stacks    foo_scratch
	// [...][..][..][..][..][..]
	// use rayon::prelude::*;
	(0..nthreads).for_each(|_| {
		let mut A = Mat::zeros(n, n);
		let B = Mat::zeros(n, n);
		foo(A.as_mut(), B.as_ref(), stack);
	});
}
