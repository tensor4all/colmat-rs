//! Probe for `scripts/check-isa-gate.sh` (colmat-rs#320): a binary that instantiates the
//! x86_64 kernels the gate inspects, so that their symbols exist in the final executable.
//! It is never run for its output; the gate disassembles it.
use colmat::prelude::*;
use colmat::{Accum, Mat};
use core::hint::black_box;

fn main() {
	let n = black_box(96usize);
	let a = Mat::<f64>::from_fn(n, n, |i, j| (i + 2 * j) as f64 + 1.0);
	let b = Mat::<f64>::from_fn(n, n, |i, j| (i * j) as f64 + 1.0);
	let mut c = Mat::<f64>::zeros(n, n);
	colmat::linalg::matmul::matmul(&mut c, Accum::Replace, &a, &b, 1.0, Par::Seq);
	let a32 = Mat::<f32>::from_fn(n, n, |i, j| (i + j) as f32);
	let mut c32 = Mat::<f32>::zeros(n, n);
	colmat::linalg::matmul::matmul(&mut c32, Accum::Replace, &a32, &a32, 1.0, Par::Seq);
	let z = Mat::<c64>::from_fn(n, n, |i, j| c64::new(i as f64, j as f64));
	let mut cz = Mat::<c64>::zeros(n, n);
	colmat::linalg::matmul::matmul(&mut cz, Accum::Replace, &z, &z, c64::new(1.0, 0.0), Par::Seq);
	// a decomposition of each family that has SIMD loops of its own
	let spd = &c * c.transpose() + Mat::<f64>::identity(n, n);
	let _ = spd.llt(colmat::Side::Lower, Par::Seq).is_ok();
	let _ = spd.self_adjoint_eigenvalues(colmat::Side::Lower, Par::Seq);
	let _ = a.svd(Par::Seq).is_ok();
	let _ = a.qr(Par::Seq);
	let _ = a.partial_piv_lu(Par::Seq);
	black_box((&c, &c32, &cz));
}
