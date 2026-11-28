use colmat::linalg::evd::{self, tridiag};
use colmat::prelude::*;
use colmat::{Mat, Par, Spec, diag::Diag};
use dyn_stack::{MemBuffer, MemStack};
use std::time::Instant;

fn make(n: usize) -> Mat<f64> {
    let mut x: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
    let b = Mat::<f64>::from_fn(n, n, |_, _| rnd());
    let mut a = &b * b.transpose();
    for i in 0..n { a[(i, i)] += n as f64; }
    a
}
fn med(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let batches: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(15);
    for &n in &[32usize, 64, 128, 256, 512] {
        let a = make(n);
        let reps = if n <= 64 { 30 } else if n <= 128 { 10 } else { 4 };
        let params = Spec::<evd::SelfAdjointEvdParams, f64>::default();
        let req = evd::self_adjoint_evd_scratch::<f64>(n, evd::ComputeEigenvectors::No, colmat::ParShape::Seq, params);
        let mut mem = MemBuffer::new(req);
        let mut s = Diag::<f64>::zeros(n);
        let (mut tv, mut tt, mut tvec) = (vec![], vec![], vec![]);
        for _ in 0..batches {
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                evd::self_adjoint_evd(a.as_ref(), s.as_mut(), None, Par::Seq, MemStack::new(&mut mem), params).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            tv.push(best * 1e6);
            // tridiag alone
            let bs = colmat::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n, n);
            let tp = Spec::<tridiag::TridiagParams, f64>::default();
            let treq = tridiag::tridiag_in_place_scratch::<f64>(n, colmat::ParShape::Seq, tp);
            let mut tmem = MemBuffer::new(treq);
            let mut h = Mat::<f64>::zeros(bs, n - 1);
            let mut best = f64::MAX;
            for _ in 0..reps {
                let mut w = a.clone();
                let t = Instant::now();
                tridiag::tridiag_in_place(w.as_mut(), h.as_mut(), Par::Seq, MemStack::new(&mut tmem), tp);
                best = best.min(t.elapsed().as_secs_f64());
            }
            tt.push(best * 1e6);
            let mut u = Mat::<f64>::zeros(n, n);
            let vreq = evd::self_adjoint_evd_scratch::<f64>(n, evd::ComputeEigenvectors::Yes, colmat::ParShape::Seq, params);
            let mut vmem = MemBuffer::new(vreq);
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                evd::self_adjoint_evd(a.as_ref(), s.as_mut(), Some(u.as_mut()), Par::Seq, MemStack::new(&mut vmem), params).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            tvec.push(best * 1e6);
        }
        println!("n={n} values_us={:.2} tridiag_us={:.2} vectors_us={:.2}", med(&mut tv), med(&mut tt), med(&mut tvec));
    }
}
