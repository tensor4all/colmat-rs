use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::qr::no_pivoting::factor as qrf;
use faer::{Mat, Par};
fn main() {
    let ns: Vec<usize> = std::env::args().nth(1).unwrap_or("96,128".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let batches: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(15);
    let inner: usize = std::env::args().nth(3).map(|s| s.parse().unwrap()).unwrap_or(200);
    for n in ns {
        let a = Mat::<f64>::from_fn(n, n, |i, j| (((i * 31 + j * 17 + i*j) % 23) as f64) / 23.0 - 0.5 + if i==j {1.0} else {0.0});
        let bs = qrf::recommended_block_size::<f64>(n, n);
        let mut q = a.clone();
        let mut h = Mat::<f64>::zeros(bs, n);
        let mut mem = MemBuffer::new(qrf::qr_in_place_scratch::<f64>(n, n, bs, faer::ParShape::Seq, Default::default()));
        let mut ts = vec![]; qrf::TMP_SWEEP.store(0,std::sync::atomic::Ordering::Relaxed); qrf::TMP_T.store(0,std::sync::atomic::Ordering::Relaxed);
        for _ in 0..batches {
            let mut tot = 0.0;
            for _ in 0..inner {
                q.copy_from(&a);
                let t = std::time::Instant::now();
                qrf::qr_in_place(q.as_mut(), h.as_mut(), Par::Seq, MemStack::new(&mut mem), Default::default());
                tot += t.elapsed().as_secs_f64();
                std::hint::black_box(&q);
            }
            ts.push(tot / inner as f64 * 1e6);
        }
        let calls=(batches*inner) as f64; println!("  stage (timer-inclusive): sweep {:.1} us, T-factor build {:.1} us per call", qrf::TMP_SWEEP.load(std::sync::atomic::Ordering::Relaxed) as f64/calls/1e3, qrf::TMP_T.load(std::sync::atomic::Ordering::Relaxed) as f64/calls/1e3);
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("colmat qr_in_place n={n} bs={bs}: median {:.1} us min {:.1} us ({batches} batches x {inner})", ts[batches/2], ts[0]);
    }
}
