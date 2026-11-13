// `faer` exports `fx128` as a supported scalar, and its own unit tests
// multiply `Mat<fx128>` matrices. Those tests pass only because the SIMD
// micro-kernels of the dense product refuse every non-native scalar with a
// bare `panic!()` unless the crate itself is compiled with `cfg(test)`. A
// downstream crate never compiles `faer` with `cfg(test)`, so the same product
// panics there.
use faer::{fx128, Mat};
use qd::Quad;

fn entry(i: usize) -> fx128 {
    Quad::from(1.0 + i as f64) / Quad::from(3.0 + 7.0 * i as f64)
}

fn main() {
    let (m, k, n) = (13, 17, 11);

    // Control: the same shape in f64 runs the same micro-kernels.
    let a64 = Mat::<f64>::from_fn(m, k, |i, j| (i + 3 * j) as f64);
    let b64 = Mat::<f64>::from_fn(k, n, |i, j| (2 * i + j + 1) as f64);
    let c64 = &a64 * &b64;
    println!("control: f64 {m}x{k} * {k}x{n} gives shape {:?}", c64.shape());

    let a = Mat::<fx128>::from_fn(m, k, |i, j| entry(i + 3 * j));
    let b = Mat::<fx128>::from_fn(k, n, |i, j| entry(2 * i + j + 1));
    let product = std::panic::catch_unwind(|| &a * &b);
    match product {
        Ok(c) => {
            let mut worst = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut r = Quad::ZERO;
                    for l in 0..k {
                        r += a[(i, l)] * b[(l, j)];
                    }
                    worst = worst.max(((c[(i, j)] - r).abs() / r.abs()).0);
                }
            }
            println!("fx128 {m}x{k} * {k}x{n} expected a product actual a product, worst relative error {worst:.1e}");
            if !(worst <= 1e-29) {
                println!("WRONG: the fx128 product is inaccurate");
                std::process::exit(1);
            }
        }
        Err(_) => {
            println!("fx128 {m}x{k} * {k}x{n} expected a product actual panic (see the message above)");
            println!("WRONG: the dense product of fx128 matrices panics outside cfg(test)");
            std::process::exit(1);
        }
    }
}
