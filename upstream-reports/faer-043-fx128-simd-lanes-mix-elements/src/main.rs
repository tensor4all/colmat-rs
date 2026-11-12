// `Mat<fx128>` stores its elements contiguously, so a column is laid out as
// `[h0 l0 h1 l1 ...]` (array of structures). The SIMD hooks of `fx128` treat
// a register pair `Quad(a, b)` as high lanes `a` and low lanes `b` (structure
// of arrays), which is what `qd::simd`'s arithmetic expects. But
// `simd_mask_load_raw` loads `a` from the first `LANES` f64 of memory and `b`
// from the next `LANES`, with no deinterleave, so `a = [h0, l0, ...]` and
// `b = [h1, l1, ...]`: each lane combines parts of different elements.
//
// The program loads a few `fx128` values through the hooks, squares them with
// `simd_mul`, stores them back, and compares each with the scalar product.
use faer::fx128;
use faer::traits::ComplexField;
use pulp::{Simd, WithSimd};
use qd::Quad;

struct Probe<'a>(&'a [fx128]);

impl WithSimd for Probe<'_> {
    type Output = bool;

    fn with_simd<S: Simd>(self, simd: S) -> bool {
        let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
        let ctx = fx128::simd_ctx(simd);
        let mask = fx128::simd_mem_mask_between(&ctx, 0, lanes as u64);
        // SAFETY: `self.0` holds at least `lanes` elements and the mask covers
        // exactly those.
        let v = unsafe { fx128::simd_mask_load_raw(&ctx, mask, self.0.as_ptr() as *const _) };
        let reg_hi: &[f64] = bytemuck::cast_slice(core::slice::from_ref(&v.0));
        let reg_lo: &[f64] = bytemuck::cast_slice(core::slice::from_ref(&v.1));
        println!("SIMD lanes: {lanes}");
        println!("register read as high lanes: {reg_hi:?}");
        println!("register read as low lanes:  {reg_lo:?}");

        let squared = fx128::simd_mul(&ctx, v, v);
        let mut out = vec![Quad::ZERO; lanes];
        // SAFETY: `out` holds `lanes` elements and the mask covers exactly those.
        unsafe { fx128::simd_mask_store_raw(&ctx, mask, out.as_mut_ptr() as *mut _, squared) };

        let mut wrong = false;
        for (i, (&x, &got)) in self.0.iter().zip(&out).enumerate() {
            let expected = x * x;
            let rel = ((got - expected).abs() / expected.abs()).0;
            println!(
                "x[{i}]^2 expected ({:e}, {:e}) actual ({:e}, {:e}) relative error {rel:.1e}",
                expected.0, expected.1, got.0, got.1
            );
            // a correct double-double product is within a few units of 2^-104
            wrong |= !(rel <= 1e-29);
        }
        wrong
    }
}

fn main() {
    // k / 3 for k = 1..8: every value has a nonzero low part except the
    // integers, so a mix-up of high and low parts cannot cancel out.
    let x: Vec<fx128> = (1..=8).map(|k| Quad::from(k as f64) / Quad::from(3.0)).collect();
    println!(
        "memory (hi, lo): {:?}",
        x.iter().map(|q| (q.0, q.1)).collect::<Vec<_>>()
    );
    if pulp::Arch::new().dispatch(Probe(&x)) {
        println!("WRONG: the fx128 SIMD hooks mix the parts of different elements");
        std::process::exit(1);
    }
}
