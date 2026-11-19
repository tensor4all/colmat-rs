//! One level of Strassen's seven products (Numerische Mathematik 13, 1969,
//! doi:10.1007/BF02165411), with sums formed during packing. Experimental
//! because the sums change rounding, cancellation and overflow behavior.
//! The original AVX2/FMA leaf writes contiguous 8-by-6 tiles (issue #208).
use core::arch::x86_64::*;
use std::cell::RefCell;

// The leaf receives complete packed panels, a positive depth divisible by
// four and a full 48-element product tile, including at the matrix fringe.

#[unsafe(naked)]
unsafe extern "sysv64" fn tile_8x6(
	lhs: *const f64,
	rhs: *const f64,
	dst: *mut f64,
	depth: usize,
	add: usize,
) {
	core::arch::naked_asm!(include_str!("strassen_8x6.s"));
}

#[derive(Clone, Copy)]
struct View {
	ptr: *const f64,
	ld: usize,
}
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec4<const S: i8>(a: View, b: View, i: usize, j: usize) -> __m256d {
	let v = _mm256_loadu_pd(a.ptr.add(i + j * a.ld));
	match S {
		1 => _mm256_add_pd(v, _mm256_loadu_pd(b.ptr.add(i + j * b.ld))),
		-1 => _mm256_sub_pd(v, _mm256_loadu_pd(b.ptr.add(i + j * b.ld))),
		_ => v,
	}
}
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_b4<const S: i8>(
	b: View,
	b2: View,
	k: usize,
	j: usize,
	out: *mut f64,
) {
	let v0 = vec4::<S>(b, b2, k, j);
	let v1 = vec4::<S>(b, b2, k, j + 1);
	let v2 = vec4::<S>(b, b2, k, j + 2);
	let v3 = vec4::<S>(b, b2, k, j + 3);
	let v4 = vec4::<S>(b, b2, k, j + 4);
	let v5 = vec4::<S>(b, b2, k, j + 5);
	let x = _mm256_unpacklo_pd(v0, v1);
	let y = _mm256_unpackhi_pd(v0, v1);
	let z = _mm256_unpacklo_pd(v2, v3);
	let t = _mm256_unpackhi_pd(v2, v3);
	let u = _mm256_unpacklo_pd(v4, v5);
	let v = _mm256_unpackhi_pd(v4, v5);
	_mm256_storeu_pd(out, _mm256_permute2f128_pd::<0x20>(x, z));
	_mm_storeu_pd(out.add(4), _mm256_castpd256_pd128(u));
	_mm256_storeu_pd(out.add(6), _mm256_permute2f128_pd::<0x20>(y, t));
	_mm_storeu_pd(out.add(10), _mm256_castpd256_pd128(v));
	_mm256_storeu_pd(out.add(12), _mm256_permute2f128_pd::<0x31>(x, z));
	_mm_storeu_pd(out.add(16), _mm256_extractf128_pd::<1>(u));
	_mm256_storeu_pd(out.add(18), _mm256_permute2f128_pd::<0x31>(y, t));
	_mm_storeu_pd(out.add(22), _mm256_extractf128_pd::<1>(v));
}
// A quadrant view carries the extent of virtual zero padding on its last
// row/column.
#[derive(Clone, Copy)]
struct Quad {
	ptr: *const f64,
	ld: usize,
	rows: usize,
	cols: usize,
}
impl Quad {
	#[inline(always)]
	fn val(self, i: usize, j: usize) -> f64 {
		if i < self.rows && j < self.cols {
			unsafe { *self.ptr.add(i + j * self.ld) }
		} else {
			0.
		}
	}
}
unsafe fn quads(x: View, n: usize, h: usize) -> [Quad; 4] {
	std::array::from_fn(|i| {
		let r = i / 2 * h;
		let c = i % 2 * h;
		Quad {
			ptr: x.ptr.add(r + c * x.ld),
			ld: x.ld,
			rows: h.min(n - r),
			cols: h.min(n - c),
		}
	})
}
#[inline(always)]
fn elt<const SIGN: i8>(x: Quad, y: Quad, i: usize, j: usize) -> f64 {
	match SIGN {
		1 => x.val(i, j) + y.val(i, j),
		-1 => x.val(i, j) - y.val(i, j),
		_ => x.val(i, j),
	}
}
#[target_feature(enable = "avx2,fma")]
unsafe fn leaf<
	const AS: i8,
	const BS: i8,
	const H: usize,
	const KC: usize,
	const MC: usize,
>(
	a: Quad,
	a2: Quad,
	b: Quad,
	b2: Quad,
	products: &mut [f64],
	work: &mut Panels,
) {
	const MR: usize = 8;
	const NR: usize = 6;
	let mt = H.div_ceil(MR);
	let nt = H.div_ceil(NR);
	for k0 in (0..H).step_by(KC) {
		let dk = (H - k0).min(KC);
		// Interior K rows can be transposed in groups of four. Remaining
		// columns and the last incomplete K group use zero-extending scalar
		// reads.
		let valid_k = if BS == 0 { b.rows } else { b.rows.min(b2.rows) };
		let bk = dk.min(valid_k.saturating_sub(k0)) / 4 * 4;
		let valid_c = if BS == 0 { b.cols } else { b.cols.min(b2.cols) };
		let bv = View {
			ptr: b.ptr,
			ld: b.ld,
		};
		let bv2 = View {
			ptr: b2.ptr,
			ld: b2.ld,
		};
		for jt in 0..nt {
			let j = jt * NR;
			let offset = jt * (NR * dk + PANEL_GAP);
			let out = &mut work.b[offset..offset + NR * dk];
			let fast = if j + NR <= valid_c { bk } else { 0 };
			for k in (0..fast).step_by(4) {
				unsafe {
					pack_b4::<BS>(
						bv,
						bv2,
						k0 + k,
						j,
						out.as_mut_ptr().add(k * NR),
					);
				}
			}
			for k in fast..dk {
				for l in 0..NR {
					out[k * NR + l] = elt::<BS>(b, b2, k0 + k, j + l);
				}
			}
		}
		for start in (0..mt).step_by(MC / MR) {
			let count = (mt - start).min(MC / MR);
			for tb in (0..dk).step_by(8) {
				for it in 0..count {
					let i = (start + it) * MR;
					let stop = (tb + 8).min(dk);
					let fast = i + MR <= a.rows
						&& (AS == 0 || i + MR <= a2.rows)
						&& k0 + stop <= a.cols
						&& (AS == 0 || k0 + stop <= a2.cols);
					for k in tb..stop {
						let offset = it * (MR * dk + PANEL_GAP) + k * MR;
						let y = &mut work.a[offset..offset + MR];
						if fast {
							unsafe {
								let x = a.ptr.add(i + (k + k0) * a.ld);
								if AS == 0 {
									std::ptr::copy_nonoverlapping(
										x,
										y.as_mut_ptr(),
										MR,
									);
								} else {
									let z = a2.ptr.add(i + (k + k0) * a2.ld);
									for l in 0..MR {
										y[l] = if AS == 1 {
											*x.add(l) + *z.add(l)
										} else {
											*x.add(l) - *z.add(l)
										};
									}
								}
							}
						} else {
							for l in 0..MR {
								y[l] = elt::<AS>(a, a2, i + l, k + k0);
							}
						}
					}
				}
			}
			for j in 0..nt {
				for it in 0..count {
					unsafe {
						tile_8x6(
							work.a.as_ptr().add(it * (MR * dk + PANEL_GAP)),
							work.b.as_ptr().add(j * (NR * dk + PANEL_GAP)),
							products
								.as_mut_ptr()
								.add((j * mt + start + it) * 48),
								dk,
							usize::from(k0 != 0),
						);
					}
				}
			}
		}
	}
}
// Applies one level of the seven-product formula to an even core. When
// n=core+1, fuse the missing outer product into each output write.
#[target_feature(enable = "avx2,fma")]
unsafe fn square<const H: usize, const KC: usize, const MC: usize>(
	a: View,
	b: View,
	c: *mut f64,
	ldc: usize,
	n: usize,
	tmp: &mut [f64],
	w: &mut Panels,
) {
	let [aa, ab, ac, ad] = quads(a, n, H);
	let [ba, bb, bc, bd] = quads(b, n, H);
	let mt = H.div_ceil(8);
	let nt = H.div_ceil(6);
	let pitch = mt * nt * 48 + 8;
	macro_rules! prod {
		($idx:expr,$as:expr,$bs:expr,$a:expr,$a2:expr,$b:expr,$b2:expr) => {
			leaf::<{ $as }, { $bs }, H, KC, MC>(
				$a,
				$a2,
				$b,
				$b2,
				&mut tmp[$idx * pitch..($idx + 1) * pitch],
				w,
			)
		};
	}
	prod!(0, 1, 1, aa, ad, ba, bd);
	prod!(1, 1, 0, ac, ad, ba, ba);
	prod!(2, 0, -1, aa, aa, bb, bd);
	prod!(3, 0, -1, ad, ad, bc, ba);
	prod!(4, 1, 0, aa, ab, bd, bd);
	prod!(5, -1, 1, ac, aa, ba, bb);
	prod!(6, -1, 1, ab, ad, bc, bd);
	if n > 2 * H {
		compose_odd::<H, true>(a, b, c, ldc, n, tmp);
	} else {
		compose_odd::<H, false>(a, b, c, ldc, n, tmp);
	}
}
#[target_feature(enable = "avx2,fma")]
unsafe fn compose_odd<const H: usize, const PEEL: bool>(
	a: View,
	b: View,
	c: *mut f64,
	ldc: usize,
	n: usize,
	tmp: &mut [f64],
) {
	let core = 2 * H;
	let mt = H / 8;
	let nt = H.div_ceil(6);
	let pitch = mt * nt * 48 + 8;
	for jt in 0..nt {
		for it in 0..mt {
			for j in 0..6.min(H - jt * 6) {
				let col = jt * 6 + j;
				for ii in [0, 4] {
					let row = it * 8 + ii;
					let q = (jt * mt + it) * 48 + j * 8 + ii;
					unsafe {
						let [p1, p2, p3, p4, p5, p6, p7] =
							std::array::from_fn(|t| {
								_mm256_loadu_pd(tmp.as_ptr().add(t * pitch + q))
							});
						let vals = [
							_mm256_add_pd(
								_mm256_sub_pd(_mm256_add_pd(p1, p4), p5),
								p7,
							),
							_mm256_add_pd(p3, p5),
							_mm256_add_pd(p2, p4),
							_mm256_add_pd(
								_mm256_add_pd(_mm256_sub_pd(p1, p2), p3),
								p6,
							),
						];
						for z in 0..4 {
							let rr = row + (z / 2) * H;
							let cc = col + (z % 2) * H;
							if !PEEL && cc >= n {
								continue;
							}
							let mut v = vals[z];
							if PEEL {
								v = _mm256_fmadd_pd(
									_mm256_loadu_pd(
										a.ptr.add(rr + core * a.ld),
									),
									_mm256_set1_pd(
										*b.ptr.add(core + cc * b.ld),
									),
									v,
								);
							}
							let dst = c.add(rr + cc * ldc);
							if !PEEL && rr + 4 > n {
								_mm256_maskstore_pd(
									dst,
									_mm256_set_epi64x(0, -1, -1, -1),
									v,
								);
							} else {
								_mm256_storeu_pd(dst, v);
							}
						}
					}
				}
			}
		}
	}
	if PEEL {
		let tail = core::slice::from_raw_parts_mut(c.add(core * ldc), n);
		tail.fill(0.);
		for k in 0..n {
			let beta = *b.ptr.add(k + core * b.ld);
			let col = core::slice::from_raw_parts(a.ptr.add(k * a.ld), n);
			for (dst, v) in tail.iter_mut().zip(col) {
				*dst = v.mul_add(beta, *dst);
			}
		}
		let row = &mut tmp[..n];
		for k in 0..n {
			row[k] = *a.ptr.add(core + k * a.ld);
		}
		for j in 0..core {
			let bcol = core::slice::from_raw_parts(b.ptr.add(j * b.ld), n);
			unsafe {
				let mut sums = [_mm256_setzero_pd(); 4];
				for k in (0..core).step_by(16) {
					for v in 0..4 {
						sums[v] = _mm256_fmadd_pd(
							_mm256_loadu_pd(row.as_ptr().add(k + v * 4)),
							_mm256_loadu_pd(bcol.as_ptr().add(k + v * 4)),
							sums[v],
						);
					}
				}
				let x = _mm256_add_pd(
					_mm256_add_pd(sums[0], sums[1]),
					_mm256_add_pd(sums[2], sums[3]),
				);
				let parts: [f64; 4] = std::mem::transmute(x);
				*c.add(core + j * ldc) =
					row[core].mul_add(bcol[core], parts.iter().sum());
			}
		}
	}
}

// Offset consecutive panel starts by one cache line at full power-of-two depth.
const PANEL_GAP: usize = 8;

struct Panels<'a> {
	a: &'a mut [f64],
	b: &'a mut [f64],
}
std::thread_local! {
	static SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

// SAFETY: the dispatcher verifies AVX2/FMA, a supported square, row stride one,
// and positive column strides. The matrix views guarantee valid nonoverlapping
// output elements and disjoint inputs/output. Padding between columns can be
// uninitialized, so all accesses are confined to logical rows.
pub(super) unsafe fn multiply(
	n: usize,
	a: *const f64,
	lda: usize,
	b: *const f64,
	ldb: usize,
	c: *mut f64,
	ldc: usize,
) {
	let (h, kc, mc) = if n == 1024 {
		(512usize, 176, 96)
	} else {
		(256, 256, 64)
	};
	let pack_a = kc * mc + mc / 8 * PANEL_GAP;
	let pack_b = (kc * 6 + PANEL_GAP) * h.div_ceil(6);
	let products = 7 * (h.div_ceil(8) * h.div_ceil(6) * 48 + 8);
	SCRATCH.with_borrow_mut(|buf| {
		let required = products + pack_a + pack_b + 7;
		if buf.len() < required {
			buf.resize(required, 0.);
		}
		let aligned = buf.as_ptr().align_offset(64);
		let (ps, rest) = buf[aligned..].split_at_mut(products);
		let (pa, pb) = rest.split_at_mut(pack_a);
		let mut panels = Panels {
			a: pa,
			b: &mut pb[..pack_b],
		};
		let a = View { ptr: a, ld: lda };
		let b = View { ptr: b, ld: ldb };
		if n == 1024 {
			square::<512, 176, 96>(a, b, c, ldc, n, ps, &mut panels);
		} else {
			square::<256, 256, 64>(a, b, c, ldc, n, ps, &mut panels);
		}
	});
}

#[cfg(test)]
mod tests {
	#[test]
	fn padded_inputs_overwrite_all_corners_and_preserve_padding() {
		use crate::{Accum, MatMut, MatRef, Par};
		for n in [511, 513, 1024] {
			let (lda, ldb, ldc) = (n + 2, n + 3, n + 5);
			let mut a = vec![f64::NAN; lda * n];
			let mut b = vec![f64::NAN; ldb * n];
			let mut c = vec![f64::NAN; ldc * n];
			for j in 0..n {
				for i in 0..n {
					a[i + j * lda] = ((i * 7 + j * 23) % 67) as f64 - 31.;
					b[i + j * ldb] = if i == n - 1 - j {
						2.
					} else if i == n - 1 {
						1.
					} else {
						0.
					};
				}
			}
			crate::linalg::matmul::matmul(
				MatMut::from_column_major_slice_with_stride_mut(
					&mut c, n, n, ldc,
				),
				Accum::Replace,
				MatRef::from_column_major_slice_with_stride(&a, n, n, lda),
				MatRef::from_column_major_slice_with_stride(&b, n, n, ldb),
				1.,
				Par::Seq,
			);
			for j in 0..n {
				for i in 0..n {
					assert_eq!(
						c[i + j * ldc],
						2. * a[i + (n - 1 - j) * lda]
							+ if j > 0 { a[i + (n - 1) * lda] } else { 0. }
					);
				}
				assert!(
					c[j * ldc + n..(j + 1) * ldc].iter().all(|v| v.is_nan())
				);
			}
		}
	}
}
