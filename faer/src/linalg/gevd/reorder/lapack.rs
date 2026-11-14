//! Bounded QZ block-move validation; checker provenance is in lapack_cases.rs.
use super::*;
use dyn_stack::MemBuffer;
macro_rules! suite {
	($name:ident,$scalar:ty,$real:ty,$make:expr,$reorder:path) => {
		// INVARIANT: shared LAPACK helpers contain routines for other families.
		#[allow(dead_code, unused_imports)]
		mod $name {
			use super::*;
			type Scalar = $scalar;
			type Real = $real;
			fn scalar(re: Real, im: Real) -> Scalar {
				($make)(re, im)
			}
			#[allow(clippy::too_many_arguments)] // faer kernel shape, kept flat to match upstream
			fn reorder(
				s: MatMut<'_, Scalar>,
				t: MatMut<'_, Scalar>,
				q: Option<MatMut<'_, Scalar>>,
				z: Option<MatMut<'_, Scalar>>,
				from: usize,
				to: usize,
				par: Par,
				stack: &mut MemStack,
			) -> Result<usize, SchurReorderError> {
				$reorder(s, t, q, z, from, to, par, stack)
			}
			include!("../../tests/lapack_support.rs");
			include!("lapack_cases.rs");
		}
	};
}
suite!(f64, f64, f64, |re, _im| re, reorder_schur_real);
suite!(f32, f32, f32, |re, _im| re, reorder_schur_real);
suite!(c64, crate::c64, f64, crate::c64::new, reorder_schur_cplx);
suite!(c32, crate::c32, f32, crate::c32::new, reorder_schur_cplx);
