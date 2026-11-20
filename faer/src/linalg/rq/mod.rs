//! the $RQ$ decomposition decomposes a matrix $A$ into the product
//! $$A = RQ$$
//! where $R$ is an upper trapezoidal matrix whose last column ends on the last
//! row (for $m \le n$ the right $m \times m$ block of $R$ is upper triangular
//! and the columns before it are zero) and $Q$ is a matrix with orthonormal rows
//! (represented using block householder sequences).
//!
//! the factorization is the $LQ$ decomposition of $A$ with its rows and its
//! columns reversed, $P_m A P_n = \tilde L \tilde Q$ (full factors: $\tilde L$ is
//! $m \times n$, $\tilde Q$ is $n \times n$), read back through the same
//! reversal: $A = (P_m \tilde L P_n)(P_n \tilde Q P_n)$. `factor::rq_in_place`
//! runs [`lq_in_place`](crate::linalg::lq::no_pivoting::factor::lq_in_place) on
//! `A.reverse_rows_and_cols_mut()`, so $R$ lands on and above the diagonal of $A$
//! that ends in its last row and column, and the householder vectors of $Q$
//! below it, the storage layout of LAPACK's `?gerqf`. `unitary` builds $Q$ (thin or
//! full), `reconstruct` rebuilds $A = RQ$, and `factor::copy_r` copies $R$ out.
pub mod no_pivoting;
