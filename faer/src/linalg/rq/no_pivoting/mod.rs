//! the $RQ$ decomposition decomposes a matrix $A$ into the product
//! $$A = RQ$$
//! where $R$ is an upper trapezoidal matrix and $Q$ is a matrix with orthonormal
//! rows (represented as a block householder sequence).
#![allow(missing_docs)]
pub mod factor;
pub mod reconstruct;
pub mod unitary;
