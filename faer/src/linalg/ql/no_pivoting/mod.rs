//! the $QL$ decomposition decomposes a matrix $A$ into the product
//! $$A = QL$$
//! where $Q$ is a unitary matrix (represented as a block householder sequence)
//! and $L$ is a lower trapezoidal matrix.
#![allow(missing_docs)]
pub mod factor;
pub mod reconstruct;
pub mod unitary;
