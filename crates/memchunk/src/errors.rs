//! Error types.

use abstractions::{LocalId, NumDimensions};

/// An error related to inserting vectors.
#[derive(Debug, thiserror::Error)]
pub enum InsertVectorError {
    /// An error with the same ID was already inserted.
    #[error("A vector with ID {0} was already inserted")]
    DuplicateId(LocalId),
    /// The dimensionality of the vector to insert does not align with the
    /// dimensionality of the registry.
    #[error("Dimensionality mismatch: expected vector of length {expected}, got vector of length {actual}")]
    DimensionalityMismatch {
        /// The expected number of dimensions.
        expected: NumDimensions,
        /// The number of dimensions of the provided vector.
        actual: NumDimensions,
    },
}
