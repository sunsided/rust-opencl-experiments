use abstractions::{LocalId, NumDimensions};

#[derive(Debug, thiserror::Error)]
pub enum InsertVectorError {
    #[error("A vector with ID {0} was already inserted")]
    DuplicateId(LocalId),
    #[error("Dimensionality mismatch: expected vector of length {expected}, got vector of length {actual}")]
    DimensionalityMismatch {
        expected: NumDimensions,
        actual: NumDimensions,
    },
}
