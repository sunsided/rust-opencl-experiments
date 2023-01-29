use abstractions::{NumDimensions, NumVectors};

pub mod reference;
pub mod topk;

/// Performs matrix-vector dot product calculations.
pub trait DotProduct {
    /// Calculates the dot product of the specified `query` vector against the `data` array.
    ///
    /// ## Argumens
    /// * `query` - The query vector of length `D` (`num_dims`).
    /// * `data` - The data matrix of `N` (`num_vecs`) vectors of length `D` (total size `N×D`).
    /// * `num_dims` - The dimensionality `D` of each vector.
    /// * `num_vecs` - The number of vectors in the `data` array.
    /// * `results` The results buffer of length `M` (`num_vecs`) to fill.
    fn dot_product(
        &self,
        query: &[f32],
        data: &[f32],
        num_dims: NumDimensions,
        num_vecs: NumVectors,
        results: &mut [f32],
    );
}
