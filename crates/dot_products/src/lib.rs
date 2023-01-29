use abstractions::{NumDimensions, NumVectors};

pub mod reference;
pub mod reference_parallel;
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

/// Utility functions for running test.
#[cfg(test)]
pub mod test_utils {
    use crate::reference::ReferenceDotProduct;
    use crate::DotProduct;
    use abstractions::{NumDimensions, NumVectors, Vecgen};
    use approx::assert_relative_eq;
    use memchunk::chunks::any_size_memory_chunk::AnySizeMemoryChunk;
    use memchunk::chunks::AccessHint;

    /// Generates test vectors for testing.
    pub fn generate_test_vectors() -> AnySizeMemoryChunk {
        Vecgen::new_from_seed(0xdeadcafe).into_filled(AnySizeMemoryChunk::new(
            NumVectors::from(1024u32),
            NumDimensions::from(384u32),
            AccessHint::Random,
        ))
    }

    /// Calculates dot products using the reference implementation ([`reference::ReferenceDotProduct`]).
    /// This is a convenience wrapper for the [`calculate_dot_products`] function.
    ///
    /// ## Arguments
    /// * `index_id` - The ID of the index to use as a query.
    /// * `chunk` - The memory chunk to process.
    ///
    /// ## Returns
    /// A vector of reference results.
    pub fn get_reference_results(index_id: usize, chunk: &AnySizeMemoryChunk) -> Vec<f32> {
        calculate_dot_products::<ReferenceDotProduct>(index_id, chunk)
    }

    /// Calculates dot products using the specified implementation.
    ///
    /// ## Arguments
    /// * `index_id` - The ID of the index to use as a query.
    /// * `chunk` - The memory chunk to process.
    ///
    /// ## Generic Arguments
    /// * `D` - The [`DotProduct`] algorithm.
    ///
    /// ## Returns
    /// A vector of reference results.
    pub fn calculate_dot_products<D: DotProduct + Default>(
        index_id: usize,
        chunk: &AnySizeMemoryChunk,
    ) -> Vec<f32> {
        assert!(index_id < chunk.num_vecs(), "Invalid index ID specified");

        let mut results = vec![f32::NAN; chunk.num_vecs().into()];

        let algo = D::default();
        algo.dot_product(
            chunk.get_row_major_vec(index_id),
            chunk.as_ref(),
            chunk.num_dims(),
            chunk.num_vecs(),
            &mut results,
        );
        results
    }

    /// Calculates the root mean squared error between two vector slices.
    pub fn rmse<L: AsRef<[f32]>, R: AsRef<[f32]>>(actual: L, expected: R) -> f32 {
        let actual = actual.as_ref();
        let expected = expected.as_ref();
        assert_eq!(actual.len(), expected.len());
        let count = expected.len() as f32;
        let sum: f32 = actual
            .iter()
            .zip(expected)
            .map(|(&lhs, &rhs)| lhs - rhs)
            .map(|error| error * error)
            .sum();
        (sum / count).sqrt()
    }

    #[test]
    fn rmse_works() {
        assert_relative_eq!(
            rmse([1.0, 2.0, 3.0], [0.0, -1.0, 0.7]),
            (((1.0f32 - 0.0).powf(2.0) + (2.0f32 + 1.0).powf(2.0) + (3.0f32 - 0.7).powf(2.0))
                / 3.0)
                .sqrt(),
            epsilon = 1e-5
        );
    }
}
