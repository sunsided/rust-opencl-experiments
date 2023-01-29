//! Reference implementations of matrix-vector dot products.

use crate::reference::unrolled_dots;
use crate::DotProduct;
use abstractions::{NumDimensions, NumVectors};
use rayon::prelude::*;

/// A naive matrix-vector dot product implementation using parallel computation of each vector.
///
/// Uses Rayon's [`IntoParallelIterator`] for the heavy lifting.
#[derive(Default)]
pub struct ReferenceDotProductParallel {}

/// A naive matrix-vector dot product implementation using parallel computation of each vector.
///
/// Uses Rayon's [`IntoParallelIterator`] for the heavy lifting.
#[derive(Default)]
pub struct ReferenceDotProductParallelUnrolled<const UNROLL_FACTOR: usize = 64> {}

impl DotProduct for ReferenceDotProductParallel {
    fn dot_product(
        &self,
        query: &[f32],
        data: &[f32],
        num_dims: NumDimensions,
        num_vecs: NumVectors,
        results: &mut [f32],
    ) {
        let num_vecs = num_vecs.get();
        let num_dims = num_dims.get();

        debug_assert_eq!(query.len(), num_dims, "query vector dimension mismatch");
        debug_assert_eq!(results.len(), num_vecs, "result vector dimension mismatch");
        debug_assert_eq!(
            data.len(),
            num_vecs * num_dims,
            "data buffer dimension mismatch"
        );

        results
            .par_iter_mut()
            .enumerate()
            .for_each(move |(v, result)| {
                let start_index = v * num_dims;

                let sum = query
                    .iter()
                    .zip(&data[start_index..])
                    .fold(0.0, |sum, (&q, &r)| sum + r * q);

                *result = sum;
            });
    }
}

impl<const UNROLL_FACTOR: usize> DotProduct for ReferenceDotProductParallelUnrolled<UNROLL_FACTOR> {
    fn dot_product(
        &self,
        query: &[f32],
        data: &[f32],
        num_dims: NumDimensions,
        num_vecs: NumVectors,
        results: &mut [f32],
    ) {
        let num_vecs = num_vecs.get();
        let num_dims = num_dims.get();

        debug_assert_eq!(query.len(), num_dims, "query vector dimension mismatch");
        debug_assert_eq!(results.len(), num_vecs, "result vector dimension mismatch");
        debug_assert_eq!(
            data.len(),
            num_vecs * num_dims,
            "data buffer dimension mismatch"
        );

        results
            .par_iter_mut()
            .enumerate()
            .for_each(move |(v, result)| {
                let start_index = v * num_dims;

                let mut sum = [0.0; UNROLL_FACTOR];
                for d in (0..num_dims).step_by(UNROLL_FACTOR) {
                    unrolled_dots(query, data, d, start_index + d, &mut sum);
                }

                *result = sum.iter().sum();
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        calculate_dot_products, generate_test_vectors, get_reference_results, rmse,
    };
    use approx::assert_relative_eq;

    #[test]
    fn parallel_works() {
        let chunk = generate_test_vectors();
        let expected = get_reference_results(0, &chunk);
        let results = calculate_dot_products::<ReferenceDotProductParallel>(0, &chunk);
        assert_relative_eq!(rmse(results, expected), 0.0, epsilon = 1e-4);
    }

    #[test]
    fn parallel_unrolled_works() {
        let chunk = generate_test_vectors();
        let expected = get_reference_results(0, &chunk);
        let results = calculate_dot_products::<ReferenceDotProductParallelUnrolled<64>>(0, &chunk);
        assert_relative_eq!(rmse(results, expected), 0.0, epsilon = 1e-4);
    }
}
