//! Reference implementations of matrix-vector dot products.

use crate::DotProduct;
use abstractions::{NumDimensions, NumVectors};

/// A naive matrix-vector dot product implementation.
#[derive(Default)]
pub struct ReferenceDotProduct {}

/// Unrolled implementation of [`ReferenceDotProduct`].
///
/// ## Generic Arguments
/// * `UNROLL_FACTOR` - How many vector elements calculations to unroll.
#[derive(Default)]
pub struct ReferenceDotProductUnrolled<const UNROLL_FACTOR: usize = 64> {}

impl DotProduct for ReferenceDotProduct {
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

        for (v, result) in results.iter_mut().enumerate() {
            let start_index = v * num_dims;

            let sum = query
                .iter()
                .zip(&data[start_index..])
                .fold(0.0, |sum, (&q, &r)| sum + r * q);

            *result = sum;
        }
    }
}

impl<const UNROLL_FACTOR: usize> DotProduct for ReferenceDotProductUnrolled<UNROLL_FACTOR> {
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

        for (v, result) in results.iter_mut().enumerate() {
            let start_index = v * num_dims;

            let mut sum = [0.0; UNROLL_FACTOR];
            for d in (0..num_dims).step_by(UNROLL_FACTOR) {
                unrolled_dots(query, data, d, start_index + d, &mut sum);
            }

            *result = sum.iter().sum();
        }
    }
}

/// Helper function to perform an unrolled dot product of two vectors.
///
/// ## Arguments
/// * `query` - The query vector.
/// * `data` - The reference matrix.
/// * `query_start_index` - The starting index in the query vector.
/// * `data_start_index` - The starting index in the `data` matrix.
/// * `sum` - The array of sums to update.
#[inline(always)]
#[unroll::unroll_for_loops]
pub(crate) fn unrolled_dots<const UNROLL_FACTOR: usize>(
    query: &[f32],
    data: &[f32],
    query_start_index: usize,
    data_start_index: usize,
    sum: &mut [f32; UNROLL_FACTOR],
) {
    for unroll in 0..UNROLL_FACTOR {
        let r = data[data_start_index + unroll];
        let q = query[query_start_index + unroll];
        sum[unroll] += r * q;
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
    fn reference_works() {
        let reference = ReferenceDotProduct::default();

        let query = vec![1., 2., 3.];
        let data = vec![4., -5., 6., 4., -5., 6., 0., 0., 0., 1., 1., 1.];
        let mut results = vec![0., 0., 0., 0.];

        reference.dot_product(
            &query,
            &data,
            NumDimensions::from(3u32),
            NumVectors::from(4u32),
            &mut results,
        );

        assert_eq!(results, [12., 12., 0., 6.])
    }

    #[test]
    fn unrolled_works() {
        let chunk = generate_test_vectors();
        let expected = get_reference_results(0, &chunk);
        let results = calculate_dot_products::<ReferenceDotProductUnrolled<64>>(0, &chunk);
        assert_relative_eq!(rmse(results, expected), 0.0, epsilon = 1e-4);
    }
}
