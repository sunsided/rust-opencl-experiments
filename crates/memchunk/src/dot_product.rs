use abstractions::{NumDimensions, NumVectors};

pub trait DotProduct {
    fn dot_product(
        &self,
        query: &[f32],
        data: &[f32],
        num_dims: NumDimensions,
        num_vecs: NumVectors,
        results: &mut [f32],
    );
}

#[derive(Default)]
pub struct ReferenceDotProduct {}

#[derive(Default)]
pub struct ReferenceDotProductUnrolled<const UNROLL_FACTOR: usize = 8> {}

impl DotProduct for ReferenceDotProduct {
    fn dot_product(
        &self,
        query: &[f32],
        data: &[f32],
        num_dims: NumDimensions,
        num_vecs: NumVectors,
        results: &mut [f32],
    ) {
        let num_vecs = num_vecs.into_inner();
        let num_dims = num_dims.into_inner();

        debug_assert_eq!(query.len(), num_dims, "query vector dimension mismatch");
        debug_assert_eq!(results.len(), num_vecs, "result vector dimension mismatch");
        debug_assert_eq!(
            data.len(),
            num_vecs * num_dims,
            "data buffer dimension mismatch"
        );

        let data: &[f32] = data.as_ref();
        for (v, result) in results.iter_mut().enumerate() {
            let start_index = v * num_dims;

            let mut sum = 0.0;
            for (d, &q) in query.iter().enumerate() {
                let r = data[start_index + d];
                sum += r * q;
            }

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
        let num_vecs = num_vecs.into_inner();
        let num_dims = num_dims.into_inner();

        debug_assert_eq!(query.len(), num_dims, "query vector dimension mismatch");
        debug_assert_eq!(results.len(), num_vecs, "result vector dimension mismatch");
        debug_assert_eq!(
            data.len(),
            num_vecs * num_dims,
            "data buffer dimension mismatch"
        );

        let data: &[f32] = data.as_ref();
        for (v, result) in results.iter_mut().enumerate() {
            let start_index = v * num_dims;

            let mut sum = [0.0; UNROLL_FACTOR];
            for d in (0..num_dims).step_by(UNROLL_FACTOR) {
                Self::unrolled_dots(query, data, d, start_index + d, &mut sum);
            }

            *result = sum.iter().sum();
        }
    }
}

impl<const UNROLL_FACTOR: usize> ReferenceDotProductUnrolled<UNROLL_FACTOR> {
    #[inline(always)]
    #[unroll::unroll_for_loops]
    fn unrolled_dots(
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_works() {
        let reference = ReferenceDotProduct::default();

        let query = vec![1., 2., 3.];
        let data = vec![4., -5., 6., 4., -5., 6., 0., 0., 0., 1., 1., 1.];
        let mut results = vec![0., 0., 0., 0.];

        reference.dot_product(
            &query,
            &data,
            NumDimensions::from(3),
            NumVectors::from(4),
            &mut results,
        );

        assert_eq!(results, [12., 12., 0., 6.])
    }
}
