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
            for d in 0..num_dims {
                let r = data[start_index + d];
                let q = query[d];
                sum += r * q;
            }

            *result = sum;
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
