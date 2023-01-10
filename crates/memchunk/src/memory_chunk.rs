use crate::topk::{topk, Entry};
use abstractions::{NumDimensions, NumVectors};

#[derive(Debug)]
pub struct MemoryChunk {
    num_vecs: usize,
    virt_num_vecs: usize,
    num_dims: usize,
    data: Vec<f32>,
}

impl MemoryChunk {
    pub fn new_from(data: Vec<f32>, num_vectors: NumVectors) -> Self {
        let num_dims = data.len() / *num_vectors;
        assert_eq!(
            num_dims % 16,
            0,
            "Number of dimensions must be a multiple of 16"
        );
        assert_eq!(
            data.len(),
            *num_vectors * num_dims,
            "Data size does not match vector count and inferred dimensionality"
        );
        Self {
            data,
            num_vecs: *num_vectors,
            virt_num_vecs: *num_vectors,
            num_dims,
        }
    }

    pub fn new(num_vectors: NumVectors, num_dimensions: NumDimensions) -> Self {
        assert_eq!(
            *num_dimensions % 16,
            0,
            "Number of dimensions must be a multiple of 16"
        );
        Self {
            data: vec![0.0; num_vectors * num_dimensions],
            num_vecs: *num_vectors,
            virt_num_vecs: *num_vectors,
            num_dims: *num_dimensions,
        }
    }

    pub fn use_num_vecs(&mut self, num_vecs: NumVectors) {
        self.virt_num_vecs = match *num_vecs {
            0 => self.num_vecs,
            x => x,
        }
    }

    pub fn search_naive(&self, query: &[f32]) -> [Entry; 10] {
        let num_vecs = self.virt_num_vecs;
        let num_dims = self.num_dims;

        let mut results = vec![0.0; num_vecs];

        let data = &self.data;
        for v in 0..num_vecs {
            let start_index = v * num_dims;

            let mut sum = 0.0;
            for d in 0..num_dims {
                let r = data[start_index + d];
                let q = query[d];
                sum += r * q;
            }

            results[v] = sum;
        }

        let topk = topk(&mut results);
        topk
    }

    pub fn search_unrolled<const UNROLL_FACTOR: usize>(&self, query: &[f32]) -> [Entry; 10] {
        let num_vecs = self.virt_num_vecs;
        let num_dims = self.num_dims;

        let mut results = vec![0.0; num_vecs];

        let data = &self.data;
        for v in 0..num_vecs {
            let start_index = v * num_dims;

            let mut sum = [0.0; UNROLL_FACTOR];
            for d in (0..num_dims).step_by(UNROLL_FACTOR) {
                unrolled_dots::<UNROLL_FACTOR>(&query, &data, start_index + d, d, &mut sum);
            }

            results[v] = sum.iter().sum();
        }

        let topk = topk(&mut results);
        return topk;

        #[inline(always)]
        #[unroll::unroll_for_loops]
        fn unrolled_dots<const UNROLL_FACTOR: usize>(
            query: &[f32],
            data: &[f32],
            start_index: usize,
            d: usize,
            sum: &mut [f32; UNROLL_FACTOR],
        ) {
            for unroll in 0..UNROLL_FACTOR {
                let r = data[start_index + unroll];
                let q = query[d + unroll];
                sum[unroll] += r * q;
            }
        }
    }

    pub fn get_vec(&self, idx: usize) -> &[f32] {
        let start = idx * self.num_dims;
        let end = (idx + 1) * self.num_dims;
        debug_assert!(idx < self.data.len());
        &self.data[start..end]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl AsRef<[f32]> for MemoryChunk {
    fn as_ref(&self) -> &[f32] {
        &self.data
    }
}

impl AsMut<[f32]> for MemoryChunk {
    fn as_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}
