use abstractions::{NumDimensions, NumVectors};

#[derive(Debug)]
pub struct MemoryChunk {
    num_vecs: usize,
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
            num_dims: *num_dimensions,
        }
    }

    pub fn search(&self, query: &[f32]) -> usize {
        let num_vecs = self.num_vecs();
        let num_dims = self.num_dims();

        let mut results = vec![0.0; num_vecs];

        const UNROLL_FACTOR: usize = 8;
        let num_dims_unrolled = num_dims / UNROLL_FACTOR;

        let data = &self.data;
        for v in 0..num_vecs {
            let start_index = v * num_dims;

            let mut sum = [0.0; UNROLL_FACTOR];
            for d in 0..num_dims_unrolled {
                let unroll = 0;
                let r = data[start_index + d + unroll];
                let q = query[d + unroll];
                sum[unroll] = r * q;

                let unroll = 1;
                let r = data[start_index + d + unroll];
                let q = query[d + unroll];
                sum[unroll] = r * q;

                let unroll = 2;
                let r = data[start_index + d + unroll];
                let q = query[d + unroll];
                sum[unroll] = r * q;

                let unroll = 3;
                let r = data[start_index + d + unroll];
                let q = query[d + unroll];
                sum[unroll] = r * q;

                let unroll = 4;
                let r = data[start_index + d + unroll];
                let q = query[d + unroll];
                sum[unroll] = r * q;

                let unroll = 5;
                let r = data[start_index + d + unroll];
                let q = query[d + unroll];
                sum[unroll] = r * q;

                let unroll = 6;
                let r = data[start_index + d + unroll];
                let q = query[d + unroll];
                sum[unroll] = r * q;

                let unroll = 7;
                let r = data[start_index + d + unroll];
                let q = query[d + unroll];
                sum[unroll] = r * q;
            }

            results[v] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
        }

        let mut max_score = 0.0;
        let mut max_idx = 0;
        for d in 0..num_vecs {
            let score = results[d];
            if score > max_score {
                max_score = score;
                max_idx = d;
            }
        }

        max_idx
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

    #[inline(always)]
    fn num_vecs(&self) -> usize {
        self.num_vecs
    }

    #[inline(always)]
    fn num_dims(&self) -> usize {
        self.num_dims
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
