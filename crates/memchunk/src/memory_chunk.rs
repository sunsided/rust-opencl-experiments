use crate::topk::{topk, Entry};
use abstractions::{NumDimensions, NumVectors};
use alloc_madvise::Memory;

#[derive(Debug)]
pub struct MemoryChunk {
    num_vecs: usize,
    virt_num_vecs: usize,
    num_dims: usize,
    data: Memory,
}

impl MemoryChunk {
    pub fn new(num_vectors: NumVectors, num_dimensions: NumDimensions) -> Self {
        assert_eq!(
            *num_dimensions % 16,
            0,
            "Number of dimensions must be a multiple of 16"
        );

        let num_elems = num_vectors * num_dimensions;
        let num_bytes = num_elems * std::mem::size_of::<f32>();
        let chunk = Memory::allocate(num_bytes, true, true).expect("memory allocation failed");

        Self {
            data: chunk,
            num_vecs: *num_vectors,
            virt_num_vecs: *num_vectors,
            num_dims: *num_dimensions,
        }
    }

    pub fn use_num_vecs(&mut self, num_vecs: NumVectors) {
        self.virt_num_vecs = match *num_vecs {
            0 => self.num_vecs,
            x => x.min(self.data.len()),
        }
    }

    pub fn search_reference(&self, query: &[f32]) -> Vec<f32> {
        let num_vecs = self.virt_num_vecs;
        let num_dims = self.num_dims;

        let mut results = vec![0.0; num_vecs];

        let data: &[f32] = self.data.as_ref();
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

        results
    }

    pub fn search_naive(&self, query: &[f32]) -> [Entry; 10] {
        let num_vecs = self.virt_num_vecs;
        let num_dims = self.num_dims;

        let mut results = vec![0.0; num_vecs];

        let data: &[f32] = self.data.as_ref();
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

        topk(&mut results)
    }

    pub fn search_unrolled<const UNROLL_FACTOR: usize>(&self, query: &[f32]) -> [Entry; 10] {
        let num_vecs = self.virt_num_vecs;
        let num_dims = self.num_dims;

        let mut results = vec![0.0; num_vecs];

        let data: &[f32] = self.data.as_ref();
        for (v, result) in results.iter_mut().enumerate() {
            let start_index = v * num_dims;

            let mut sum = [0.0; UNROLL_FACTOR];
            for d in (0..num_dims).step_by(UNROLL_FACTOR) {
                unrolled_dots::<UNROLL_FACTOR>(query, data, start_index + d, d, &mut sum);
            }

            *result = sum.iter().sum();
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
        let data: &[f32] = self.data.as_ref();
        &data[start..end]
    }

    pub fn len(&self) -> usize {
        self.num_dims * self.virt_num_vecs
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_vecs(&self) -> usize {
        self.virt_num_vecs
    }

    pub fn num_dims(&self) -> usize {
        self.num_dims
    }

    pub fn as_transposed(&self) -> Vec<f32> {
        let mut vec = Vec::from(self.as_ref());
        transpose::transpose(self.as_ref(), &mut vec, self.num_dims, self.virt_num_vecs);
        vec
    }

    pub fn double(&mut self) {
        self.num_vecs *= 2;
        self.virt_num_vecs *= 2;

        let num_elems = self.num_dims * self.num_vecs;
        let num_bytes = num_elems * std::mem::size_of::<f32>();
        let mut chunk = Memory::allocate(num_bytes, true, true).expect("memory allocation failed");

        let src: &[f32] = self.data.as_ref();
        let dest: &mut [f32] = chunk.as_mut();
        dest[..src.len()].copy_from_slice(src);
        dest[src.len()..].copy_from_slice(src);

        self.data = chunk;
    }
}

impl AsRef<[f32]> for MemoryChunk {
    fn as_ref(&self) -> &[f32] {
        let data: &[f32] = self.data.as_ref();
        &data[..self.num_dims * self.virt_num_vecs]
    }
}

impl AsMut<[f32]> for MemoryChunk {
    fn as_mut(&mut self) -> &mut [f32] {
        let data: &mut [f32] = self.data.as_mut();
        &mut data[..self.num_dims * self.virt_num_vecs]
    }
}
