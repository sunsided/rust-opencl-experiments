//! Provides memory chunks of arbitrary sizes.

use crate::chunks::AccessHint;
use abstractions::{NumDimensions, NumVectors};
use alloc_madvise::Memory;

/// A memory chunk whose size is specified at runtime.
#[derive(Debug)]
pub struct AnySizeMemoryChunk {
    num_vecs: usize,
    virt_num_vecs: usize,
    num_dims: usize,
    data: Memory,
}

impl AnySizeMemoryChunk {
    /// Initializes a new memory chunk.
    ///
    /// ## Arguments
    /// * `num_vectors`: The number of vectors to store.
    /// * `num_dimensions`: The number of dimensions per vector. Must be a multiple of 16.
    /// * `access_hint`: Specifies the intended access pattern.
    ///
    /// ## Panics
    /// Will panic if the number of vector dimensions is not a multiple of 16.
    pub fn new(
        num_vectors: NumVectors,
        num_dimensions: NumDimensions,
        access_hint: AccessHint,
    ) -> Self {
        assert_eq!(
            *num_dimensions % 16,
            0,
            "Number of dimensions must be a multiple of 16"
        );

        let num_elems = num_vectors * num_dimensions;
        let num_bytes = num_elems * std::mem::size_of::<f32>();
        let chunk = Memory::allocate(num_bytes, access_hint.is_sequential(), true)
            .expect("memory allocation failed");

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

    pub fn num_vecs(&self) -> NumVectors {
        NumVectors::from(self.virt_num_vecs)
    }

    pub fn num_dims(&self) -> NumDimensions {
        NumDimensions::from(self.num_dims)
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
        let mut chunk =
            Memory::allocate(num_bytes, false, false).expect("memory allocation failed");

        let src: &[f32] = self.data.as_ref();
        let dest: &mut [f32] = chunk.as_mut();
        dest[..src.len()].copy_from_slice(src);
        dest[src.len()..].copy_from_slice(src);

        self.data = chunk;
    }
}

impl AsRef<[f32]> for AnySizeMemoryChunk {
    fn as_ref(&self) -> &[f32] {
        let data: &[f32] = self.data.as_ref();
        &data[..self.num_dims * self.virt_num_vecs]
    }
}

impl AsMut<[f32]> for AnySizeMemoryChunk {
    fn as_mut(&mut self) -> &mut [f32] {
        let data: &mut [f32] = self.data.as_mut();
        &mut data[..self.num_dims * self.virt_num_vecs]
    }
}
