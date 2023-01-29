//! Provides memory chunks of arbitrary sizes.

use crate::chunks::AccessHint;
use abstractions::{NumDimensions, NumVectors};
use alloc_madvise::Memory;
use std::borrow::{Borrow, BorrowMut};

/// A memory chunk whose size is specified at runtime.
#[derive(Debug)]
pub struct AnySizeMemoryChunk {
    /// The number of vectors manageable by this chunk.
    num_vecs: usize,
    /// The artificially limited number of vectors.
    virt_num_vecs: usize,
    /// The number of dimensions for each vector.
    num_dims: usize,
    /// The underlying memory allocation.
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

    /// Artificially limits this chunk to the specified number of vectors assuming
    /// a row-major layout of memory.
    ///
    /// ## Warning
    /// Be careful when applying this operation to column-major chunks as obtained
    /// by the [`AnySizeMemoryChunk::as_transposed`] function. The changed layout
    /// may result in cutting off vectors.
    ///
    /// ## Arguments
    /// * `num_vecs` - The new maximum number of vectors to use.
    ///
    /// ## Notes
    /// This acts as a view into a smaller set of vectors to the outside.
    /// It does not, however, allocate or deallocate any memory.
    pub fn use_num_vecs(&mut self, num_vecs: NumVectors) {
        self.virt_num_vecs = match *num_vecs {
            0 => self.num_vecs,
            x => x.min(self.data.len()),
        }
    }

    /// Gets the vector at the specified index in the chunk
    /// assuming a row-major layout of the chunk.
    pub fn get_row_major_vec(&self, idx: usize) -> &[f32] {
        let start = idx * self.num_dims;
        let end = (idx + 1) * self.num_dims;
        debug_assert!(idx < self.data.len());
        let data: &[f32] = self.data.as_ref();
        &data[start..end]
    }

    /// Gets the total number of elements according to any
    /// limits set by [`AnySizeMemoryChunk::use_num_vecs`].
    pub fn len(&self) -> usize {
        self.num_dims * self.virt_num_vecs
    }

    /// Returns whether this chunk has zero length.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of vectors managed by this chunk,
    /// taking into account any limits set by [`AnySizeMemoryChunk::use_num_vecs`].
    pub fn num_vecs(&self) -> NumVectors {
        NumVectors::from(self.virt_num_vecs)
    }

    /// Returns the dimensionality of the vectors stored in this chunk.
    pub fn num_dims(&self) -> NumDimensions {
        NumDimensions::from(self.num_dims)
    }

    /// Returns a vector of transposed values.
    pub fn as_transposed_vec(&self) -> Vec<f32> {
        let mut vec = Vec::from(self.as_ref());
        transpose::transpose(self.as_ref(), &mut vec, self.num_dims, self.virt_num_vecs);
        vec
    }

    /// Allocates a new chunk of the same dimensions and fills it with a
    /// transposed view of the this instance's contents.
    ///
    /// In practice, this flips the chunk from row-major to column-major
    /// ordering and vice versa.
    pub fn as_transposed(&self, access_hint: AccessHint) -> AnySizeMemoryChunk {
        let mut transposed =
            AnySizeMemoryChunk::new(self.num_vecs.into(), self.num_dims.into(), access_hint);
        let src: &[f32] = self.data.as_ref();
        let dst: &mut [f32] = transposed.data.as_mut();
        transpose::transpose(src, dst, self.num_dims, self.num_vecs);
        debug_assert_eq!(
            src[0], dst[0],
            "The first element is the pivot and should not change under transposition"
        );
        debug_assert_eq!(
            src[1], dst[self.num_vecs],
            "The first row contains of the first elements of all vectors"
        );

        // Artificial limits don't work with odd numbers of vectors.
        // transposed.use_num_vecs(self.virt_num_vecs.into());
        transposed
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

impl Borrow<[f32]> for AnySizeMemoryChunk {
    #[inline(always)]
    fn borrow(&self) -> &[f32] {
        self.as_ref()
    }
}

impl BorrowMut<[f32]> for AnySizeMemoryChunk {
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut [f32] {
        self.as_mut()
    }
}
