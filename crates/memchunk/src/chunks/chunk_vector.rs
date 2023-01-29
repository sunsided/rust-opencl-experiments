//! Tracking of memory chunks.

use crate::chunks::chunk_index::ChunkIndex;
use crate::chunks::fixed_size_memory_chunk::FixedSizeMemoryChunk;
use crate::chunks::AccessHint;
use std::ops::{Index, IndexMut};

/// Vector of chunk entries, indexed by [`ChunkIndex`].
#[derive(Debug, Default)]
pub(crate) struct ChunkVector {
    /// The intended access pattern.
    access_hint: AccessHint,
    /// Each index in this vector represents an allocated memory chunk.
    chunks: Vec<FixedSizeMemoryChunk>,
}

impl ChunkVector {
    /// Initializes the type with the specified access pattern
    /// and an initial allocated chunk.
    pub fn new(access_hint: AccessHint) -> Self {
        Self {
            access_hint,
            chunks: vec![FixedSizeMemoryChunk::allocate(access_hint)],
        }
    }

    /// Returns the number of chunks allocated.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Returns true if this vector contains no chunks.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Returns the last index in this vector, i.e. [`ChunkIndex::len`] reduced by one.
    #[inline(always)]
    pub fn last_index(&self) -> ChunkIndex {
        ChunkIndex::new(self.len() - 1)
    }

    /// Allocates another chunk and returns its index in this vector.
    pub fn allocate_next(&mut self) -> ChunkIndex {
        self.chunks
            .push(FixedSizeMemoryChunk::allocate(self.access_hint));
        self.last_index()
    }
}

impl Index<ChunkIndex> for ChunkVector {
    type Output = FixedSizeMemoryChunk;

    fn index(&self, index: ChunkIndex) -> &Self::Output {
        self.chunks.index(index.get())
    }
}

impl IndexMut<ChunkIndex> for ChunkVector {
    fn index_mut(&mut self, index: ChunkIndex) -> &mut Self::Output {
        self.chunks.index_mut(index.get())
    }
}
