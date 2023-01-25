use crate::chunk_index::ChunkIndex;
use crate::chunk_vector::ChunkVector;
use crate::fixed_size_memory_chunk::{AccessHint, FixedSizeMemoryChunk};
use crate::index_vector_assignments::IndexVectorAssignments;
use crate::local_id_registry::IdRegistry;
use crate::InsertVectorError;
use abstractions::{LocalId, NumDimensions, NumVectors};
use std::ops::Deref;

pub trait ChunkManager {
    /// Creates a new chunk manager.
    ///
    /// ## Arguments
    /// * `dims` - The number of dimensions of the vectors to manager.
    /// * `access_hint` - The intended access pattern of the underlying memory chunks.
    fn new(dims: NumDimensions, access_hint: AccessHint) -> Self;

    /// Gets the total number of vectors that can be currently stored.
    /// This number does not indicate a limit and will change when chunks
    /// are allocated or deallocated.
    fn max_vecs(&self) -> NumVectors;

    /// Inserts a vector into this chunk.
    ///
    /// ## Arguments
    /// * `id` - The ID of the vector.
    /// * `vector` - The vector to insert.
    fn insert_vector<V: AsRef<[f32]>>(
        &mut self,
        id: LocalId,
        vector: V,
    ) -> Result<(), InsertVectorError>;
}

/// The [`BaseChunkManager`] provides functionality shared between more
/// specialized managers, such as the [`RowMajorChunkManager`](crate::RowMajorChunkManager) type.
pub(crate) struct BaseChunkManager {
    num_dims: NumDimensions,
    num_vecs_per_chunk: NumVectors,
    /// A registry of ID to chunk index.
    registry: IdRegistry<LocalId, ChunkIndex>,
    /// The allocated memory chunks.
    chunks: ChunkVector,
    /// The vector assignments per chunk. This vector always has the same number
    /// of elements as the `chunks` vector itself.
    assignments: IndexVectorAssignments,
}

impl BaseChunkManager {
    const NUM_FLOATS: usize = FixedSizeMemoryChunk::NUM_FLOATS;

    /// Creates a new chunk manager.
    ///
    /// ## Arguments
    /// * `dims` - The number of dimensions of the vectors to manager.
    /// * `access_hint` - The intended access pattern of the underlying memory chunks.
    pub fn new(dims: NumDimensions, access_hint: AccessHint) -> Self {
        let num_vecs_per_chunk = Self::NUM_FLOATS / dims.get();
        debug_assert!(dims.get() * num_vecs_per_chunk <= Self::NUM_FLOATS);

        Self {
            num_dims: dims,
            num_vecs_per_chunk: num_vecs_per_chunk.into(),
            registry: IdRegistry::new(),
            chunks: ChunkVector::new(access_hint),
            assignments: IndexVectorAssignments::new(num_vecs_per_chunk.into()),
        }
    }

    /// Gets the total number of vectors that can be currently stored.
    /// This number does not indicate a limit and will change when chunks
    /// are allocated or deallocated.
    #[inline]
    pub fn max_vecs(&self) -> NumVectors {
        NumVectors::from(self.chunks.len() * self.num_vecs_per_chunk)
    }

    /// Registers a vector with this manager and returns a reference
    /// to the chunk that should contain it, as well as the index at which to place it.
    ///
    /// ## Arguments
    /// * `id` - The ID of the vector to register.
    /// * `register_fn` - The function used to register the vector.
    pub fn register_vector<F, R>(
        &mut self,
        id: LocalId,
        register_fn: F,
    ) -> Result<R, InsertVectorError>
    where
        F: Fn(usize, &mut FixedSizeMemoryChunk) -> R,
    {
        if cfg!(not(feature = "optimistic")) && self.registry.contains_key(&id) {
            return Err(InsertVectorError::DuplicateId(id));
        }

        // Get a chunk to insert into.
        let selected_chunk = self.ensure_chunk_with_capacity();
        let chunk_idx = *selected_chunk;

        if let Some(previous_index) = self.registry.insert(id, chunk_idx) {
            // Revert the assignment we just did.
            self.registry.insert(id, previous_index);
            // TODO: We may just have allocated a new chunk - deallocate?
            debug_assert!(
                !selected_chunk.is_allocated(),
                "the previous allocation was not freed"
            );
            return Err(InsertVectorError::DuplicateId(id));
        }

        let chunk = &mut self.chunks[chunk_idx];
        let assignments = &mut self.assignments[chunk_idx];
        debug_assert!(
            assignments.len() < self.num_vecs_per_chunk.get(),
            "No space left for vectors; allocation failed?"
        );

        // The last slot is always empty unless the previous condition fails.
        let target_slot = assignments.len();
        let _previous_value = assignments.replace(target_slot, Some(id));
        debug_assert!(
            _previous_value,
            None,
            "Overwrote slot that was already in use"
        );

        Ok(register_fn(target_slot, chunk))
    }

    // TODO: Add unregister; should return Ok(Some(reassignment)) or Ok(None) if the chunk has less than two elements left.

    /// Gets the number of dimensions of each vector.
    pub const fn num_dims(&self) -> NumDimensions {
        self.num_dims
    }

    /// Gets the number of registered chunks.
    #[cfg(debug_assertions)]
    pub fn num_chunks(&self) -> usize {
        debug_assert_eq!(self.chunks.len(), self.assignments.len());
        self.chunks.len()
    }

    /// Gets the number of vectors in the specified chunk.
    #[cfg(debug_assertions)]
    fn chunk_vector_count(&self, idx: ChunkIndex) -> usize {
        self.assignments[idx].len()
    }

    /// Returns a chunk index with free capacity to insert to.
    /// If no free chunk is available, another chunk is allocated.
    fn ensure_chunk_with_capacity(&mut self) -> SelectedChunk {
        // We always insert into the last chunk. If the last chunk is full
        // we allocate another one.
        let chunk_idx = self.chunks.last_index();
        if !(self.assignments[chunk_idx].is_full()) {
            return SelectedChunk::Reused(chunk_idx);
        }

        let chunk_idx = self.chunks.allocate_next();
        let assignment_idx = self.assignments.allocate_next();
        debug_assert_eq!(chunk_idx, assignment_idx, "indexes diverged in registries");
        SelectedChunk::AllocatedNew(chunk_idx)
    }
}

enum SelectedChunk {
    Reused(ChunkIndex),
    AllocatedNew(ChunkIndex),
}

impl SelectedChunk {
    pub fn is_allocated(&self) -> bool {
        if let Self::AllocatedNew(_) = self {
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn get(&self) -> ChunkIndex {
        *(*self)
    }
}

impl Deref for SelectedChunk {
    type Target = ChunkIndex;

    fn deref(&self) -> &Self::Target {
        match self {
            SelectedChunk::Reused(x) => x,
            SelectedChunk::AllocatedNew(x) => x,
        }
    }
}

impl Into<ChunkIndex> for SelectedChunk {
    #[inline(always)]
    fn into(self) -> ChunkIndex {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_num_dims_works() {
        let manager = BaseChunkManager::new(NumDimensions::DIMS_384, AccessHint::Random);
        assert_eq!(manager.num_dims, NumDimensions::from(384u32));
    }

    #[test]
    fn default_max_vecs_works() {
        let manager = BaseChunkManager::new(NumDimensions::DIMS_384, AccessHint::Random);
        assert_eq!(
            manager.max_vecs(),
            manager.num_vecs_per_chunk,
            "Exactly one chunk was registered"
        );
    }

    #[test]
    fn insert_many_works() {
        let mut manager = BaseChunkManager::new(NumDimensions::DIMS_384, AccessHint::Random);
        let expected_vecs = if cfg!(feature = "power-of-two-chunks") {
            21845_usize
        } else {
            21728_usize
        };
        assert_eq!(manager.num_vecs_per_chunk, NumVectors::from(expected_vecs));
        for i in 0..manager.num_vecs_per_chunk.get() {
            let id = LocalId::try_from(i + 1).expect("invalid ID");
            {
                manager
                    .register_vector(id, |index, _chunk| {
                        assert_eq!(index, i, "unexpected index assignment")
                    })
                    .expect("registration failed");
            }
            assert_eq!(
                manager.chunk_vector_count(ChunkIndex::ZERO),
                i + 1,
                "unexpected length"
            );
        }

        assert_eq!(
            manager.num_chunks(),
            1,
            "no additional chunk allocation was expected"
        );

        // Adding another vector will allocate. The next vector to be inserted
        // is then placed at the zero-th index within that chunk.
        let next_id = LocalId::try_from(manager.num_vecs_per_chunk.get() + 1).expect("invalid ID");
        manager
            .register_vector(next_id, |index, _chunk| {
                assert_eq!(index, 0, "vector expected at index 0")
            })
            .expect("registration failed");

        assert_eq!(
            manager.num_chunks(),
            2,
            "additional chunk allocation was expected"
        );
    }
}
