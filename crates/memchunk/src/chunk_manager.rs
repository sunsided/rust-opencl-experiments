use crate::fixed_size_memory_chunk::{AccessHint, FixedSizeMemoryChunk};
use crate::InsertVectorError;
use abstractions::{LocalId, NumDimensions, NumVectors};

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

pub(crate) struct BaseChunkManager {
    num_dims: NumDimensions,
    num_vecs_per_chunk: NumVectors,
    chunks: Vec<FixedSizeMemoryChunk>,
}

impl BaseChunkManager {
    /// Creates a new chunk manager.
    ///
    /// ## Arguments
    /// * `dims` - The number of dimensions of the vectors to manager.
    /// * `access_hint` - The intended access pattern of the underlying memory chunks.
    pub fn new(dims: NumDimensions, access_hint: AccessHint) -> Self {
        let num_vecs_per_chunk = FixedSizeMemoryChunk::NUM_FLOATS / dims.get();
        debug_assert!(dims.get() * num_vecs_per_chunk <= FixedSizeMemoryChunk::NUM_FLOATS);

        Self {
            num_dims: dims,
            num_vecs_per_chunk: num_vecs_per_chunk.into(),
            chunks: vec![FixedSizeMemoryChunk::allocate(access_hint)],
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
    pub fn register_vector(&mut self, id: LocalId) -> ChunkAssignmentMut {
        // TODO: This should fail when the vector was already registered.
        todo!()
    }

    // TODO: Add unregister; should return Ok(Some(reassignment)) or Ok(None) if the chunk has less than two elements left.

    /// Gets the number of dimensions of each vector.
    pub const fn num_dims(&self) -> NumDimensions {
        self.num_dims
    }
}

pub(crate) struct ChunkAssignmentMut<'chunk> {
    pub index: usize,
    pub chunk: &'chunk mut FixedSizeMemoryChunk,
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
}
