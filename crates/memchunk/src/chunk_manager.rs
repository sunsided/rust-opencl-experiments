use crate::fixed_size_memory_chunk::{AccessHint, FixedSizeMemoryChunk};
use crate::InsertVectorError;
use abstractions::{LocalId, NumDimensions, NumVectors};
use std::ops::{Index, IndexMut};
use std::slice::SliceIndex;

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
    /// The allocated memory chunks.
    chunks: Vec<FixedSizeMemoryChunk>,
    /// The vector assignments per chunk. This vector always has the same number
    /// of elements as the `chunks` vector itself.
    assignments: Vec<IndexVectorAssignment>,
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
            chunks: vec![FixedSizeMemoryChunk::allocate(access_hint)],
            assignments: vec![IndexVectorAssignment::new(num_vecs_per_chunk.into())],
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
    pub fn register_vector(&mut self, id: LocalId) -> AssignmentMut {
        // TODO: This should fail when the vector was already registered.

        let chunk_idx = 0;
        // TODO: Use BTreeMap to look up chunk index from local ID.

        let chunk = &mut self.chunks[chunk_idx];
        let assignments = &mut self.assignments[chunk_idx];
        debug_assert!(
            assignments.len() < self.num_vecs_per_chunk.get(),
            "No space left for vectors; additional allocation required"
        );

        /// The last slot is always empty unless the previous condition fails.
        let target_slot = assignments.len();
        let _previous_value = assignments.replace(target_slot, Some(id));
        assert_eq!(
            _previous_value, None,
            "Overwrote slot that was already in use"
        );

        AssignmentMut {
            chunk,
            index: target_slot,
        }
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
    fn chunk_vector_count(&self, idx: usize) -> usize {
        self.assignments[idx].len()
    }
}

pub(crate) struct AssignmentMut<'chunk> {
    /// The vector's index in the chunk. Can be used to calculate the element or pointer offset
    /// at which to store the slice of data.
    pub index: usize,
    /// The chunk in which to store the vector data.
    pub chunk: &'chunk mut FixedSizeMemoryChunk,
}

/// Registers the used [`LocalId`] for each index.
pub(crate) struct IndexVectorAssignment {
    count: usize,
    assignments: Vec<Option<LocalId>>,
}

impl IndexVectorAssignment {
    /// Creates a new [`IndexVectorAssignment`] instance with the specified number of elements.
    pub fn new(num_vecs: NumVectors) -> Self {
        Self {
            count: 0,
            assignments: vec![None; num_vecs.get()],
        }
    }

    /// Gets the value at the specified index.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<LocalId> {
        self.assignments[index]
    }

    /// Replaces the value at the specified index and returns the old value contained.
    #[inline(always)]
    pub fn replace(&mut self, index: usize, value: Option<LocalId>) -> Option<LocalId> {
        let slot = &mut self.assignments[index];
        let previous = slot.clone();
        *slot = value;

        // If an empty v is overwritten with a value, we increase the count.
        // If an full slot is overwritten with an empty value, we decrease.
        // In all other cases the value does not count.
        if value.is_some() && previous.is_none() {
            self.count += 1;
        } else if value.is_none() && previous.is_some() {
            self.count -= 1;
        }

        previous
    }

    /// Gets the number of elements in this list.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Gets the number of elements in this list.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Index<usize> for IndexVectorAssignment {
    type Output = Option<LocalId>;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.assignments[index]
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
        assert_eq!(manager.num_vecs_per_chunk, NumVectors::from(21845_usize));
        for i in 0..manager.num_vecs_per_chunk.get() {
            let id = LocalId::try_from(i + 1).expect("invalid ID");
            {
                let assignment = manager.register_vector(id);
                assert_eq!(assignment.index, i, "unexpected index assignment");
            }
            assert_eq!(manager.chunk_vector_count(0), i + 1, "unexpected length");
        }

        assert_eq!(
            manager.num_chunks(),
            1,
            "no additional chunk allocation was expected"
        );

        // TODO: Adding another vector crashes.
        let assignment = manager.register_vector(
            LocalId::try_from(manager.num_vecs_per_chunk.get()).expect("invalid ID"),
        );
    }
}
