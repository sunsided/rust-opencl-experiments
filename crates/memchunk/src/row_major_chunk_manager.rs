use crate::alignments::Alignment;
use crate::chunk_manager::{BaseChunkManager, ChunkManager};
use crate::fixed_size_memory_chunk::AccessHint;
use crate::InsertVectorError;
use abstractions::{LocalId, NumDimensions, NumVectors};

/// A chunk manager that stores vectors in row-major order.
pub struct RowMajorChunkManager {
    manager: BaseChunkManager,
}

impl ChunkManager for RowMajorChunkManager {
    fn new(dims: NumDimensions, access_hint: AccessHint) -> Self {
        Self {
            manager: BaseChunkManager::new(dims, access_hint),
        }
    }

    fn max_vecs(&self) -> NumVectors {
        self.manager.max_vecs()
    }

    fn insert_vector<V: AsRef<[f32]>>(
        &mut self,
        id: LocalId,
        vector: V,
    ) -> Result<(), InsertVectorError> {
        let num_dims = self.manager.num_dims();

        let src = vector.as_ref();
        debug_assert_eq!(src.len(), num_dims.get());

        // TODO: This should fail when the vector was already registered.
        let mut assignment = self.manager.register_vector(id);

        // Since this is a row-major format, the n-th vector
        // starts at index n * dimensions.
        let idx = assignment.index * num_dims;

        let data: &mut [f32] = assignment.chunk.as_mut();
        let target = &mut data[idx..idx + num_dims.get()];
        debug_assert!(target.as_ptr().is_64byte_aligned());
        target.copy_from_slice(vector.as_ref());

        Ok(())
    }
}