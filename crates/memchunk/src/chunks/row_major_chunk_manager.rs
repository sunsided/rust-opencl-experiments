use crate::chunks::chunk_manager::{BaseChunkManager, ChunkManager};
use crate::chunks::AccessHint;
use crate::InsertVectorError;
use abstractions::{Alignment, LocalId, NumDimensions, NumVectors};

/// A chunk manager that stores vectors in row-major order.
pub struct RowMajorChunkManager {
    manager: BaseChunkManager,
}

impl ChunkManager for RowMajorChunkManager {
    /// Creates a new manager capable of handling vectors of the specified dimensionality.
    ///
    /// ## Arguments
    /// * `dims` - The number of dimensions of each vector.
    /// * `access_hint` - The intended access pattern.
    fn new(dims: NumDimensions, access_hint: AccessHint) -> Self {
        Self {
            manager: BaseChunkManager::new(dims, access_hint),
        }
    }

    /// Gets the total number of vectors that can be currently stored.
    /// This number does not indicate a limit and will change when chunks
    /// are allocated or deallocated.
    fn max_vecs(&self) -> NumVectors {
        self.manager.max_vecs()
    }

    /// Inserts a vector.
    ///
    /// If all managed memory chunks are fully utilized, this will allocate
    /// a new chunk.
    ///
    /// ## Arguments
    /// * `id` - The ID of the vector.
    /// * `vector` - The vector to insert.
    fn insert_vector<V: AsRef<[f32]>>(
        &mut self,
        id: LocalId,
        vector: V,
    ) -> Result<(), InsertVectorError> {
        let num_dims = self.manager.num_dims();

        let src = vector.as_ref();
        if src.len() != self.manager.num_dims() {
            return Err(InsertVectorError::DimensionalityMismatch {
                actual: src.len().into(),
                expected: self.manager.num_dims(),
            });
        }

        self.manager.register_vector(id, |index, chunk| {
            // Since this is a row-major format, the n-th vector
            // starts at index n * dimensions.
            let idx = index * num_dims;

            let data: &mut [f32] = chunk.as_mut();
            let target = &mut data[idx..idx + num_dims.get()];
            debug_assert!(target.as_ptr().is_64byte_aligned());
            target.copy_from_slice(vector.as_ref());
        })?;

        Ok(())
    }
}
