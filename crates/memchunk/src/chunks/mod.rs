//! Aligned buffer allocations and buffer management functionalities.

pub mod any_size_memory_chunk;
mod chunk_index;
mod chunk_manager;
mod chunk_vector;
pub mod fixed_size_memory_chunk;
mod index_vector_assignments;
mod row_major_chunk_manager;

pub use chunk_manager::ChunkManager;
pub use row_major_chunk_manager::RowMajorChunkManager;

/// Hints at the intended memory access pattern.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum AccessHint {
    /// Memory access will be mostly or entirely sequential.
    Sequential,
    /// Memory access follows no sequential pattern.
    Random,
}

impl AccessHint {
    /// Indicates whether this instance is the [`AccessHint::Sequential`] variant.
    #[inline(always)]
    pub const fn is_sequential(&self) -> bool {
        matches!(self, AccessHint::Sequential)
    }

    /// Indicates whether this instance is the [`AccessHint::Random`] variant.
    #[inline(always)]
    pub const fn is_random(&self) -> bool {
        matches!(self, AccessHint::Random)
    }
}

impl Default for AccessHint {
    fn default() -> Self {
        Self::Random
    }
}
