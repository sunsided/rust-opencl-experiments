//! Provides memory chunks of fixed sizes.
//!
//! See [`CHUNK_SIZE_BYTES`] for the actual size depending on the selected crate features.

use crate::chunks::AccessHint;
use abstractions::Alignment;
use alloc_madvise::Memory;
use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut};

/// The number of bytes in a memory chunk.
///
/// ## Features
/// - `power-of-two-chunks`: If enabled, the fixed buffer size will be `33_554_432` bytes (32 MiB);
///     if disabled, the buffer size is `33_374_208`. See below for reasoning.
///
/// ## Chunk size considerations
/// Typical vector lengths in question include 256, 384, 512, 768, 1024, 1536, 1792 and 2048,
/// the least common multiple of which is 43008.
/// Following this, the most efficient chunk size appears to be `33374208` bytes
/// (194 × 4 bytes × 43008) rather than `33554432` bytes (32 MiB).
pub const CHUNK_SIZE_BYTES: usize = if cfg!(not(feature = "power-of-two-chunks")) {
    33_374_208
} else {
    megabytes_to_bytes(32)
};

/// The number of [`f32`] values in a memory chunk.
pub const CHUNK_NUM_FLOATS: usize = CHUNK_SIZE_BYTES / std::mem::size_of::<f32>();

/// A slice of [`f32`] of exactly [`CHUNK_NUM_FLOATS`] elements.
pub type ChunkTypeF32 = [f32; CHUNK_NUM_FLOATS];

#[derive(Debug)]
pub struct FixedSizeMemoryChunk {
    data: Memory,
}

/// A memory chunk whose size is defined at compile time. See [`crate::chunks::fixed_size_memory_chunk::CHUNK_SIZE_BYTES`] for the
/// specifics.
impl FixedSizeMemoryChunk {
    /// The number of bytes in this memory chunk.
    pub const SIZE_BYTES: usize = CHUNK_SIZE_BYTES;

    /// The number of [`f32`] elements in this memory chunk.
    pub const NUM_FLOATS: usize = CHUNK_NUM_FLOATS;

    pub fn allocate(access_pattern: AccessHint) -> Self {
        let sequential = access_pattern.is_sequential();
        let chunk =
            Memory::allocate(Self::SIZE_BYTES, sequential, true).expect("memory allocation failed");
        debug_assert!((chunk.as_ptr() as *const f32).is_64byte_aligned());

        Self { data: chunk }
    }

    pub const fn len(&self) -> usize {
        Self::NUM_FLOATS
    }

    pub const fn is_empty(&self) -> bool {
        false
    }
}

trait DotProduct<const NUM_FLOATS: usize> {
    fn dot_product(coeffs: [f32; NUM_FLOATS]);
}

impl Deref for FixedSizeMemoryChunk {
    type Target = Memory;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for FixedSizeMemoryChunk {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl AsRef<[f32]> for FixedSizeMemoryChunk {
    fn as_ref(&self) -> &[f32] {
        self.data.as_ref()
    }
}

impl AsMut<[f32]> for FixedSizeMemoryChunk {
    fn as_mut(&mut self) -> &mut [f32] {
        self.data.as_mut()
    }
}

impl AsRef<ChunkTypeF32> for FixedSizeMemoryChunk {
    #[inline(always)]
    fn as_ref(&self) -> &ChunkTypeF32 {
        let data: &[f32] = self.data.as_ref();
        data.try_into().expect("invalid size")
    }
}

impl AsMut<ChunkTypeF32> for FixedSizeMemoryChunk {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut ChunkTypeF32 {
        let data: &mut [f32] = self.data.as_mut();
        data.try_into().expect("invalid size")
    }
}

impl Borrow<[f32]> for FixedSizeMemoryChunk {
    #[inline(always)]
    fn borrow(&self) -> &[f32] {
        self.as_ref()
    }
}

impl BorrowMut<[f32]> for FixedSizeMemoryChunk {
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut [f32] {
        self.as_mut()
    }
}

/// Converts from megabytes to bytes.
///
/// ## Arguments
/// * `mb` - The number of megabytes to represent as bytes.
const fn megabytes_to_bytes(mb: usize) -> usize {
    mb * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn megabytes_to_bytes_works() {
        assert_eq!(megabytes_to_bytes(1), 1_048_576);
    }
}
