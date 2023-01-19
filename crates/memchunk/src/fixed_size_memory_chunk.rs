use alloc_madvise::Memory;
use std::ops::{Deref, DerefMut};

pub type ChunkTypeF32 = [f32; 8388608];

#[derive(Debug)]
pub struct FixedSizeMemoryChunk {
    data: Memory,
}

impl FixedSizeMemoryChunk {
    /// The number of bytes in this memory chunk.
    pub const SIZE_BYTES: usize = 32 * 1024 * 1024;

    /// The number of [`f32`] elements in this memory chunk.
    pub const LENGTH: usize = Self::SIZE_BYTES / std::mem::size_of::<f32>();

    pub fn allocate() -> Self {
        let chunk =
            Memory::allocate(Self::SIZE_BYTES, false, true).expect("memory allocation failed");

        Self { data: chunk }
    }

    pub const fn len(&self) -> usize {
        Self::LENGTH
    }

    pub const fn is_empty(&self) -> bool {
        false
    }
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
