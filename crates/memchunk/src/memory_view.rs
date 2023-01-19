use crate::fixed_size_memory_chunk::FixedSizeMemoryChunk;

pub struct RowMajorMatrixView<const COLS: usize> {
    pub memory: FixedSizeMemoryChunk,
}

impl<const COLS: usize> RowMajorMatrixView<COLS> {
    /// The number of columns.
    pub const COLS: usize = COLS;

    /// The number of rows.
    pub const ROWS: usize = FixedSizeMemoryChunk::LENGTH / Self::COLS;

    /// The number of [`f32`] elements in this memory view.
    pub const LENGTH: usize = Self::COLS * Self::ROWS;

    /// The number of bytes in this memory chunk.
    pub const SIZE_BYTES: usize = Self::LENGTH * std::mem::size_of::<f32>();

    pub fn wrap(memory: FixedSizeMemoryChunk) -> Self {
        assert!(Self::SIZE_BYTES <= FixedSizeMemoryChunk::SIZE_BYTES);
        Self { memory }
    }

    pub const fn rows(&self) -> usize {
        Self::ROWS
    }

    pub const fn cols(&self) -> usize {
        Self::COLS
    }

    pub const fn len(&self) -> usize {
        Self::LENGTH
    }

    pub const fn is_empty(&self) -> bool {
        false
    }

    /*
    #[inline(always)]
    pub fn to_array(&self) -> &[f32] {
        let data: &[f32] = self.memory.as_ref();
        const LENGTH: usize = Self::LENGTH;
        let data: &[f32; Self::LENGTH] = data.try_into().expect("invalid dimensions");
        todo!()
    }
    */
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_size_memory_chunk::FixedSizeMemoryChunk;

    #[test]
    fn it_works() {
        let chunk = FixedSizeMemoryChunk::allocate();
        let view = RowMajorMatrixView::<384>::wrap(chunk);

        assert_eq!(RowMajorMatrixView::<384>::COLS, 384);
        assert_eq!(RowMajorMatrixView::<384>::ROWS, 21845);

        assert_eq!(view.cols(), 384);
        assert_eq!(view.rows(), 21845);
    }
}
