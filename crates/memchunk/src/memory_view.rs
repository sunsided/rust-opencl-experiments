use crate::chunks::fixed_size_memory_chunk::FixedSizeMemoryChunk;

pub struct RowMajorMatrixView<const COLS: usize> {
    pub memory: FixedSizeMemoryChunk,
}

impl<const COLS: usize> RowMajorMatrixView<COLS> {
    /// The number of columns.
    pub const COLS: usize = COLS;

    /// The number of rows.
    pub const ROWS: usize = FixedSizeMemoryChunk::NUM_FLOATS / Self::COLS;

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
    use crate::chunks::AccessHint;

    #[test]
    fn creating_view_works() {
        let chunk = FixedSizeMemoryChunk::allocate(AccessHint::Sequential);
        let view = RowMajorMatrixView::<384>::wrap(chunk);

        let expected_vecs = if cfg!(feature = "power-of-two-chunks") {
            21845_usize
        } else {
            21728_usize
        };

        assert_eq!(RowMajorMatrixView::<384>::COLS, 384);
        assert_eq!(RowMajorMatrixView::<384>::ROWS, expected_vecs);

        assert_eq!(view.cols(), 384);
        assert_eq!(view.rows(), expected_vecs);
    }

    #[test]
    fn wait_what() {
        let vec = vec![1, 2, 3];
        let slice: &[i32] = vec.as_ref();
        assert_eq!(std::mem::size_of_val(slice), 12);
    }
}
