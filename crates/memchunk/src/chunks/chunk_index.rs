/// Index of a chunk in an internal registry.
#[derive(Debug, Default, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct ChunkIndex(usize);

impl ChunkIndex {
    /// The zero-th chunk index.
    pub const ZERO: ChunkIndex = ChunkIndex::new(0);

    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline(always)]
    pub const fn get(&self) -> usize {
        self.0
    }
}

impl From<usize> for ChunkIndex {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}
