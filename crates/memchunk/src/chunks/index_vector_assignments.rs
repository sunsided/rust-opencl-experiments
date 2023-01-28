use crate::chunks::chunk_index::ChunkIndex;
use abstractions::{LocalId, NumVectors};
use std::ops::{Index, IndexMut};

#[derive(Debug, Default)]
pub(crate) struct IndexVectorAssignments {
    num_vecs_per_chunk: NumVectors,
    assignments: Vec<IndexVectorAssignment>,
}

impl IndexVectorAssignments {
    pub fn new(num_vecs_per_chunk: NumVectors) -> Self {
        Self {
            num_vecs_per_chunk,
            assignments: vec![IndexVectorAssignment::new(num_vecs_per_chunk)],
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    #[inline(always)]
    pub fn last_index(&self) -> ChunkIndex {
        ChunkIndex::new(self.len() - 1)
    }

    pub fn allocate_next(&mut self) -> ChunkIndex {
        self.assignments
            .push(IndexVectorAssignment::new(self.num_vecs_per_chunk));
        self.last_index()
    }
}

impl Index<ChunkIndex> for IndexVectorAssignments {
    type Output = IndexVectorAssignment;

    fn index(&self, index: ChunkIndex) -> &Self::Output {
        self.assignments.index(index.get())
    }
}

impl IndexMut<ChunkIndex> for IndexVectorAssignments {
    fn index_mut(&mut self, index: ChunkIndex) -> &mut Self::Output {
        self.assignments.index_mut(index.get())
    }
}

/// Registers the used [`LocalId`] for each index.
#[derive(Debug, Default)]
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

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.count == self.assignments.len()
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
