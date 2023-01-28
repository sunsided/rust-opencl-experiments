//! Reference implementations of top-K retrieval from sequences of values.

#![allow(dead_code)]

use std::cmp::Ordering;

/// Returns the top-K largest elements of the values.
///
/// This is a convenience method; for finer grained control, please
/// have a look at the [`TopK`] trait and its implementations.
///
/// ## Arguments
/// * `values` - The values to scan.
///
/// ## Generic Arguments
/// * `K` - The number of elements to return.
///
/// ## Returns
/// An array of length `K` consisting of the `K` largest values.
/// This resulting values are not guaranteed to be sorted.
#[inline(always)]
pub fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
    // NaiveBubble::topk::<K>(values)
    // NaiveUnstable::topk::<K>(values)
    QuickSelect::topk::<K>(values)
    // QuickSelectIterative::topk::<K>(values)
    // MinHeap::topk::<K>(values)
}

/// Retrieves the top-K elements of a list of values.
pub trait TopK {
    /// Returns the top-K largest values.
    ///
    /// ## Arguments
    /// * `values` - The values to select from.
    ///
    /// ## Generic Arguments
    /// * `K` - The number of elements to return.
    ///
    /// ## Returns
    /// The `K` largest elements and their indices.
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K];
}

/// A naive top-K reference implementation using bubble sort.
struct NaiveBubble {}
impl TopK for NaiveBubble {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        debug_assert_ne!(values.len(), 0);

        // Initialize the results array.
        let mut results = [Entry::from((0usize, -1.0f32)); K];
        let last_idx = K - 1;

        for (i, &v) in values.iter().enumerate() {
            // Ignore all values that are smaller than the last entry in the list.
            if v <= results[last_idx].value {
                continue;
            }

            // Insert the value into the last position.
            results[last_idx] = (i, v).into();

            // Bubble up.
            for j in (0..last_idx).rev() {
                if v > results[j].value {
                    results.swap(j, j + 1);
                } else {
                    break;
                }
            }
        }

        results
    }
}

/// A naive top-K reference implementation using an unstable insertion sort-like algorithm.
struct NaiveUnstable {}
impl TopK for NaiveUnstable {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        debug_assert_ne!(values.len(), 0);

        // Initialize the results array.
        let mut results = [Entry::from((0usize, 0.0f32)); K];
        let mut min = f32::MAX;
        for i in 0..K {
            results[i] = (i, values[i]).into();
            min = min.min(values[i]);
        }

        // Scan for values bigger than our current maximum.
        for (i, &v) in values.iter().enumerate().take(K) {
            if v <= min {
                continue;
            }

            // We found a value bigger than the smallest value we know.
            // Replace the smallest value with the new one.
            let mut new_min = f32::MAX;
            for result in results.iter_mut().take(K) {
                if result.value == min {
                    *result = (i, v).into();

                    // Prevent condition from triggering again.
                    min = f32::MAX;
                }

                // Determine the new smallest value.
                new_min = new_min.min(result.value);
            }

            min = new_min;
        }

        results
    }
}

/// A top-K implementation using the quick-select algorithm.
///
/// This implementation partially sorts the entire buffer and returns the top-K largest elements.
struct QuickSelect {}
impl TopK for QuickSelect {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        let mut indices: Vec<_> = (0..values.len()).collect();
        let _ = quickselect_max::<K>(values, &mut indices);
        merge_into(values, &indices)
    }
}

/// A top-K implementation using an iterative quick-select algorithm.
///
/// This implementation keeps a working set of `2×K` elements,
/// performs quick-select and iteratively replaces the last `K` elements
/// with new values as it scans the input buffer.
///
/// After all elements are processed, the first `K` elements of the working
/// set are returned.
struct QuickSelectIterative {}
impl TopK for QuickSelectIterative {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        debug_assert!(values.len() >= K);

        let buf_size = (2 * K).min(values.len());
        let mut vectors = vec![f32::NAN; buf_size];
        vectors.copy_from_slice(&values[0..buf_size]);
        let mut indices: Vec<_> = (0..buf_size).collect();

        let _ = quickselect_max::<K>(&mut vectors, &mut indices);

        // Iteratively replace the second half of the buffer with
        // new data, then sort again. Repeat until the entire data is processed.
        for i in (buf_size..values.len()).step_by(K) {
            let count = K.min(values.len() - i);
            for j in 0..count {
                vectors[K + j] = values[i + j];
                indices[K + j] = i + j;
            }

            let _ = quickselect_max::<K>(&mut vectors, &mut indices);
        }

        merge_into(&vectors, &indices)
    }
}

/// Merges a data slice and an index slice into an array of elements.
///
/// ## Arguments
/// * `vectors` - The values to retain. Must have at least `K` elements.
/// * `indices` - The indices to retain. Must have the same length as `vectors`.
///
/// ## Generic Arguments
/// * `K` - The number of elements to return.
fn merge_into<const K: usize>(vectors: &[f32], indices: &[usize]) -> [Entry; K] {
    debug_assert!(vectors.len() >= K);
    debug_assert_eq!(vectors.len(), indices.len());
    let mut results = [Entry::new(0, 0f32); K];
    for i in 0..K {
        results[i] = Entry::new(indices[i], vectors[i]);
    }
    results
}

/// Performs quick-select on the specified values and indices.
///
/// ## Arguments
/// * `data` - The data to select from.
/// * `indices` - The indices for each entry in `data`.
/// * `k` - The number of elements to return.
///
/// ## Generic Arguments
/// * `K` - The number of elements to select.
///
/// ## Return
/// The k-th largest entry in `data`.
fn quickselect_max<const K: usize>(data: &mut [f32], indices: &mut [usize]) -> Entry {
    debug_assert_eq!(data.len(), indices.len());

    let mut left = 0;
    let mut right = data.len() - 1;

    loop {
        match partition_max(data, indices, left, right) {
            pivot_index if K < pivot_index => right = pivot_index - 1,
            pivot_index if K > pivot_index => left = pivot_index + 1,
            k => return Entry::new(k, data[k]),
        };
    }
}

/// The partition function for [`quickselect_max`].
///
/// ## Arguments
/// * `data` - The data to partition.
/// * `indices` - The indices to partition along with the `data`.
/// * `left` - The left index.
/// * `right` - The right index.
///
/// ## Returns
/// The new pivot index.
fn partition_max<T: PartialOrd>(
    data: &mut [T],
    indices: &mut [usize],
    left: usize,
    right: usize,
) -> usize {
    let pivot = right;
    let mut i = left;

    for j in left..right {
        if data[j] >= data[pivot] {
            data.swap(i, j);
            indices.swap(i, j);
            i += 1;
        }
    }

    data.swap(i, pivot);
    indices.swap(i, pivot);
    i
}

/// A top-K implementation using a min-heap.
pub struct MinHeap {}
impl TopK for MinHeap {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        let mut heap = std::collections::BinaryHeap::new();

        // Insert the first K elements into the heap
        for (i, &v) in values.iter().enumerate().take(K) {
            heap.push(std::cmp::Reverse(Entry::new(i, v)));
        }

        // Insert the rest of the elements into the heap and pop off the smallest element
        for (i, &v) in values.iter().enumerate().skip(K) {
            heap.push(std::cmp::Reverse(Entry::new(i, v)));
            heap.pop();
        }

        // Extract the top K elements from the heap
        let mut result = vec![Entry::new(0, 0.0); K];
        for i in (0..K).rev() {
            result[i] = heap.pop().unwrap().0;
        }

        result
            .try_into()
            .expect("the vector is appropriately sized")
    }
}

/// The result entry of a quick-select operation.
#[derive(Debug, Copy, Clone)]
pub struct Entry {
    /// The index of the k-th largest element.
    pub index: usize,
    /// The k-th largest element.
    pub value: f32,
}

impl Entry {
    /// Creates a new instance of the [`Entry`] struct.
    ///
    /// ## Arguments
    /// * `index` - The index of the element.
    /// * `value` - The value of the element.
    pub fn new(index: usize, value: f32) -> Self {
        Self { index, value }
    }
}

impl From<Entry> for (usize, f32) {
    fn from(value: Entry) -> Self {
        (value.index, value.value)
    }
}

impl From<(usize, f32)> for Entry {
    fn from(value: (usize, f32)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl PartialEq<Self> for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .partial_cmp(&other.value)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use crate::topk::{quickselect_max, Entry, MinHeap, TopK};

    #[test]
    fn quickselect_works() {
        let mut arr = [30f32, 3f32, 1f32, 12f32, 2f32, 11f32];
        let mut indices: Vec<_> = (0..arr.len()).collect();
        const K: usize = 3;
        let kth_largest = quickselect_max::<K>(&mut arr, &mut indices);
        assert_eq!(kth_largest, Entry::new(1, 3f32));

        println!("The {}-th smallest element is {}", K + 1, kth_largest.value);
    }

    #[test]
    fn minheap_works() {
        let mut arr = [30f32, 3f32, 1f32, 12f32, 2f32, 11f32];
        const K: usize = 4;
        let k_largest = MinHeap::topk::<K>(&mut arr);
        assert_eq!(k_largest[K - 1], Entry::new(1, 3f32));
    }
}
