#![allow(dead_code)]

use std::cmp::Ordering;

#[inline(always)]
pub fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
    // NaiveBubble::topk::<K>(values)
    // NaiveUnstable::topk::<K>(values)
    QuickSelect::topk::<K>(values)
    // QuickSelectIterative::topk::<K>(values)
    // MinHeap::topk::<K>(values)
}

pub trait TopK {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K];
}

struct NaiveBubble {}
impl TopK for NaiveBubble {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        debug_assert_ne!(values.len(), 0);

        // Initialize the results array.
        let mut results = [Entry::from((0usize, -1.0f32)); K];
        let last_idx = K - 1;

        for i in 0..values.len() {
            // Ignore all values that are smaller than the last entry in the list.
            let v = values[i];
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
        for i in K..values.len() {
            let v = values[i];
            if v <= min {
                continue;
            }

            // We found a value bigger than the smallest value we know.
            // Replace the smallest value with the new one.
            let mut new_min = f32::MAX;
            for j in 0..K {
                if results[j].value == min {
                    results[j] = (i, v).into();

                    // Prevent condition from triggering again.
                    min = f32::MAX;
                }

                // Determine the new smallest value.
                new_min = new_min.min(results[j].value);
            }

            min = new_min;
        }

        results
    }
}

struct QuickSelect {}
impl TopK for QuickSelect {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        let mut indexes: Vec<_> = (0..values.len()).collect();
        let _ = quickselect_max(values, &mut indexes, K);
        merge_into(&values, &indexes)
    }
}

struct QuickSelectIterative {}
impl TopK for QuickSelectIterative {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        debug_assert!(values.len() >= K);

        let buf_size = (2 * K).min(values.len());
        let mut vs = vec![f32::NAN; buf_size];
        vs.copy_from_slice(&values[0..buf_size]);
        let mut is: Vec<_> = (0..buf_size).collect();

        let _ = quickselect_max(&mut vs, &mut is, K);

        // Iteratively replace the second half of the buffer with
        // new data, then sort again. Repeat until the entire data is processed.
        for i in (buf_size..values.len()).step_by(K) {
            let count = K.min(values.len() - i);
            for j in 0..count {
                vs[K + j] = values[i + j];
                is[K + j] = i + j;
            }

            let _ = quickselect_max(&mut vs, &mut is, K);
        }

        merge_into(&vs, &is)
    }
}

fn merge_into<const K: usize>(vs: &[f32], is: &[usize]) -> [Entry; K] {
    debug_assert_eq!(vs.len(), is.len());
    let mut results = [Entry::new(0, 0f32); K];
    for i in 0..K {
        results[i] = Entry::new(is[i], vs[i]);
    }
    results
}

fn quickselect_max(data: &mut [f32], indexes: &mut [usize], k: usize) -> Entry {
    let mut left = 0;
    let mut right = data.len() - 1;

    loop {
        let pivot_index = partition_max(data, indexes, left, right);
        if pivot_index == k {
            return Entry::new(k, data[k]);
        } else if k < pivot_index {
            right = pivot_index - 1;
        } else {
            left = pivot_index + 1;
        }
    }
}

fn partition_max<T: PartialOrd>(
    data: &mut [T],
    indexes: &mut [usize],
    left: usize,
    right: usize,
) -> usize {
    let pivot = right;
    let mut i = left;

    for j in left..right {
        if data[j] >= data[pivot] {
            data.swap(i, j);
            indexes.swap(i, j);
            i += 1;
        }
    }

    data.swap(i, pivot);
    indexes.swap(i, pivot);
    i
}

pub struct MinHeap {}
impl TopK for MinHeap {
    fn topk<const K: usize>(values: &mut [f32]) -> [Entry; K] {
        let mut heap = std::collections::BinaryHeap::new();

        // Insert the first K elements into the heap
        for i in 0..K {
            heap.push(std::cmp::Reverse(Entry::new(i, values[i])));
        }

        // Insert the rest of the elements into the heap and pop off the smallest element
        for i in K..values.len() {
            heap.push(std::cmp::Reverse(Entry::new(i, values[i])));
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

#[derive(Debug, Copy, Clone)]
pub struct Entry {
    index: usize,
    value: f32,
}

impl Entry {
    pub fn new(index: usize, value: f32) -> Self {
        Self { index, value }
    }
}

impl Into<(usize, f32)> for Entry {
    fn into(self) -> (usize, f32) {
        (self.index, self.value)
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
        let mut indexes: Vec<_> = (0..arr.len()).collect();
        let k = 3;
        let kth_largest = quickselect_max(&mut arr, &mut indexes, k);
        assert_eq!(kth_largest, Entry::new(1, 3f32));

        println!("The {}-th smallest element is {}", k + 1, kth_largest.value);
    }

    #[test]
    fn minheap_works() {
        let mut arr = [30f32, 3f32, 1f32, 12f32, 2f32, 11f32];
        const K: usize = 4;
        let k_largest = MinHeap::topk::<K>(&mut arr);
        assert_eq!(k_largest[K - 1], Entry::new(1, 3f32));
    }
}
