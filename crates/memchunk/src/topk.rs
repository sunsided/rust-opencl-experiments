#![allow(dead_code)]

#[inline(always)]
pub fn topk<const K: usize>(values: &[f32]) -> [(usize, f32); K] {
    topk_naive_bubble::<K>(values)
}

pub fn topk_naive_bubble<const K: usize>(values: &[f32]) -> [(usize, f32); K] {
    debug_assert_ne!(values.len(), 0);

    // Initialize the results array.
    let mut results = [(0usize, -1.0f32); K];
    let last_idx = K - 1;

    for i in 0..values.len() {
        // Ignore all values that are smaller than the last entry in the list.
        let v = values[i];
        if v <= results[last_idx].1 {
            continue;
        }

        // Insert the value into the last position.
        results[last_idx] = (i, v);

        // Bubble up.
        for j in (0..last_idx).rev() {
            if v > results[j].1 {
                results.swap(j, j + 1);
            } else {
                break;
            }
        }
    }

    results
}

pub fn topk_naive_unstable<const K: usize>(values: &[f32]) -> [(usize, f32); K] {
    debug_assert_ne!(values.len(), 0);

    // Initialize the results array.
    let mut results = [(0usize, 0.0f32); K];
    let mut min = f32::MAX;
    for i in 0..K {
        results[i] = (i, values[i]);
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
            if results[j].1 == min {
                results[j] = (i, v);

                // Prevent condition from triggering again.
                min = f32::MAX;
            }

            // Determine the new smallest value.
            new_min = new_min.min(results[j].1);
        }

        min = new_min;
    }

    results
}
