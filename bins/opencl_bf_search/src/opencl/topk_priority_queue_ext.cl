// Requires the cl_khr_priority_queue extension

// TODO: Need to add indexes queue here!
__kernel void topk(__global float* dot_product_results,
                                    __global int* priority_queue,
                                    int num_elements, int k) {
    __priority_queue(k) pq;
    for (int i = 0; i < num_elements; i++) {
        pq.push(dot_product_results[i], i);
    }
    for (int i = 0; i < k; i++) {
        priority_queue[i] = pq.pop().value;
    }
}
