use ocl::core::DeviceInfoResult;
use ocl::{Context, Device, Program};

// TODO: Need to add indexes queue here!
// Requires the cl_khr_priority_queue extension
const EXTENSION_PRIORITY_QUEUE_SOURCE: &'static str = "
    __kernel void priority_queue_kernel(__global float* dot_product_results,
                                        __global int* priority_queue,
                                        int num_elements, int k) {
        __priority_queue(k) pq;
        for (int i = 0; i < num_elements; i++) {
            pq.push(dot_product_results[i], i);
        }
        for (int i = 0; i < k; i++) {
            priority_queue[i] = pq.pop().value;
        }
    }";

const MANUAL_PRIORITY_QUEUE_SOURCE: &'static str = "
    __kernel void priority_queue(__global float* dot_product_results,
                                __global int* priority_queue,
                                __global int* index_queue,
                                int num_elements, int k) {
        int i = get_global_id(0);
        if (i < k) {
            priority_queue[i] = dot_product_results[i];
            index_queue[i] = i;
        } else if (dot_product_results[i] > priority_queue[0]) {
            priority_queue[0] = dot_product_results[i];
            index_queue[0] = i;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        for (int j = 0; j < (int)log2((float)k); j++) {
            int parent = i >> (j + 1);
            if (parent < k) {
                int left = (i >> j) | (1 << j);
                if (left < k) {
                    int min_index = (priority_queue[left] < priority_queue[parent]) ? left : parent;
                    priority_queue[parent] = priority_queue[min_index];
                    index_queue[parent] = index_queue[min_index];
                }
            }
        }
    }";

pub fn build_priority_queue_program(
    device: &Device,
    context: &Context,
) -> Option<ocl::Result<Program>> {
    // Check if the device supports the cl_khr_priority_queue extension
    match device.info(ocl::enums::DeviceInfo::Extensions).unwrap() {
        DeviceInfoResult::Extensions(extensions) => {
            if !extensions.contains("cl_khr_priority_queue") {
                eprintln!("👎 Device does not support the cl_khr_priority_queue extension.");

                let program = Program::builder()
                    .devices(device)
                    .src(MANUAL_PRIORITY_QUEUE_SOURCE)
                    .build(context);

                Some(program)
            } else {
                println!("🎉 Device support the cl_khr_priority_queue extension!");
                let _program = Program::builder()
                    .devices(device)
                    .src(EXTENSION_PRIORITY_QUEUE_SOURCE)
                    .build(context);

                todo!("Ensure Kernel has the correct inputs");
            }
        }
        _ => None,
    }
}
