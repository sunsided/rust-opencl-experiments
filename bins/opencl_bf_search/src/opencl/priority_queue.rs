#![allow(dead_code)]

use ocl::core::DeviceInfoResult;
use ocl::{Context, Device, Program};

// TODO: Need to add indexes queue here!
// Requires the cl_khr_priority_queue extension
const EXTENSION_PRIORITY_QUEUE_SOURCE: &str = "
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
    }";

const MANUAL_PRIORITY_QUEUE_SOURCE: &str = "
    __kernel void topk(__global float* data, __global float* topk, __global uint* topk_idx, uint data_size, uint topk_size) {
        int gid = get_global_id(0);
        float cur_val = data[gid];
        uint cur_idx = gid;
        if (gid < topk_size) {
            topk[gid] = cur_val;
            topk_idx[gid] = cur_idx;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int i = topk_size / 2; i > 0; i >>= 1) {
            if (gid < i) {
                if (cur_val > topk[gid + i]) {
                    topk[gid + i] = cur_val;
                    topk_idx[gid + i] = cur_idx;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);
            }
        }
    }";

pub fn build_priority_queue_program(device: &Device, context: &Context) -> ocl::Result<Program> {
    // Check if the device supports the cl_khr_priority_queue extension
    match device.info(ocl::enums::DeviceInfo::Extensions).unwrap() {
        DeviceInfoResult::Extensions(extensions) => {
            if !extensions.contains("cl_khr_priority_queue") {
                eprintln!("👎 Device does not support the cl_khr_priority_queue extension.");

                Program::builder()
                    .devices(device)
                    .src(MANUAL_PRIORITY_QUEUE_SOURCE)
                    .build(context)
            } else {
                println!("🎉 Device support the cl_khr_priority_queue extension!");
                let _program = Program::builder()
                    .devices(device)
                    .src(EXTENSION_PRIORITY_QUEUE_SOURCE)
                    .build(context);

                todo!("Ensure Kernel has the correct inputs");
            }
        }
        _ => unreachable!(),
    }
}
