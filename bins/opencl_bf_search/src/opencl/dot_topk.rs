use ocl::builders::DeviceSpecifier;
use ocl::{Context, Program};

const DOT_PRODUCT_SOURCE: &'static str = "
    __kernel void dot_topk(__global float* matrix, __global float* vector, __global float* results, __global uint* indexes, uint top_k) {
        int gid = get_global_id(0);
        float dot = 0.f;
        for (int i = 0; i < get_global_size(1); i++) {
            dot += matrix[gid*get_global_size(1) + i] * vector[i];
        }
        results[gid] = dot;
        indexes[gid] = gid;
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int i = 0; i < top_k; i++) {
            if (gid == i) continue;
            float cur_val = results[gid];
            float cmp_val = results[i];
            uint cur_index = indexes[gid];
            uint cmp_index = indexes[i];
            uint prev_val = atomic_cmpxchg(results + i, cmp_val, cur_val);
            if (prev_val == cmp_val) {
                atomic_min(indexes + i, cur_index);
            }
        }
    }";

pub fn build_dot_topk_program<D: Into<DeviceSpecifier>>(
    device: D,
    context: &Context,
) -> ocl::Result<Program> {
    Program::builder()
        .devices(device)
        .src(DOT_PRODUCT_SOURCE)
        .build(context)
}
