use ocl::builders::DeviceSpecifier;
use ocl::{Context, Program};

const DOT_PRODUCT_SOURCE: &str = "
    __kernel void dot_product_topk(__global float* matrix, __global float* vector, __global float* result, __global float* topk, __global uint* topk_idx, uint matrix_rows, uint matrix_cols, uint topk_size) {
        int gid = get_global_id(0);
        float dot_product = 0;
        for (int i = 0; i < matrix_cols; i++) {
            dot_product += matrix[gid * matrix_cols + i] * vector[i];
        }
        result[gid] = dot_product;

        // manual priority queue
        if (gid < topk_size) {
            topk[gid] = dot_product;
            topk_idx[gid] = gid;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int i = topk_size / 2; i > 0; i >>= 1) {
            if (gid < i) {
                if (dot_product > topk[gid + i]) {
                    topk[gid + i] = dot_product;
                    topk_idx[gid + i] = gid;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);
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
