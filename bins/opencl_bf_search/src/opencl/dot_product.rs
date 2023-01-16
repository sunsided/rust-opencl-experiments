use ocl::builders::DeviceSpecifier;
use ocl::{Context, Program};

const DOT_PRODUCT_SOURCE: &str = "
    __kernel void dot_product(const __global float4 *matrix,
                             const __global float4 *vector,
                             __global float *result,
                             const unsigned int rows,
                             const unsigned int cols) {
        int id = get_global_id(0);
        float4 dot_product = (float4)(0);
        for (int i = 0; i < cols/4; i++) {
            dot_product += matrix[id*(cols/4) + i] * vector[i];
        }
        result[id] = dot(dot_product, 1);
    }";

pub fn build_dot_product_program<D: Into<DeviceSpecifier>>(
    device: D,
    context: &Context,
) -> ocl::Result<Program> {
    Program::builder()
        .devices(device)
        .src(DOT_PRODUCT_SOURCE)
        .build(context)
}
