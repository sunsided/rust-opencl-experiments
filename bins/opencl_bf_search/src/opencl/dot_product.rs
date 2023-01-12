use ocl::builders::DeviceSpecifier;
use ocl::{Context, Program};

const DOT_PRODUCT_SOURCE: &'static str = "
    __kernel void dot_product(const __global float *matrix,
                             const __global float *vector,
                             __global float *result,
                             const unsigned int cols) {
        int row = get_global_id(0);
        float dot = 0.f;
        // TODO: get_global_size(1)?
        for (int i = 0; i < cols; i++) {
            dot += matrix[row*cols + i] * vector[i];
        }
        result[row] = dot;
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
