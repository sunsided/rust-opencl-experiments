use ocl::builders::DeviceSpecifier;
use ocl::{Context, Program};

const DOT_PRODUCT_SOURCE: &'static str = "
    __kernel void dot_product(const __global float *matrix,
                             const __global float *vector,
                             __global float *result,
                             const int rows,
                             const int cols) {
        int row = get_global_id(0);
        float sum = 0;
        for (int i = 0; i < cols; i++) {
            sum += matrix[row * cols + i] * vector[i];
        }
        result[row] = sum;
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
