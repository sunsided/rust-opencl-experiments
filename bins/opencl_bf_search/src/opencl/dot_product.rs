use ocl::builders::DeviceSpecifier;
use ocl::{Context, Program};

const DOT_PRODUCT_SOURCE: &str = "
    __kernel void dot_product(const __global float *matrix,
                             const __global float *vector,
                             __global float *result,
                             const unsigned int rows,
                             const unsigned int cols) {
        float sum = 0.0f;
        int i = get_global_id(0); // row index
        for (int k=0; k<cols; k++)
        {
            sum += matrix[i + rows*k] * vector[k];
        }
        result[i] = sum;
    }
    ";

pub fn build_dot_product_program<D: Into<DeviceSpecifier>>(
    device: D,
    context: &Context,
) -> ocl::Result<Program> {
    Program::builder()
        .devices(device)
        .src(DOT_PRODUCT_SOURCE)
        .build(context)
}
