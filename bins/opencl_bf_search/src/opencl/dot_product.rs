use ocl::builders::DeviceSpecifier;
use ocl::{Context, Program};

const DOT_PRODUCT_SOURCE: &str = "
    __kernel void dot_product(const __global float4 *matrix,
                             const __global float4 *vector,
                             __global float *result,
                             const unsigned int rows,
                             const unsigned int cols) {
        int id = get_global_id(0);
        int lid = get_local_id(0);

        unsigned int local_cols = cols / 4;

        // Vector length must be less than 2048 elements.
        __local float4 local_vec[512];
        event_t copy_event = async_work_group_copy(local_vec, vector, local_cols, 0);
        wait_group_events(1, &copy_event);

        float4 dot_product = (float4)(0);
        for (int i = 0; i < local_cols; i++) {
            dot_product += matrix[i + id*local_cols] * local_vec[i];
        }

        result[id] = dot(dot_product, 1);
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
