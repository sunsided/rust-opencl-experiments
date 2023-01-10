mod vec_traits;
mod vecgen;

use memchunk::MemoryChunk;
use ocl::core::DeviceInfoResult;
use ocl::{Buffer, Context, Device, Kernel, Platform, ProQue};
use std::path::PathBuf;
use std::time::Instant;
use vecdb::VecDb;

#[tokio::main]
async fn main() {
    #[cfg(debug_assertions)]
    const K: usize = 10_000;
    #[cfg(not(debug_assertions))]
    const K: usize = 0;
    let chunk = load_vectors::<K>().await;
    let first_vec = Vec::from(chunk.get_vec(0));

    let start = Instant::now();
    let _reference = chunk.search_naive(&first_vec);
    println!(
        "Duration processing {vecs} vectors on CPU: {duration} s",
        vecs = chunk.num_vecs(),
        duration = (Instant::now() - start).as_secs_f32()
    );

    // Default setup.
    let platform = Platform::first().unwrap();
    let device = Device::first(platform).unwrap();
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()
        .unwrap();

    // Check if the device supports the cl_khr_priority_queue extension
    if let DeviceInfoResult::Extensions(extensions) =
        device.info(ocl::enums::DeviceInfo::Extensions).unwrap()
    {
        if !extensions.contains("cl_khr_priority_queue") {
            eprintln!("👎 Device does not support the cl_khr_priority_queue extension.");

            /*
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
            }
             */
        } else {
            println!("🎉 Device support the cl_khr_priority_queue extension!");
        }
    }

    let q = ProQue::builder()
        .src(
            "
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
            }",
        )
        .build()
        .unwrap();

    let mut result = vec![0.0; chunk.num_vecs()];
    let matrix_buffer = Buffer::<f32>::builder()
        .queue(q.queue().clone())
        .len(chunk.num_vecs() * chunk.num_dims())
        .build()
        .unwrap();
    let vector_buffer = Buffer::<f32>::builder()
        .queue(q.queue().clone())
        .len(chunk.num_dims())
        .build()
        .unwrap();
    let result_buffer = Buffer::<f32>::builder()
        .queue(q.queue().clone())
        .len(chunk.num_vecs())
        .build()
        .unwrap();

    let start = Instant::now();
    matrix_buffer.write(chunk.as_ref()).enq().unwrap();
    vector_buffer.write(&first_vec).enq().unwrap();

    let kernel = Kernel::builder()
        .program(&q.program())
        .name("dot_product")
        .queue(q.queue().clone())
        .global_work_size(chunk.num_vecs())
        .arg(&matrix_buffer)
        .arg(&vector_buffer)
        .arg(&result_buffer)
        .arg(chunk.num_vecs() as i32)
        .arg(chunk.num_dims() as i32)
        .build()
        .unwrap();

    unsafe { kernel.enq().unwrap() };
    result_buffer.read(&mut result).enq().unwrap();

    println!(
        "Duration processing {vecs} vectors in OpenCL (full roundtrip): {duration} s",
        vecs = chunk.num_vecs(),
        duration = (Instant::now() - start).as_secs_f32()
    );

    println!("{:?} ...", &result[..10]);
}

async fn load_vectors<const SAMPLE_SIZE: usize>() -> MemoryChunk {
    let mut db = VecDb::open_read(PathBuf::from("vectors.bin"))
        .await
        .unwrap();

    let start = Instant::now();

    let num_vecs = *db.num_vectors;
    let num_dims = *db.num_dimensions;

    let sample_size = (if SAMPLE_SIZE > 0 {
        num_vecs.min(SAMPLE_SIZE)
    } else {
        num_vecs
    })
    .into();

    let mut chunk = MemoryChunk::new(sample_size, db.num_dimensions);
    let data = chunk.as_mut();

    println!("Loading {sample_size} elements from vector database ...");
    let num_read = db
        .read_n_vecs(sample_size, |v, vec| {
            debug_assert_eq!(vec.len(), num_dims);
            #[cfg(debug_assertions)]
            {
                let norm = vec.iter().fold(0.0f32, |prev, x| prev + x * x).sqrt();
                debug_assert!((norm - 1.0f32).abs() < 0.001f32, "Denormal vector detected");
            }

            let start = v * num_dims;
            let end = start + num_dims;
            data[start..end].copy_from_slice(vec);

            true
        })
        .await
        .unwrap();

    let duration = Instant::now() - start;
    println!(
        "Loading duration {} s for {num_read} vectors",
        duration.as_secs_f32()
    );

    chunk
}

#[cfg(test)]
mod test {
    #[test]
    fn it_works() {
        assert!(true);
    }
}
