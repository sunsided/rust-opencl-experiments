mod opencl;
mod vec_traits;
mod vecgen;

use crate::opencl::{
    build_dot_product_program, build_dot_topk_program, build_priority_queue_program,
};
use memchunk::MemoryChunk;
use ocl::{Buffer, Context, Device, Kernel, Platform, Queue};
use std::path::PathBuf;
use std::time::Instant;
use vecdb::VecDb;

#[tokio::main]
async fn main() {
    #[cfg(debug_assertions)]
    const K: usize = 10_000;
    #[cfg(not(debug_assertions))]
    const K: usize = 0;
    let mut chunk = load_vectors::<K>().await;
    let first_vec = Vec::from(chunk.get_vec(0));

    chunk.use_num_vecs((chunk.num_vecs() & !(32 - 1)).into());
    println!("Using {} vectors.", chunk.num_vecs());

    let start = Instant::now();
    let _reference = chunk.search_naive(&first_vec);
    println!(
        "Duration processing {vecs} vectors on CPU: {duration} s",
        vecs = chunk.num_vecs(),
        duration = (Instant::now() - start).as_secs_f32()
    );

    // Default setup.
    let platform = Platform::first().unwrap();
    println!(
        "Using platform {} with {}",
        platform.name().unwrap(),
        platform.version().unwrap()
    );

    let device = Device::first(platform).unwrap();
    println!("Using device {}", device.name().unwrap());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()
        .unwrap();

    let dot_product = build_dot_product_program(device, &context).unwrap();

    // Create three queues.
    let matrix_queue = Queue::new(&context, device, None).unwrap();
    let vector_queue = Queue::new(&context, device, None).unwrap();
    let result_queue = Queue::new(&context, device, None).unwrap();
    // TODO: Introduce another queue for reducing the results?

    // Write matrix data to the device using matrix_queue.
    let matrix_buffer = Buffer::<f32>::builder()
        .queue(matrix_queue.clone())
        .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
        .len(chunk.num_vecs() * chunk.num_dims())
        .build()
        .unwrap();

    // Write vector data to the device using vector_queue.
    let vector_buffer = Buffer::<f32>::builder()
        .queue(vector_queue.clone())
        .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY)
        .len(chunk.num_dims())
        .build()
        .unwrap();

    // Write result data to the device using result_queue.
    let result_buffer = Buffer::<f32>::builder()
        .queue(result_queue.clone())
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
        .len(chunk.num_vecs())
        .build()
        .unwrap();

    // Execute kernel using result_queue.
    let dot_product_kernel = Kernel::builder()
        .program(&dot_product)
        .name("dot_product")
        .queue(result_queue.clone())
        .global_work_size(chunk.num_vecs())
        //.local_work_size(16)
        .local_work_size(4)
        .arg(&matrix_buffer)
        .arg(&vector_buffer)
        .arg(&result_buffer)
        .arg(chunk.num_vecs() as u32)
        .arg(chunk.num_dims() as u32)
        .build()
        .unwrap();

    let transposed = chunk.as_transposed();

    let start = Instant::now();

    matrix_buffer.cmd().write(&transposed).enq().unwrap();
    vector_buffer.cmd().write(&first_vec).enq().unwrap();

    // Flush the matrix and vector queues to make sure that the write
    // operations have been sent to the device
    matrix_queue.flush().unwrap();
    vector_queue.flush().unwrap();

    // Execute the dot product kernel.
    unsafe { dot_product_kernel.cmd().enq().unwrap() };

    let mut results = vec![f32::NAN; chunk.num_vecs()];
    result_buffer.cmd().read(&mut results).enq().unwrap();

    // Flush result_queue to make sure that the read operation has been sent to the device.
    result_queue.flush().unwrap();

    // TODO: Write next matrix ...
    // TODO: Write next vector ...

    // Just to ensure we have everything set here in the single-matrix example, we now
    // block on the result queue to make sure that the read operation has completed
    result_queue.finish().unwrap();

    println!(
        "Duration processing {vecs} vectors in OpenCL (full roundtrip): {duration} s",
        vecs = chunk.num_vecs(),
        duration = (Instant::now() - start).as_secs_f32()
    );

    println!("{:?} ...", &results[..10]);
    println!(
        "{:?} ...",
        &results[chunk.num_dims()..(chunk.num_dims() + 10)]
    );
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
