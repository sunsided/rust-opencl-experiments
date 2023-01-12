mod opencl;
mod vec_traits;
mod vecgen;

use crate::opencl::{build_dot_product_program, build_priority_queue_program};
use memchunk::MemoryChunk;
use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};
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

    let dot_product = build_dot_product_program(device, &context).unwrap();
    let _program = build_priority_queue_program(&device, &context);

    let mut result = vec![0.0; chunk.num_vecs()];

    let queue = Queue::new(&context, device, None).unwrap();

    let matrix_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(chunk.num_vecs() * chunk.num_dims())
        .build()
        .unwrap();
    let vector_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(chunk.num_dims())
        .build()
        .unwrap();
    let result_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(chunk.num_vecs())
        .build()
        .unwrap();

    let start = Instant::now();
    matrix_buffer.write(chunk.as_ref()).enq().unwrap();
    vector_buffer.write(&first_vec).enq().unwrap();

    let dot_product_kernel = Kernel::builder()
        .program(&dot_product)
        .name("dot_product")
        .queue(queue.clone())
        .global_work_size(chunk.num_vecs())
        // .local_work_size(79)
        .arg(&matrix_buffer)
        .arg(&vector_buffer)
        .arg(&result_buffer)
        .arg(chunk.num_vecs() as i32)
        .arg(chunk.num_dims() as i32)
        .build()
        .unwrap();

    unsafe { dot_product_kernel.enq().unwrap() };
    result_buffer.read(&mut result).enq().unwrap();
    queue.finish().unwrap();

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
