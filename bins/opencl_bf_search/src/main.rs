mod cli;
mod opencl;
mod vec_traits;
mod vecgen;

use crate::cli::match_cli_arguments;
use crate::opencl::{
    build_dot_product_program, get_opencl_selection, ocl_print_platforms, OpenClDeviceSelection,
};
use memchunk::{AnySizeMemoryChunk, DotProduct, ReferenceDotProduct};
use ocl::{Buffer, Context, Kernel, MemFlags, Queue};
use std::path::PathBuf;
use std::time::Instant;
use vecdb::VecDb;

#[tokio::main]
async fn main() {
    let matches = match_cli_arguments();

    if matches.get_flag("ocl-list-platforms") {
        ocl_print_platforms();
        std::process::exit(0);
    }

    let db_file = matches
        .get_one("vector-db")
        .expect("input argument missing");

    let num_vecs = matches
        .get_one::<usize>("max-vectors")
        .expect("invalid number of vectors")
        .to_owned();

    let opencl_selection = get_opencl_selection(&matches);

    let mut chunk = load_vectors(db_file, num_vecs).await;
    let first_vec = Vec::from(chunk.get_vec(0));

    chunk.double();

    // HACK: Ensure number of vectors is multiple of 32.
    chunk.use_num_vecs((chunk.num_vecs().into_inner() & !(32 - 1)).into());
    println!("Using {} vectors.", chunk.num_vecs());

    let reference_algo = ReferenceDotProduct::default();
    let mut reference = vec![0.0; chunk.num_vecs().into_inner()];

    let start = Instant::now();
    reference_algo.dot_product(
        &first_vec,
        chunk.as_ref(),
        chunk.num_dims(),
        chunk.num_vecs(),
        &mut reference,
    );
    let duration_cpu = (Instant::now() - start).as_secs_f32();
    println!(
        "Duration processing {vecs} vectors on CPU: {duration} s",
        vecs = chunk.num_vecs(),
        duration = duration_cpu
    );

    println!("{:?} ...", &reference[..10]);
    println!(
        "{:?} ...",
        &reference[chunk.num_dims().into_inner()..(chunk.num_dims().into_inner() + 10)]
    );

    if opencl_selection.is_none() {
        return;
    }

    let OpenClDeviceSelection {
        platform, device, ..
    } = opencl_selection.expect("invalid selection");

    // Default setup.
    println!(
        "Using platform {} with {}",
        platform.name().unwrap(),
        platform.version().unwrap()
    );

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
        .flags(MemFlags::new().read_only().host_write_only())
        .len(chunk.num_vecs() * chunk.num_dims())
        .build()
        .unwrap();

    // Write vector data to the device using vector_queue.
    let vector_buffer = Buffer::<f32>::builder()
        .queue(vector_queue.clone())
        .flags(MemFlags::new().read_only().host_write_only())
        .len(chunk.num_dims().into_inner())
        .build()
        .unwrap();

    // Write result data to the device using result_queue.
    let result_buffer = Buffer::<f32>::builder()
        .queue(result_queue.clone())
        .flags(MemFlags::new().write_only().host_read_only())
        .len(chunk.num_vecs().into_inner())
        .build()
        .unwrap();

    // Execute kernel using result_queue.
    const X: usize = 16;
    const P: usize = 16;

    let dot_product_kernel = Kernel::builder()
        .program(&dot_product)
        .name("dot_product")
        .queue(result_queue.clone())
        .global_work_size([chunk.num_vecs().into_inner(), P])
        .local_work_size([X, P])
        .arg(&matrix_buffer)
        .arg(&vector_buffer)
        .arg(&result_buffer)
        .arg_local::<f32>(X * (P + 1))
        .arg(chunk.num_vecs().into_inner() as u32)
        .arg(chunk.num_dims().into_inner() as u32)
        .build()
        .unwrap();

    println!("Transposing matrix ...");
    let transposed = chunk.as_transposed();

    println!("Processing using OpenCL ...");
    let start = Instant::now();

    // Write the buffer using memory mapping (since pinning isn't supported).
    // This did not provide any noticeable performance benefit on the Intel Iris XE
    // and is kept here only for reference.
    //
    // Moreover, this also seemed to produce empty buffers on an NVidia GTX 980 Ti.
    /*unsafe {
        let mut mem_map = matrix_buffer.map().enq().unwrap();
        mem_map.copy_from_slice(&transposed);
        mem_map.unmap().enq().unwrap();
    }*/

    matrix_buffer.cmd().write(&transposed).enq().unwrap();
    vector_buffer.cmd().write(&first_vec).enq().unwrap();

    // Flush the matrix and vector queues to make sure that the write
    // operations have been sent to the device
    matrix_queue.flush().unwrap();
    vector_queue.flush().unwrap();

    // Execute the dot product kernel.
    let start_kernel = Instant::now();
    unsafe { dot_product_kernel.cmd().enq().unwrap() };

    let mut results = vec![f32::NAN; chunk.num_vecs().into_inner()];
    result_buffer.cmd().read(&mut results).enq().unwrap();

    // Flush result_queue to make sure that the read operation has been sent to the device.
    result_queue.flush().unwrap();

    // TODO: Write next matrix ...
    // TODO: Write next vector ...

    // Just to ensure we have everything set here in the single-matrix example, we now
    // block on the result queue to make sure that the read operation has completed
    result_queue.finish().unwrap();

    let duration_ocl = (Instant::now() - start).as_secs_f32();
    let duration_ocl_kernel = (Instant::now() - start_kernel).as_secs_f32();
    println!(
        "Duration processing {vecs} vectors in OpenCL (full roundtrip): {duration} s (x{ratio}), kernel only: {duration_kernel} s (x{ratio_kernel})",
        vecs = chunk.num_vecs(),
        duration = duration_ocl,
        ratio = duration_cpu / duration_ocl,
        duration_kernel = duration_ocl_kernel,
        ratio_kernel = duration_cpu / duration_ocl_kernel,
    );

    println!("{:?} ...", &results[..10]);
    println!(
        "{:?} ...",
        &results[chunk.num_dims().into_inner()..(chunk.num_dims().into_inner() + 10)]
    );
}

async fn load_vectors(db_file: &PathBuf, sample_size: usize) -> AnySizeMemoryChunk {
    let mut db = VecDb::open_read(db_file).await.unwrap();

    let start = Instant::now();

    let num_vecs = *db.num_vectors;
    let num_dims = *db.num_dimensions;

    let sample_size = (if sample_size > 0 {
        num_vecs.min(sample_size)
    } else {
        num_vecs
    })
    .into();

    let mut chunk = AnySizeMemoryChunk::new(sample_size, db.num_dimensions);
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
