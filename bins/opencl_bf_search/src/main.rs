mod vec_traits;
mod vecgen;

use memchunk::MemoryChunk;
use ocl::ProQue;
use ocl_stream::OCLStreamExecutor;
use std::path::PathBuf;
use std::time::Instant;
use vecdb::VecDb;

#[tokio::main]
async fn main() {
    let mut db = VecDb::open_read(PathBuf::from("vectors.bin"))
        .await
        .unwrap();

    let start = Instant::now();

    let num_vecs = *db.num_vectors;
    let num_dims = *db.num_dimensions;

    const SAMPLE_SIZE: usize = 10_000;
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

    let first_vec = Vec::from(&data[0..num_dims]);
    const TRIALS: usize = 100;
    println!("Running {TRIALS} queries");

    let _ref = chunk.search_naive(&first_vec);

    let start = Instant::now();
    for _ in 0..TRIALS {
        let _result =
            std::hint::black_box(chunk.search_unrolled::<8>(std::hint::black_box(&first_vec)));
    }
    let duration = Instant::now() - start;
    println!(
        "Duration searching for {TRIALS} vectors: {} s ({} queries/s, {} vecs/s)",
        duration.as_secs_f32(),
        TRIALS as f32 / duration.as_secs_f32(),
        (TRIALS * num_vecs) as f32 / duration.as_secs_f32()
    );

    // create the ProQue
    let pro_que = ProQue::builder()
        .src(
            "
            __kernel void bench_int(const uint limit, __global int *NUMBERS) {
                uint id = get_global_id(0);
                int num = NUMBERS[id];
                for (int i = 0; i < limit; i++) {
                    num += i;
                }
                NUMBERS[id] = num;
            }",
        )
        .dims(1u32)
        .build()
        .unwrap();

    // create the executor
    let stream_executor = OCLStreamExecutor::new(pro_que);

    // execute a closure that provides the results in the given stream
    let mut stream = stream_executor.execute_unbounded(|ctx| {
        let pro_que = ctx.pro_que();
        let tx = ctx.sender();
        let input_buffer = pro_que
            .buffer_builder()
            .len(100)
            .fill_val(0i32)
            .build()
            .expect("building the buffer failed");

        let kernel = pro_que
            .kernel_builder("bench_int")
            .arg(100u32)
            .arg(&input_buffer)
            .global_work_size(100)
            .build()
            .expect("building the kernel failed");
        unsafe {
            kernel.enq().expect("enqueueing the kernel failed");
        }

        let mut result = vec![0i32; 100];
        input_buffer
            .read(&mut result)
            .enq()
            .expect("enqueueing the input buffer failed");

        for num in result {
            // send the results to the receiving thread
            tx.send(num).expect("sending the value failed");
        }

        Ok(())
    });

    let mut count = 0;

    // calculate the expected result values
    let num = (99f32.powf(2.0) + 99f32) / 2f32;
    // read the results from the stream
    while let Ok(n) = stream.next() {
        assert_eq!(n, num as i32);
        count += 1;
    }
    assert_eq!(count, 100)
}

#[cfg(test)]
mod test {
    #[test]
    fn it_works() {
        assert!(true);
    }
}
