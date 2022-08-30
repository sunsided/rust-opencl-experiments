use ocl::ProQue;
use ocl_stream::OCLStreamExecutor;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn dotproduct(veca: &Vec<f64>, vecb: &Vec<f64>) -> f64 {
    veca.iter().zip(vecb.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(vec: &Vec<f64>) -> f64 {
    match f64::sqrt(vec.iter().map(|x| x * x).sum()) {
        0.0 => 1.0,
        x => x
    }
}

fn normalize(vec: Vec<f64>) -> Vec<f64> {
    let inv_norm = 1.0 / l2_norm(&vec);
    vec.iter().map(|x| x * inv_norm).collect()
}

fn main() {
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
    use super::*;
    use approx::relative_eq;

    #[test]
    fn norm_works() {
        let normalized = normalize(vec![1.0, 1.0, 0.0]);
        relative_eq!(normalized[0], 0.5*f64::sqrt(2.0), epsilon = 1e-5);
        relative_eq!(normalized[1], 0.5*f64::sqrt(2.0), epsilon = 1e-5);
        assert_eq!(normalized[2], 0.0);
    }
}
