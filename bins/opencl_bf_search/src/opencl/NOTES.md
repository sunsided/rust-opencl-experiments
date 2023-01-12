# OpenCL Notes

> In this example, I am using three queues to perform a matrix-vector dot product on
> two sets of matrices and vectors, such that writing, kernel execution, and reading are interleaved.
>
> I am writing the first matrix and vector data to the device using matrix_queue and vector_queue, and the first
> result is read from the device using result_queue. After that, I am flushing matrix_queue and vector_queue 
> to make sure that the write operations have been sent to the device, and I am executing the kernel.
>
> Then, I am writing the second matrix and vector data to the device using matrix_queue and vector_queue, 
> and the second result is read from the device using result_queue. After that, I am flushing 
> matrix_queue and vector_queue again to make sure that the write operations have been sent to the device,
> and I am executing the kernel again.
>
> Finally, I am blocking on result_queue to make sure that the read operation has completed, and we have
> the correct results.
>
> By interleaving the operations in this way, we can potentially reduce the overall execution time by
> overlapping the execution of different queues.
>
> It's important to note that, when using multiple queues, you need to be careful to ensure that the
> data is ready before the kernel execution and read operation, otherwise you might get incorrect results.

```rust
fn interleaved_dot_products() {
    // create three queues
    let matrix_queue = ocl::Queue::new(&context, device, None).unwrap();
    let vector_queue = ocl::Queue::new(&context, device, None).unwrap();
    let result_queue = ocl::Queue::new(&context, device, None).unwrap();
    
    // Create buffer for the matrices
    let matrix_buffer = ocl::Buffer::<f32>::builder()
        .queue(matrix_queue.clone())
        .flags(ocl::flags::MEM_READ_ONLY)
        .len(matrix_rows*matrix_cols*2)
        .build()
        .unwrap();
    
    // Create buffer for the vectors
    let vector_buffer = ocl::Buffer::<f32>::builder()
        .queue(vector_queue.clone())
        .flags(ocl::flags::MEM_READ_ONLY)
        .len(matrix_cols*2)
        .build()
        .unwrap();
    
    // Create buffer for the results
    let result_buffer = ocl::Buffer::<f32>::builder()
        .queue(result_queue.clone())
        .flags(ocl::flags::MEM_WRITE_ONLY)
        .len(matrix_rows*2)
        .build()
        .unwrap();
    
    // Create the kernel
    let kernel = ocl::Kernel::builder()
        .program(&program)
        .name("dot")
        .global_work_size(matrix_rows*2)
        .arg(&matrix_buffer)
        .arg(&vector_buffer)
        .arg(&result_buffer)
        .build()
        .unwrap();
    
    // write the first matrix and vector data to the device using matrix_queue
    unsafe {
        matrix_buffer.cmd().write(matrix1.as_slice()).enq().unwrap();
        vector_buffer.cmd().write(vector1.as_slice()).enq().unwrap();
    }
    
    // flush matrix_queue and vector_queue to make sure that the write operations have been sent to the device
    matrix_queue.flush().unwrap();
    vector_queue.flush().unwrap();
    
    // Execute the kernel
    unsafe {
        kernel.cmd().enq().unwrap();
    }
    
    // read the first result from the device using result_queue
    unsafe {
        result_buffer.cmd().read(result1.as_mut_slice()).enq().unwrap();
    }
    
    // flush result_queue to make sure that the read operation has been sent to the device
    result_queue.flush().unwrap();
    
    // write the second matrix and vector data to the device using matrix_queue
    unsafe {
        matrix_buffer.offset(matrix_rows*matrix_cols).write(matrix2.as_slice()).enq().unwrap();
        vector_buffer.offset(matrix_cols).write(vector2.as_slice()).enq().unwrap();
    }
    
    // flush matrix_queue and vector_queue to make sure that the write operations have been sent to the device
    matrix_queue.flush().unwrap();
    vector_queue.flush().unwrap();
    
    // Execute the kernel
    unsafe {
    kernel.cmd().enq().unwrap();
    }
    
    // read the second result from the device using result_queue
    unsafe {
    result_buffer.offset(matrix_rows).read(result2.as_mut_slice()).enq().unwrap();
    }
    
    // block on result_queue to make sure that the read operation has completed
    result_queue.finish().unwrap();
}
```
