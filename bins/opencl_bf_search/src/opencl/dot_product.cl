#define ROW_DIM 0
#define COL_DIM 1

__kernel void dot_product(const __global float *a,
                         const __global float *x,
                         __global float *y,
                         __local float *work,
                         unsigned int m,
                         unsigned int n) {

    // Compute partial dot product
    float sum = (float)0;
    for (int k = get_global_id(COL_DIM); k < n; k += get_global_size(COL_DIM))
    {
        sum += a[get_global_id(ROW_DIM) + m * k] * x[k];
    }

    // Each thread stores its partial sum in WORK
    int rows = get_local_size(ROW_DIM); // rows in group
    int cols = get_local_size(COL_DIM); // initial cols in group
    int ii = get_local_id(ROW_DIM); // local row index in group, 0<=ii<rows
    int jj = get_local_id(COL_DIM); // block index in column, 0<=jj<cols
    work[ii + rows * jj] = sum;
    barrier(CLK_LOCAL_MEM_FENCE); // sync group

    // Reduce sums in log2(cols) steps
    while ( cols > 1 )
    {
        cols >>= 1;
        if (jj < cols) {
            work[ii + rows * jj] += work[ii + rows * (jj + cols)];
        }
        barrier(CLK_LOCAL_MEM_FENCE); // sync group
    }

    // Write final result in Y
    if ( jj == 0 ) {
        y[get_global_id(ROW_DIM)] = work[ii];
    }
}
