__kernel void topk(__global float* data, __global float* topk, __global uint* topk_idx, uint data_size, uint topk_size) {
    int gid = get_global_id(0);
    float cur_val = data[gid];
    uint cur_idx = gid;
    if (gid < topk_size) {
        topk[gid] = cur_val;
        topk_idx[gid] = cur_idx;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int i = topk_size / 2; i > 0; i >>= 1) {
        if (gid < i) {
            if (cur_val > topk[gid + i]) {
                topk[gid + i] = cur_val;
                topk_idx[gid + i] = cur_idx;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
