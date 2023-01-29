use abstractions::{NumDimensions, NumVectors, Vecgen};
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use criterion::{BenchmarkId, Throughput};
use dot_products::reference::{ReferenceDotProduct, ReferenceDotProductUnrolled};
use dot_products::reference_parallel::{
    ReferenceDotProductParallel, ReferenceDotProductParallelUnrolled,
};
use dot_products::DotProduct;
use memchunk::chunks::{any_size_memory_chunk::AnySizeMemoryChunk, AccessHint};
use std::hint::black_box;

fn reference_dot_products(c: &mut Criterion) {
    let mut chunk = generate_vectors(NumVectors::from(131_072u32), NumDimensions::from(384u32));

    let sizes = [1024usize, 2048, 131_072];

    run_bench_group::<ReferenceDotProduct>(c, "ReferenceDotProduct", &sizes, &mut chunk);
    run_bench_group::<ReferenceDotProductUnrolled<8>>(
        c,
        "ReferenceDotProductUnrolled::<8>",
        &sizes,
        &mut chunk,
    );
    run_bench_group::<ReferenceDotProductUnrolled<16>>(
        c,
        "ReferenceDotProductUnrolled::<16>",
        &sizes,
        &mut chunk,
    );
    run_bench_group::<ReferenceDotProductUnrolled<64>>(
        c,
        "ReferenceDotProductUnrolled::<64>",
        &sizes,
        &mut chunk,
    );
    run_bench_group::<ReferenceDotProductParallel>(
        c,
        "ReferenceDotProductParallel",
        &sizes,
        &mut chunk,
    );
    run_bench_group::<ReferenceDotProductParallelUnrolled<64>>(
        c,
        "ReferenceDotProductParallelUnrolled::<64>",
        &sizes,
        &mut chunk,
    );
}

fn run_bench_group<T: DotProduct + Default>(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[usize],
    chunk: &mut AnySizeMemoryChunk,
) {
    let first_vec = Vec::from(chunk.get_row_major_vec(0));

    let mut group = c.benchmark_group(group_name);
    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));

        chunk.use_num_vecs(size.into());
        let mut reference = vec![0.0; size];

        let algo = T::default();

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                algo.dot_product(
                    black_box(&first_vec),
                    black_box(chunk.as_ref()),
                    black_box(chunk.num_dims()),
                    black_box(chunk.num_vecs()),
                    black_box(&mut reference),
                )
            });
        });
    }
    group.finish();
}

fn generate_vectors(sample_size: NumVectors, num_dimensions: NumDimensions) -> AnySizeMemoryChunk {
    let mut chunk = AnySizeMemoryChunk::new(sample_size, num_dimensions, AccessHint::Random);
    let data = chunk.as_mut();

    println!("Generating {sample_size} random vectors ...");
    let mut vecgen = Vecgen::new_from_seed(0xdeadcafe);
    vecgen.fill(data);
    chunk
}

criterion_group!(benches, reference_dot_products);
criterion_main!(benches);
