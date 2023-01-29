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
use std::path::PathBuf;
use vecdb::VecDb;

fn reference_dot_products(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut chunk = rt.block_on(async { load_vectors(131_072).await });

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
    let first_vec = Vec::from(chunk.get_vec(0));

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

async fn load_vectors(sample_size: usize) -> AnySizeMemoryChunk {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("vectors.bin");
    let mut db = VecDb::open_read(path).await.unwrap();

    let num_vecs = *db.num_vectors;
    let num_dims = *db.num_dimensions;

    let sample_size = (if sample_size > 0 {
        num_vecs.min(sample_size)
    } else {
        num_vecs
    })
    .into();

    let mut chunk = AnySizeMemoryChunk::new(sample_size, db.num_dimensions, AccessHint::Sequential);
    let data = chunk.as_mut();

    println!("Loading {sample_size} elements from vector database ...");
    let num_read = db
        .read_n_vecs(sample_size, |v, vec| {
            let start = v * num_dims;
            let end = start + num_dims;
            data[start..end].copy_from_slice(vec);
            true
        })
        .await
        .unwrap();
    assert_eq!(num_read, *sample_size);
    chunk
}

criterion_group!(benches, reference_dot_products);
criterion_main!(benches);
