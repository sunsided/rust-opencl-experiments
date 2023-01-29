use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main};
use criterion::{BenchmarkGroup, Criterion};
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

fn bench<T: DotProduct>(
    group: &mut BenchmarkGroup<WallTime>,
    size: usize,
    first_vec: &[f32],
    chunk: &AnySizeMemoryChunk,
    algo: T,
) {
    let mut reference = vec![0.0; chunk.num_vecs().get()];

    group.bench_function(BenchmarkId::from_parameter(size), |b| {
        b.iter(|| {
            algo.dot_product(
                black_box(first_vec),
                black_box(chunk.as_ref()),
                black_box(chunk.num_dims()),
                black_box(chunk.num_vecs()),
                black_box(&mut reference),
            )
        });
    });
}

fn from_elem(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut chunk = rt.block_on(async { load_vectors(131_072).await });

    let first_vec = Vec::from(chunk.get_vec(0));
    let sizes = [1024usize, 2048, 131_072];

    let mut group = c.benchmark_group("search_naive");
    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));
        chunk.use_num_vecs(size.into());

        let algo = ReferenceDotProduct::default();
        bench(&mut group, size, &first_vec, &chunk, algo);
    }
    group.finish();

    let mut group = c.benchmark_group("search_unrolled::<8>");
    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));
        chunk.use_num_vecs(size.into());

        let algo = ReferenceDotProductUnrolled::<8>::default();
        bench(&mut group, size, &first_vec, &chunk, algo);
    }
    group.finish();

    let mut group = c.benchmark_group("search_unrolled::<16>");
    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));
        chunk.use_num_vecs(size.into());

        let algo = ReferenceDotProductUnrolled::<16>::default();
        bench(&mut group, size, &first_vec, &chunk, algo);
    }
    group.finish();

    let mut group = c.benchmark_group("search_unrolled::<64>");
    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));
        chunk.use_num_vecs(size.into());

        let algo = ReferenceDotProductUnrolled::<64>::default();
        bench(&mut group, size, &first_vec, &chunk, algo);
    }
    group.finish();

    let mut group = c.benchmark_group("search_naive_parallel");
    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));
        chunk.use_num_vecs(size.into());

        let algo = ReferenceDotProductParallel::default();
        bench(&mut group, size, &first_vec, &chunk, algo);
    }
    group.finish();

    let mut group = c.benchmark_group("search_parallel_unrolled::<64>");
    for &size in sizes.iter() {
        group.throughput(Throughput::Elements(size as u64));
        chunk.use_num_vecs(size.into());

        let algo = ReferenceDotProductParallelUnrolled::<64>::default();
        bench(&mut group, size, &first_vec, &chunk, algo);
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

criterion_group!(benches, from_elem);
criterion_main!(benches);
