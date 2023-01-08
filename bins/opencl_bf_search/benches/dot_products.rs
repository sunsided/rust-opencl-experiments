use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use criterion::{BenchmarkId, Throughput};
use memchunk::MemoryChunk;
use std::hint::black_box;
use std::path::PathBuf;
use vecdb::VecDb;

fn from_elem(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut chunk = rt.block_on(async { load_vectors(131_072).await });

    let first_vec = Vec::from(chunk.get_vec(0));
    let sizes = [1024usize, 2048, 131_072];

    let mut group = c.benchmark_group("search_naive");
    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));
        chunk.use_num_vecs((*size).into());
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| chunk.search_naive(black_box(&first_vec)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("search_unrolled::<8>");
    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));
        chunk.use_num_vecs((*size).into());
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| chunk.search_unrolled::<8>(black_box(&first_vec)));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("search_unrolled::<16>");
    for size in sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));
        chunk.use_num_vecs((*size).into());
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| chunk.search_unrolled::<16>(black_box(&first_vec)));
        });
    }
    group.finish();
}

async fn load_vectors(sample_size: usize) -> MemoryChunk {
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

    let mut chunk = MemoryChunk::new(sample_size, db.num_dimensions);
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
