use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use memchunk::MemoryChunk;
use std::hint::black_box;
use std::path::PathBuf;
use vecdb::VecDb;

fn from_elem(c: &mut Criterion) {
    let size: usize = 1024;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let chunk = rt.block_on(async { load_vectors(size).await });

    let first_vec = Vec::from(chunk.get_vec(0));

    c.bench_with_input(BenchmarkId::new("search_naive", size), &size, |b, &s| {
        b.iter(|| chunk.search_naive(black_box(&first_vec)));
    });

    c.bench_with_input(
        BenchmarkId::new("search_unrolled::<4>", size),
        &size,
        |b, &s| {
            b.iter(|| chunk.search_unrolled::<4>(black_box(&first_vec)));
        },
    );

    c.bench_with_input(
        BenchmarkId::new("search_unrolled::<8>", size),
        &size,
        |b, &s| {
            b.iter(|| chunk.search_unrolled::<8>(black_box(&first_vec)));
        },
    );

    c.bench_with_input(
        BenchmarkId::new("search_unrolled::<16>", size),
        &size,
        |b, &s| {
            b.iter(|| chunk.search_unrolled::<16>(black_box(&first_vec)));
        },
    );
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