// SPDX-License-Identifier: Apache-2.0
//
// Criterion benchmarks for Faiss operations via the Rust JNI layer.
// Measures the native-side performance without JVM overhead.
//
// Run: KNN_JNI_STATIC=1 cargo bench --bench faiss_bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::ffi::CString;
use std::ptr;

// FFI declarations for Faiss C API (same as used in the crate)
extern "C" {
    fn faiss_index_factory(
        p_index: *mut *mut u8,
        d: i32,
        description: *const i8,
        metric: i32,
    ) -> i32;
    fn faiss_index_add_with_ids(
        index: *mut u8,
        n: i64,
        x: *const f32,
        xids: *const i64,
    ) -> i32;
    fn faiss_index_search(
        index: *mut u8,
        n: i64,
        x: *const f32,
        k: i64,
        distances: *mut f32,
        labels: *mut i64,
    ) -> i32;
    fn faiss_index_free(index: *mut u8);
    fn faiss_index_train(index: *mut u8, n: i64, x: *const f32) -> i32;
}

const METRIC_L2: i32 = 1;

/// Generate deterministic test vectors
fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for d in 0..dim {
            vectors.push(((i * dim + d) as f32 * 0.01).sin());
        }
    }
    vectors
}

/// Generate IDs
fn generate_ids(n: usize) -> Vec<i64> {
    (0..n as i64).collect()
}

/// Create an HNSW index, add vectors, return index pointer
unsafe fn create_hnsw_index(vectors: &[f32], ids: &[i64], dim: i32) -> *mut u8 {
    let desc = CString::new("HNSW32,Flat").unwrap();
    let mut index: *mut u8 = ptr::null_mut();
    let ret = faiss_index_factory(&mut index, dim, desc.as_ptr(), METRIC_L2);
    assert_eq!(ret, 0, "faiss_index_factory failed");
    assert!(!index.is_null());

    let n = ids.len() as i64;
    let ret = faiss_index_add_with_ids(index, n, vectors.as_ptr(), ids.as_ptr());
    assert_eq!(ret, 0, "faiss_index_add_with_ids failed");

    index
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_index_creation(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("index_creation");

    for &n in &[1000, 10000] {
        let vectors = generate_vectors(n, dim);
        let ids = generate_ids(n);

        group.bench_with_input(
            BenchmarkId::new("HNSW32_Flat", n),
            &n,
            |b, _| {
                b.iter(|| unsafe {
                    let index = create_hnsw_index(&vectors, &ids, dim as i32);
                    faiss_index_free(index);
                });
            },
        );
    }
    group.finish();
}

fn bench_query_latency(c: &mut Criterion) {
    let dim = 128;
    let n = 10000;
    let vectors = generate_vectors(n, dim);
    let ids = generate_ids(n);
    let query = generate_vectors(1, dim);

    let index = unsafe { create_hnsw_index(&vectors, &ids, dim as i32) };

    let mut group = c.benchmark_group("query_latency");

    for &k in &[1, 10, 100] {
        group.bench_with_input(
            BenchmarkId::new("HNSW32_k", k),
            &k,
            |b, &k| {
                let mut distances = vec![0.0f32; k];
                let mut labels = vec![0i64; k];
                b.iter(|| unsafe {
                    faiss_index_search(
                        black_box(index),
                        1,
                        query.as_ptr(),
                        k as i64,
                        distances.as_mut_ptr(),
                        labels.as_mut_ptr(),
                    );
                    black_box(&distances);
                    black_box(&labels);
                });
            },
        );
    }

    group.finish();
    unsafe { faiss_index_free(index); }
}

fn bench_vector_storage(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("vector_storage");

    for &n in &[1000, 10000, 100000] {
        group.bench_with_input(
            BenchmarkId::new("alloc_and_fill", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let v: Vec<f32> = generate_vectors(n, dim);
                    black_box(&v);
                });
            },
        );
    }
    group.finish();
}

fn bench_train_index(c: &mut Criterion) {
    let dim = 64;
    let n = 10000;
    let vectors = generate_vectors(n, dim);

    let mut group = c.benchmark_group("train_index");

    group.bench_function("IVF16_Flat", |b| {
        b.iter(|| unsafe {
            let desc = CString::new("IVF16,Flat").unwrap();
            let mut index: *mut u8 = ptr::null_mut();
            let ret = faiss_index_factory(&mut index, dim as i32, desc.as_ptr(), METRIC_L2);
            assert_eq!(ret, 0);

            let ret = faiss_index_train(index, n as i64, vectors.as_ptr());
            assert_eq!(ret, 0);

            faiss_index_free(index);
        });
    });

    group.finish();
}

fn bench_add_vectors(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("add_vectors");

    for &n in &[1000, 10000] {
        let vectors = generate_vectors(n, dim);
        let ids = generate_ids(n);

        group.bench_with_input(
            BenchmarkId::new("HNSW32", n),
            &n,
            |b, _| {
                b.iter(|| unsafe {
                    let desc = CString::new("HNSW32,Flat").unwrap();
                    let mut index: *mut u8 = ptr::null_mut();
                    faiss_index_factory(&mut index, dim as i32, desc.as_ptr(), METRIC_L2);
                    faiss_index_add_with_ids(
                        index,
                        n as i64,
                        vectors.as_ptr(),
                        ids.as_ptr(),
                    );
                    faiss_index_free(index);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_index_creation,
    bench_query_latency,
    bench_vector_storage,
    bench_train_index,
    bench_add_vectors,
);
criterion_main!(benches);
