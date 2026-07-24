// SPDX-License-Identifier: Apache-2.0
//
// Criterion benchmarks for SIMD similarity functions.
// Measures vectorized distance computation throughput.
//
// Run: KNN_JNI_STATIC=1 cargo bench --bench simd_bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

/// Scalar L2 distance (baseline)
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Scalar inner product (baseline)
fn inner_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Vectorized L2 using auto-vectorization hints
fn l2_distance_autovec(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Vectorized inner product using auto-vectorization hints
fn inner_product_autovec(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Bulk inner product: compute IP for multiple vectors against a single query
fn bulk_inner_product(query: &[f32], vectors: &[f32], dim: usize, scores: &mut [f32]) {
    let n = scores.len();
    for i in 0..n {
        let offset = i * dim;
        let vec = &vectors[offset..offset + dim];
        scores[i] = inner_product_autovec(query, vec);
    }
}

/// FP16 to FP32 conversion + inner product (simulates what SIMD code does)
fn fp16_inner_product(query: &[f32], fp16_vec: &[u16], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..dim {
        let val = half_to_float(fp16_vec[i]);
        sum += query[i] * val;
    }
    sum
}

/// Simple FP16 to FP32 conversion (matches what NEON/AVX512 vcvt does)
fn half_to_float(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        // Denormal
        let mut val = mant as f32 / 1024.0 * (2.0f32).powi(-14);
        if sign == 1 { val = -val; }
        return val;
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else { f32::NAN };
    }

    let exp32 = exp as i32 - 15 + 127;
    let bits = (sign << 31) | ((exp32 as u32) << 23) | (mant << 13);
    f32::from_bits(bits)
}

/// Generate FP16 vectors (stored as u16)
fn generate_fp16_vectors(n: usize, dim: usize) -> Vec<u16> {
    let mut vecs = Vec::with_capacity(n * dim);
    for i in 0..n * dim {
        let val = (i as f32 * 0.001).sin();
        vecs.push(float_to_half(val));
    }
    vecs
}

fn float_to_half(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 { return (sign << 15) as u16; }
    if exp >= 31 { return ((sign << 15) | (31 << 10)) as u16; }
    ((sign << 15) | ((exp as u32) << 10) | mant) as u16
}

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n * dim {
        vectors.push((i as f32 * 0.01).sin());
    }
    vectors
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance");

    for &dim in &[128, 256, 768] {
        let a = generate_vectors(1, dim);
        let b = generate_vectors(1, dim);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(l2_distance_scalar(&a, &b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("autovec", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(l2_distance_autovec(&a, &b)));
            },
        );
    }
    group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");

    for &dim in &[128, 256, 768] {
        let a = generate_vectors(1, dim);
        let b = generate_vectors(1, dim);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(inner_product_scalar(&a, &b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("autovec", dim),
            &dim,
            |bench, _| {
                bench.iter(|| black_box(inner_product_autovec(&a, &b)));
            },
        );
    }
    group.finish();
}

fn bench_bulk_similarity(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("bulk_similarity");

    for &n in &[100, 1000, 10000] {
        let query = generate_vectors(1, dim);
        let vectors = generate_vectors(n, dim);
        let mut scores = vec![0.0f32; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("IP_f32", n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    bulk_inner_product(&query, &vectors, dim, &mut scores);
                    black_box(&scores);
                });
            },
        );
    }
    group.finish();
}

fn bench_fp16_similarity(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("fp16_similarity");

    for &n in &[100, 1000] {
        let query = generate_vectors(1, dim);
        let fp16_vectors = generate_fp16_vectors(n, dim);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("IP_fp16_convert", n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    let mut total = 0.0f32;
                    for i in 0..n {
                        let offset = i * dim;
                        total += fp16_inner_product(
                            &query,
                            &fp16_vectors[offset..offset + dim],
                            dim,
                        );
                    }
                    black_box(total)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_l2_distance,
    bench_inner_product,
    bench_bulk_similarity,
    bench_fp16_similarity,
);
criterion_main!(benches);
