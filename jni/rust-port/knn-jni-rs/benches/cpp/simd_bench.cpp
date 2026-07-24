// SPDX-License-Identifier: Apache-2.0
//
// C++ SIMD micro-benchmark — equivalent to benches/simd_bench.rs
// Measures raw distance computation throughput for comparison with Rust.
//
// Build:
//   clang++ -std=c++17 -O3 -o simd_bench simd_bench.cpp -I/opt/homebrew/opt/libomp/include
//
// Run:
//   ./simd_bench

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Distance functions (matching the Rust implementations)
// ---------------------------------------------------------------------------

float l2_distance_scalar(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float inner_product_scalar(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void bulk_inner_product(const float* query, const float* vectors, int dim, int n, float* scores) {
    for (int i = 0; i < n; i++) {
        const float* vec = vectors + i * dim;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum += query[d] * vec[d];
        }
        scores[i] = sum;
    }
}

// FP16 conversion (matches Rust half_to_float)
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = (float)mant / 1024.0f * powf(2.0f, -14.0f);
        return sign ? -val : val;
    }
    if (exp == 31) {
        return mant == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
    }

    int exp32 = (int)exp - 15 + 127;
    uint32_t bits = (sign << 31) | ((uint32_t)exp32 << 23) | (mant << 13);
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static inline uint16_t float_to_half(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(uint32_t));
    uint32_t sign = (bits >> 31) & 1;
    int exp = (int)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | (31 << 10));
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant);
}

float fp16_inner_product(const float* query, const uint16_t* fp16_vec, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += query[i] * half_to_float(fp16_vec[i]);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

std::vector<float> generate_vectors(int n, int dim) {
    std::vector<float> v(n * dim);
    for (int i = 0; i < n * dim; i++) {
        v[i] = sinf(i * 0.01f);
    }
    return v;
}

std::vector<uint16_t> generate_fp16_vectors(int n, int dim) {
    std::vector<uint16_t> v(n * dim);
    for (int i = 0; i < n * dim; i++) {
        v[i] = float_to_half(sinf(i * 0.001f));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchResult {
    double ns_per_iter;
    int iterations;
};

template <typename Fn>
BenchResult bench(Fn fn, int target_ms = 2000) {
    // Warmup
    for (int i = 0; i < 100; i++) fn();

    // Auto-calibrate iterations
    int iters = 1000;
    auto start = Clock::now();
    for (int i = 0; i < iters; i++) fn();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();

    double ns_per = (double)elapsed / iters;
    if (ns_per > 0) {
        iters = (int)((double)target_ms * 1e6 / ns_per);
        if (iters < 100) iters = 100;
        if (iters > 10000000) iters = 10000000;
    }

    // Measure
    start = Clock::now();
    for (int i = 0; i < iters; i++) fn();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();

    return {(double)elapsed / iters, iters};
}

void print_result(const char* name, BenchResult r) {
    if (r.ns_per_iter < 1000) {
        printf("  %-45s  %8.1f ns/iter  (%d iters)\n", name, r.ns_per_iter, r.iterations);
    } else if (r.ns_per_iter < 1000000) {
        printf("  %-45s  %8.1f us/iter  (%d iters)\n", name, r.ns_per_iter / 1000.0, r.iterations);
    } else {
        printf("  %-45s  %8.2f ms/iter  (%d iters)\n", name, r.ns_per_iter / 1e6, r.iterations);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║        C++ SIMD Micro-Benchmark (compare with Rust)          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // L2 distance
    printf("── L2 Distance ──\n");
    for (int dim : {128, 256, 768}) {
        auto a = generate_vectors(1, dim);
        auto b = generate_vectors(1, dim);
        // Shift b slightly so it's different from a
        for (int i = 0; i < dim; i++) b[i] += 0.1f;

        char name[64];
        snprintf(name, sizeof(name), "l2_scalar/dim=%d", dim);
        volatile float result;
        print_result(name, bench([&]() {
            result = l2_distance_scalar(a.data(), b.data(), dim);
        }));
    }

    // Inner product
    printf("\n── Inner Product ──\n");
    for (int dim : {128, 256, 768}) {
        auto a = generate_vectors(1, dim);
        auto b = generate_vectors(1, dim);
        for (int i = 0; i < dim; i++) b[i] += 0.1f;

        char name[64];
        snprintf(name, sizeof(name), "ip_scalar/dim=%d", dim);
        volatile float result;
        print_result(name, bench([&]() {
            result = inner_product_scalar(a.data(), b.data(), dim);
        }));
    }

    // Bulk inner product
    printf("\n── Bulk Inner Product (dim=128) ──\n");
    for (int n : {100, 1000, 10000}) {
        int dim = 128;
        auto query = generate_vectors(1, dim);
        auto vectors = generate_vectors(n, dim);
        std::vector<float> scores(n);

        char name[64];
        snprintf(name, sizeof(name), "bulk_ip/n=%d", n);
        print_result(name, bench([&]() {
            bulk_inner_product(query.data(), vectors.data(), dim, n, scores.data());
        }));
    }

    // FP16 inner product
    printf("\n── FP16 Inner Product (dim=128) ──\n");
    for (int n : {100, 1000}) {
        int dim = 128;
        auto query = generate_vectors(1, dim);
        auto fp16_vecs = generate_fp16_vectors(n, dim);

        char name[64];
        snprintf(name, sizeof(name), "fp16_ip/n=%d", n);
        volatile float total;
        print_result(name, bench([&]() {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += fp16_inner_product(query.data(), fp16_vecs.data() + i * dim, dim);
            }
            total = sum;
        }));
    }

    printf("\nDone.\n");
    return 0;
}
