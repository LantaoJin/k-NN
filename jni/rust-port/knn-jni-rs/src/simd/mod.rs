// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! SIMD dispatch module: selects appropriate similarity function implementation
//! based on CPU features (AVX512, NEON, default scalar).
//!
//! This is a Rust port of the C++ SIMD similarity function dispatch system from:
//! `jni/src/simd/similarity_function/similarity_function.cpp`

use std::alloc::{self, Layout};
use std::ptr;

// ---------------------------------------------------------------------------
// FFI declarations for Faiss distance computation
// These will be moved to ffi/ module later.
// ---------------------------------------------------------------------------
pub mod ffi {
    use std::os::raw::c_void;

    /// Opaque type representing faiss::DistanceComputer
    #[repr(C)]
    pub struct FaissDistanceComputer {
        _opaque: [u8; 0],
    }

    /// Opaque type representing faiss::ScalarQuantizer::SQDistanceComputer
    #[repr(C)]
    pub struct FaissSQDistanceComputer {
        _opaque: [u8; 0],
    }

    extern "C" {
        /// Create a ScalarQuantizer distance computer for the given dimension, quantizer type,
        /// and metric type. Returns an owning pointer that must be freed with
        /// `faiss_distance_computer_free`.
        ///
        /// quantizer_type: 0 = QT_fp16
        /// metric_type: 0 = METRIC_INNER_PRODUCT, 1 = METRIC_L2
        pub fn faiss_sq_get_distance_computer(
            dimension: usize,
            quantizer_type: i32,
            metric_type: i32,
        ) -> *mut FaissDistanceComputer;

        /// Set the query vector on a DistanceComputer.
        pub fn faiss_distance_computer_set_query(
            dc: *mut FaissDistanceComputer,
            query: *const f32,
        );

        /// Compute distance from query to a single encoded vector (code).
        pub fn faiss_sq_distance_computer_query_to_code(
            dc: *mut FaissSQDistanceComputer,
            code: *const u8,
        ) -> f32;

        /// Free a DistanceComputer.
        pub fn faiss_distance_computer_free(dc: *mut FaissDistanceComputer);
    }
}

// ---------------------------------------------------------------------------
// Score transform functions (Faiss score -> Lucene score)
// Port of faiss_score_to_lucene_transform.cpp
// ---------------------------------------------------------------------------

/// Convert Faiss inner product value to Max Inner Product scheme.
/// Range: [0, +Inf)
#[inline]
pub fn ip_to_max_ip_transform(inner_product_value: f32) -> f32 {
    if inner_product_value < 0.0 {
        1.0 / (1.0 - inner_product_value)
    } else {
        1.0 + inner_product_value
    }
}

/// Bulk transform Faiss inner product values to Max Inner Product scheme.
#[inline]
pub fn ip_to_max_ip_transform_bulk(scores: &mut [f32]) {
    for score in scores.iter_mut() {
        *score = ip_to_max_ip_transform(*score);
    }
}

/// Transform Faiss L2 distance to be bounded (0, 1].
#[inline]
pub fn l2_transform(l2_distance: f32) -> f32 {
    1.0 / (1.0 + l2_distance)
}

/// Bulk transform Faiss L2 distances to bounded (0, 1].
#[inline]
pub fn l2_transform_bulk(scores: &mut [f32]) {
    for score in scores.iter_mut() {
        *score = l2_transform(*score);
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Native similarity function type, mirroring the C++ enum.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSimilarityFunctionType {
    /// Max inner product for FP16.
    Fp16MaximumInnerProduct = 0,
    /// L2 for FP16.
    Fp16L2 = 1,
    /// Scalar quantized inner product.
    SqIp = 2,
    /// Scalar quantized L2.
    SqL2 = 3,
}

impl NativeSimilarityFunctionType {
    pub fn from_ordinal(ord: i32) -> Result<Self, SimdError> {
        match ord {
            0 => Ok(Self::Fp16MaximumInnerProduct),
            1 => Ok(Self::Fp16L2),
            2 => Ok(Self::SqIp),
            3 => Ok(Self::SqL2),
            _ => Err(SimdError::InvalidFunctionType(ord)),
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum SimdError {
    #[error("Invalid native similarity function type: {0}")]
    InvalidFunctionType(i32),
    #[error("Search context not initialized: mmapPages is empty")]
    NotInitialized,
    #[error("Offset [{0}] exceeds chunk size [{1}]")]
    OffsetExceedsChunkSize(u64, u64),
    #[error("Vector [vid={0}] straddles two regions ({1} and {2}), but no next region exists. Total regions: {3}")]
    NoNextRegion(i32, i32, i32, usize),
    #[error("Vector [vid={0}] straddles two regions ({1} and {2}), second part size={3} exceeds next region size={4}")]
    SecondPartExceedsRegion(i32, i32, i32, i32, u64),
    #[error("Mapped region for vector (vid={0}) was not found")]
    RegionNotFound(i32),
    #[error("Failed to allocate SIMD-aligned memory of size {0}")]
    AllocationFailed(usize),
    #[error("Null pointer encountered: {0}")]
    NullPointer(&'static str),
}

// ---------------------------------------------------------------------------
// SimdVectorSearchContext
// ---------------------------------------------------------------------------

/// Thread-local search context holding query vector, mmap page table, and
/// similarity function state.
pub struct SimdVectorSearchContext {
    /// SIMD-aligned query bytes (64-byte aligned).
    pub query_vector_simd_aligned: *mut u8,
    /// Allocated byte size for the query vector buffer.
    pub query_vector_byte_size: i32,
    /// Vector dimension.
    pub dimension: i32,
    /// Stored vector byte size (depends on quantization).
    pub one_vector_byte_size: i64,
    /// Underlying mmap page table pointers.
    pub mmap_pages: Vec<*mut u8>,
    /// Prefix-sum page sizes (cumulative end offsets).
    pub mmap_page_sizes: Vec<i64>,
    /// Function type ordinal.
    pub native_function_type_ord: i32,
    /// Selected similarity function implementation.
    pub similarity_function: Option<Box<dyn SimilarityFunction>>,
    /// Faiss distance computer (opaque FFI pointer, owned).
    pub faiss_function: *mut ffi::FaissDistanceComputer,
    /// Temporary buffer, reset per search.
    pub tmp_buffer: Vec<u8>,
}

// SAFETY: The raw pointers in SimdVectorSearchContext point to memory that is
// either thread-local or mmap'd regions whose lifetime exceeds the search.
// The struct is stored in thread_local storage and not shared across threads.
unsafe impl Send for SimdVectorSearchContext {}

impl Default for SimdVectorSearchContext {
    fn default() -> Self {
        Self {
            query_vector_simd_aligned: ptr::null_mut(),
            query_vector_byte_size: 0,
            dimension: 0,
            one_vector_byte_size: 0,
            mmap_pages: Vec::new(),
            mmap_page_sizes: Vec::new(),
            native_function_type_ord: -1,
            similarity_function: None,
            faiss_function: ptr::null_mut(),
            tmp_buffer: Vec::new(),
        }
    }
}

impl Drop for SimdVectorSearchContext {
    fn drop(&mut self) {
        if !self.query_vector_simd_aligned.is_null() {
            let rounded_size = ((self.query_vector_byte_size as usize + 63) / 64) * 64;
            if rounded_size > 0 {
                unsafe {
                    let layout = Layout::from_size_align_unchecked(rounded_size, 64);
                    alloc::dealloc(self.query_vector_simd_aligned, layout);
                }
            }
            self.query_vector_simd_aligned = ptr::null_mut();
        }
        if !self.faiss_function.is_null() {
            unsafe {
                ffi::faiss_distance_computer_free(self.faiss_function);
            }
            self.faiss_function = ptr::null_mut();
        }
    }
}

impl SimdVectorSearchContext {
    /// Look up internal mapping table and acquire raw pointers pointing to vectors
    /// with the passed vector ids, storing them into `vectors`.
    pub fn get_vector_pointers_in_bulk(
        &mut self,
        vectors: &mut [*mut u8],
        internal_vector_ids: &[i32],
        num_vectors: i32,
    ) -> Result<(), SimdError> {
        let num = num_vectors as usize;

        if self.mmap_pages.len() == 1 {
            // Fast case: single mmap area
            let base = self.mmap_pages[0];
            for i in 0..num {
                let offset = self.one_vector_byte_size as u64 * internal_vector_ids[i] as u64;
                if offset < self.mmap_page_sizes[0] as u64 {
                    vectors[i] = unsafe { base.add(offset as usize) };
                } else {
                    return Err(SimdError::OffsetExceedsChunkSize(
                        offset,
                        self.mmap_page_sizes[0] as u64,
                    ));
                }
            }
            return Ok(());
        }

        if !self.mmap_pages.is_empty() {
            for i in 0..num {
                vectors[i] = self.get_vector_pointer(internal_vector_ids[i])?;
            }
            return Ok(());
        }

        Err(SimdError::NotInitialized)
    }

    /// Get a raw pointer to the vector data for the given internal vector id.
    pub fn get_vector_pointer(&mut self, internal_vector_id: i32) -> Result<*mut u8, SimdError> {
        if self.mmap_pages.len() == 1 {
            // Fast case: single mmap area
            let offset = self.one_vector_byte_size as usize * internal_vector_id as usize;
            return Ok(unsafe { self.mmap_pages[0].add(offset) });
        }

        if !self.mmap_pages.is_empty() {
            let start_offset = self.one_vector_byte_size as u64 * internal_vector_id as u64;
            let end_offset_inclusive = start_offset + self.one_vector_byte_size as u64 - 1;

            let mut region_start_offset: u64 = 0;
            for j in 0..self.mmap_page_sizes.len() {
                if start_offset < self.mmap_page_sizes[j] as u64 {
                    // Found the first region having the vector.
                    let relative_offset = (start_offset - region_start_offset) as usize;

                    if end_offset_inclusive < self.mmap_page_sizes[j] as u64 {
                        // Entire vector is in this region.
                        return Ok(unsafe { self.mmap_pages[j].add(relative_offset) });
                    } else {
                        // Vector spans two regions - need to copy to temp buffer.
                        if (j + 1) >= self.mmap_page_sizes.len() || (j + 1) >= self.mmap_pages.len()
                        {
                            return Err(SimdError::NoNextRegion(
                                internal_vector_id,
                                j as i32,
                                (j + 1) as i32,
                                self.mmap_page_sizes.len(),
                            ));
                        }

                        // Ensure even alignment in temp buffer.
                        let padding = self.tmp_buffer.len() & 1;
                        let copy_dest_index = self.tmp_buffer.len() + padding;
                        self.tmp_buffer
                            .resize(copy_dest_index + self.one_vector_byte_size as usize, 0);

                        // Copy first part
                        let first_part_size =
                            (self.mmap_page_sizes[j] as u64 - start_offset) as usize;
                        unsafe {
                            ptr::copy_nonoverlapping(
                                self.mmap_pages[j].add(relative_offset),
                                self.tmp_buffer.as_mut_ptr().add(copy_dest_index),
                                first_part_size,
                            );
                        }

                        // Copy second part
                        let second_part_size = self.one_vector_byte_size as usize - first_part_size;
                        let next_region_size = (self.mmap_page_sizes[j + 1]
                            - self.mmap_page_sizes[j])
                            as u64;
                        if second_part_size as u64 > next_region_size {
                            return Err(SimdError::SecondPartExceedsRegion(
                                internal_vector_id,
                                j as i32,
                                (j + 1) as i32,
                                second_part_size as i32,
                                next_region_size,
                            ));
                        }
                        unsafe {
                            ptr::copy_nonoverlapping(
                                self.mmap_pages[j + 1],
                                self.tmp_buffer
                                    .as_mut_ptr()
                                    .add(copy_dest_index + first_part_size),
                                second_part_size,
                            );
                        }

                        return Ok(
                            unsafe { self.tmp_buffer.as_mut_ptr().add(copy_dest_index) }
                        );
                    }
                }
                region_start_offset = self.mmap_page_sizes[j] as u64;
            }

            return Err(SimdError::RegionNotFound(internal_vector_id));
        }

        Err(SimdError::NotInitialized)
    }
}

// ---------------------------------------------------------------------------
// Thread-local search context
// ---------------------------------------------------------------------------

thread_local! {
    static THREAD_LOCAL_SIMD_VEC_SRCH_CTX: std::cell::RefCell<SimdVectorSearchContext> =
        std::cell::RefCell::new(SimdVectorSearchContext::default());
}

// ---------------------------------------------------------------------------
// SimilarityFunction trait
// ---------------------------------------------------------------------------

/// Trait mirroring the C++ `SimilarityFunction` virtual class.
pub trait SimilarityFunction {
    /// Calculate similarity scores for multiple vectors in bulk.
    fn calculate_similarity_in_bulk(
        &self,
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_ids: &[i32],
        scores: &mut [f32],
        num_vectors: i32,
    );

    /// Calculate similarity score for a single vector.
    fn calculate_similarity(
        &self,
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_id: i32,
    ) -> f32;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FOUR_BIT_SCALE: f32 = 1.0 / 15.0;

// ---------------------------------------------------------------------------
// Helper: read data correction factors from unaligned memory
// ---------------------------------------------------------------------------

/// Reads per-vector correction factors from a potentially unaligned address.
/// On-disk layout: [lowerInterval(f32)][upperInterval(f32)][additionalCorrection(f32)][quantizedComponentSum(i32)]
#[inline]
fn read_data_corrections(ptr: *const u8) -> (f32, f32, f32, f32) {
    unsafe {
        let mut lower: f32 = 0.0;
        let mut upper: f32 = 0.0;
        let mut additional: f32 = 0.0;
        let mut component_sum: i32 = 0;

        ptr::copy_nonoverlapping(ptr, &mut lower as *mut f32 as *mut u8, 4);
        ptr::copy_nonoverlapping(ptr.add(4), &mut upper as *mut f32 as *mut u8, 4);
        ptr::copy_nonoverlapping(ptr.add(8), &mut additional as *mut f32 as *mut u8, 4);
        ptr::copy_nonoverlapping(ptr.add(12), &mut component_sum as *mut i32 as *mut u8, 4);

        let ax = lower;
        let lx = upper - lower;
        let x1 = component_sum as f32;
        (ax, lx, additional, x1)
    }
}

// ---------------------------------------------------------------------------
// Helper: int4BitDotProduct (scalar fallback)
// ---------------------------------------------------------------------------

/// Scalar int4BitDotProduct.
/// q has 4 * binary_code_bytes (4 bit planes), d has binary_code_bytes bytes.
#[inline]
fn int4_bit_dot_product(q: *const u8, d: *const u8, binary_code_bytes: i32) -> i64 {
    let mut result: i64 = 0;
    let bcb = binary_code_bytes as usize;

    for bit_plane in 0..4u32 {
        let words = bcb / 8;
        let mut sub_result: i64 = 0;

        for w in 0..words {
            let offset_q = (bit_plane as usize) * bcb + w * 8;
            let offset_d = w * 8;
            let q_word: u64 =
                unsafe { ptr::read_unaligned(q.add(offset_q) as *const u64) };
            let d_word: u64 =
                unsafe { ptr::read_unaligned(d.add(offset_d) as *const u64) };
            sub_result += (q_word & d_word).count_ones() as i64;
        }

        let remain_start = words * 8;
        for r in remain_start..bcb {
            let qb = unsafe { *q.add((bit_plane as usize) * bcb + r) };
            let db = unsafe { *d.add(r) };
            sub_result += ((qb & db) as u32).count_ones() as i64;
        }

        result += sub_result << bit_plane;
    }
    result
}

// ---------------------------------------------------------------------------
// Default (scalar) FP16 similarity function
// Uses Faiss SQDistanceComputer via FFI for actual computation.
// ---------------------------------------------------------------------------

/// Default FP16 similarity function using Faiss distance computer.
/// This is the fallback when no SIMD acceleration is available for the bulk path.
struct DefaultFp16SimilarityFunction {
    /// If true, apply IP->MaxIP transform; if false, apply L2 transform.
    is_max_ip: bool,
}

impl SimilarityFunction for DefaultFp16SimilarityFunction {
    fn calculate_similarity_in_bulk(
        &self,
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_ids: &[i32],
        scores: &mut [f32],
        num_vectors: i32,
    ) {
        let num = num_vectors as usize;
        let faiss_func = srch_context.faiss_function as *mut ffi::FaissSQDistanceComputer;
        assert!(
            !faiss_func.is_null(),
            "Faiss distance computer is null in DefaultFp16SimilarityFunction"
        );

        for i in 0..num {
            let vector_ptr = srch_context
                .get_vector_pointer(internal_vector_ids[i])
                .expect("Failed to get vector pointer");
            scores[i] = unsafe {
                ffi::faiss_sq_distance_computer_query_to_code(faiss_func, vector_ptr)
            };
        }

        if self.is_max_ip {
            ip_to_max_ip_transform_bulk(&mut scores[..num]);
        } else {
            l2_transform_bulk(&mut scores[..num]);
        }
    }

    fn calculate_similarity(
        &self,
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_id: i32,
    ) -> f32 {
        let faiss_func = srch_context.faiss_function as *mut ffi::FaissSQDistanceComputer;
        assert!(
            !faiss_func.is_null(),
            "Faiss distance computer is null in DefaultFp16SimilarityFunction"
        );

        let vector_ptr = srch_context
            .get_vector_pointer(internal_vector_id)
            .expect("Failed to get vector pointer");
        let score = unsafe {
            ffi::faiss_sq_distance_computer_query_to_code(faiss_func, vector_ptr)
        };

        if self.is_max_ip {
            ip_to_max_ip_transform(score)
        } else {
            l2_transform(score)
        }
    }
}

// ---------------------------------------------------------------------------
// SQ Similarity Function (scalar / default implementation)
// ADC: 4-bit query x 1-bit data
// ---------------------------------------------------------------------------

struct DefaultSqSimilarityFunction {
    is_max_ip: bool,
}

impl DefaultSqSimilarityFunction {
    /// Batched 4-bit dot product computation (scalar, no explicit SIMD intrinsics).
    /// The compiler may auto-vectorize this.
    fn default_4bit_dot_product_batch(
        query_ptr: *const u8,
        data_vecs: &[*mut u8],
        binary_code_bytes: i32,
        results: &mut [f32],
    ) {
        let bcb = binary_code_bytes as usize;
        let words = bcb / 8;
        let remain_start = words * 8;

        let plane0 = query_ptr;
        let plane1 = unsafe { query_ptr.add(bcb) };
        let plane2 = unsafe { query_ptr.add(2 * bcb) };
        let plane3 = unsafe { query_ptr.add(3 * bcb) };

        for (b, &data_ptr) in data_vecs.iter().enumerate() {
            let mut acc: i64 = 0;

            // 8-byte word loop
            for w in 0..words {
                let offset = w * 8;
                let d_word: u64 =
                    unsafe { ptr::read_unaligned(data_ptr.add(offset) as *const u64) };
                let q0: u64 =
                    unsafe { ptr::read_unaligned(plane0.add(offset) as *const u64) };
                let q1: u64 =
                    unsafe { ptr::read_unaligned(plane1.add(offset) as *const u64) };
                let q2: u64 =
                    unsafe { ptr::read_unaligned(plane2.add(offset) as *const u64) };
                let q3: u64 =
                    unsafe { ptr::read_unaligned(plane3.add(offset) as *const u64) };

                acc += (q0 & d_word).count_ones() as i64 * 1
                    + (q1 & d_word).count_ones() as i64 * 2
                    + (q2 & d_word).count_ones() as i64 * 4
                    + (q3 & d_word).count_ones() as i64 * 8;
            }

            // Byte remainder
            for r in remain_start..bcb {
                let db = unsafe { *data_ptr.add(r) };
                acc += unsafe {
                    ((*plane0.add(r) & db) as u32).count_ones() as i64 * 1
                        + ((*plane1.add(r) & db) as u32).count_ones() as i64 * 2
                        + ((*plane2.add(r) & db) as u32).count_ones() as i64 * 4
                        + ((*plane3.add(r) & db) as u32).count_ones() as i64 * 8
                };
            }

            results[b] = acc as f32;
        }
    }
}

impl SimilarityFunction for DefaultSqSimilarityFunction {
    fn calculate_similarity_in_bulk(
        &self,
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_ids: &[i32],
        scores: &mut [f32],
        num_vectors: i32,
    ) {
        let num = num_vectors as usize;
        let query_ptr = srch_context.query_vector_simd_aligned;
        let dim = srch_context.dimension;
        let binary_code_bytes = (dim + 7) / 8;

        // Read query correction factors from tmp_buffer
        let query_correction_ptr = srch_context.tmp_buffer.as_ptr() as *const f32;
        let (ay, ly, query_additional, y1, centroid_dp) = unsafe {
            let ay = *query_correction_ptr;
            let ly = (*query_correction_ptr.add(1) - *query_correction_ptr) * FOUR_BIT_SCALE;
            let query_additional = *query_correction_ptr.add(2);
            let y1_raw: i32 = ptr::read_unaligned(query_correction_ptr.add(3) as *const i32);
            let y1 = y1_raw as f32;
            let centroid_dp = *query_correction_ptr.add(4);
            (ay, ly, query_additional, y1, centroid_dp)
        };

        let dim_f32 = dim as f32;
        let mut processed_count: usize = 0;
        const VEC_BLOCK: usize = 8;
        const VEC_HALF_BLOCK: usize = 4;

        // Batch size 8
        while (processed_count + VEC_BLOCK) <= num {
            let mut vectors: [*mut u8; VEC_BLOCK] = [ptr::null_mut(); VEC_BLOCK];
            let ids_slice = &internal_vector_ids[processed_count..processed_count + VEC_BLOCK];

            // Get vector pointers
            for (i, &vid) in ids_slice.iter().enumerate() {
                vectors[i] = srch_context
                    .get_vector_pointer(vid)
                    .expect("Failed to get vector pointer");
            }

            Self::default_4bit_dot_product_batch(
                query_ptr,
                &vectors[..VEC_BLOCK],
                binary_code_bytes,
                &mut scores[processed_count..processed_count + VEC_BLOCK],
            );

            for i in 0..VEC_BLOCK {
                let corrections_ptr =
                    unsafe { vectors[i].add(binary_code_bytes as usize) };
                let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

                scores[processed_count + i] = ax * ay * dim_f32
                    + ay * lx * x1
                    + ax * ly * y1
                    + lx * ly * scores[processed_count + i];

                if self.is_max_ip {
                    scores[processed_count + i] +=
                        query_additional + additional - centroid_dp;
                } else {
                    scores[processed_count + i] = f32::max(
                        0.0,
                        query_additional + additional
                            - 2.0 * scores[processed_count + i],
                    );
                }
            }

            processed_count += VEC_BLOCK;
        }

        // Batch size 4
        while (processed_count + VEC_HALF_BLOCK) <= num {
            let mut vectors: [*mut u8; VEC_HALF_BLOCK] = [ptr::null_mut(); VEC_HALF_BLOCK];
            let ids_slice =
                &internal_vector_ids[processed_count..processed_count + VEC_HALF_BLOCK];

            for (i, &vid) in ids_slice.iter().enumerate() {
                vectors[i] = srch_context
                    .get_vector_pointer(vid)
                    .expect("Failed to get vector pointer");
            }

            Self::default_4bit_dot_product_batch(
                query_ptr,
                &vectors[..VEC_HALF_BLOCK],
                binary_code_bytes,
                &mut scores[processed_count..processed_count + VEC_HALF_BLOCK],
            );

            for i in 0..VEC_HALF_BLOCK {
                let corrections_ptr =
                    unsafe { vectors[i].add(binary_code_bytes as usize) };
                let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

                scores[processed_count + i] = ax * ay * dim_f32
                    + ay * lx * x1
                    + ax * ly * y1
                    + lx * ly * scores[processed_count + i];

                if self.is_max_ip {
                    scores[processed_count + i] +=
                        query_additional + additional - centroid_dp;
                } else {
                    scores[processed_count + i] = f32::max(
                        0.0,
                        query_additional + additional
                            - 2.0 * scores[processed_count + i],
                    );
                }
            }

            processed_count += VEC_HALF_BLOCK;
        }

        // Tail: remaining vectors (scalar)
        while processed_count < num {
            let data_vec = srch_context
                .get_vector_pointer(internal_vector_ids[processed_count])
                .expect("Failed to get vector pointer");
            let qc_dist =
                int4_bit_dot_product(query_ptr, data_vec, binary_code_bytes) as f32;

            let corrections_ptr =
                unsafe { data_vec.add(binary_code_bytes as usize) };
            let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

            scores[processed_count] = ax * ay * dim_f32
                + ay * lx * x1
                + ax * ly * y1
                + lx * ly * qc_dist;

            if self.is_max_ip {
                scores[processed_count] += query_additional + additional - centroid_dp;
            } else {
                scores[processed_count] = f32::max(
                    0.0,
                    query_additional + additional - 2.0 * scores[processed_count],
                );
            }

            processed_count += 1;
        }

        // Final score transform
        if self.is_max_ip {
            ip_to_max_ip_transform_bulk(&mut scores[..num]);
        } else {
            l2_transform_bulk(&mut scores[..num]);
        }
    }

    fn calculate_similarity(
        &self,
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_id: i32,
    ) -> f32 {
        let query_ptr = srch_context.query_vector_simd_aligned;
        let dim = srch_context.dimension;
        let binary_code_bytes = (dim + 7) / 8;

        let query_correction_ptr = srch_context.tmp_buffer.as_ptr() as *const f32;
        let (ay, ly, query_additional, y1, centroid_dp) = unsafe {
            let ay = *query_correction_ptr;
            let ly = (*query_correction_ptr.add(1) - *query_correction_ptr) * FOUR_BIT_SCALE;
            let query_additional = *query_correction_ptr.add(2);
            let y1_raw: i32 = ptr::read_unaligned(query_correction_ptr.add(3) as *const i32);
            let y1 = y1_raw as f32;
            let centroid_dp = *query_correction_ptr.add(4);
            (ay, ly, query_additional, y1, centroid_dp)
        };

        let data_vec = srch_context
            .get_vector_pointer(internal_vector_id)
            .expect("Failed to get vector pointer");
        let qc_dist =
            int4_bit_dot_product(query_ptr, data_vec, binary_code_bytes) as f32;

        let corrections_ptr = unsafe { data_vec.add(binary_code_bytes as usize) };
        let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

        let dim_f32 = dim as f32;
        let mut score = ax * ay * dim_f32 + ay * lx * x1 + ax * ly * y1 + lx * ly * qc_dist;

        if self.is_max_ip {
            score += query_additional + additional - centroid_dp;
            ip_to_max_ip_transform(score)
        } else {
            score = f32::max(0.0, query_additional + additional - 2.0 * score);
            l2_transform(score)
        }
    }
}

// ---------------------------------------------------------------------------
// AVX512 FP16 similarity functions (stub - requires x86_64 + avx512)
// These use SIMD intrinsics that are not portable to all architectures.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512 {
    //! AVX512 SIMD implementations for FP16 and SQ similarity functions.
    //! These require AVX512F + AVX512BW + F16C at minimum.
    use super::*;

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // -----------------------------------------------------------------------
    // Helper: AVX512 per-byte popcount using nibble LUT (vpshufb).
    // Works on all AVX512F/BW targets without requiring AVX512-VPOPCNTDQ.
    // -----------------------------------------------------------------------
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn avx512_popcnt_epi8(v: __m512i) -> __m512i {
        // Nibble popcount LUT: popcount(0..15) = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4}
        let pop_lut = _mm512_set_epi64(
            0x0403030203020201u64 as i64,
            0x0302020102010100u64 as i64,
            0x0403030203020201u64 as i64,
            0x0302020102010100u64 as i64,
            0x0403030203020201u64 as i64,
            0x0302020102010100u64 as i64,
            0x0403030203020201u64 as i64,
            0x0302020102010100u64 as i64,
        );
        let low_mask = _mm512_set1_epi8(0x0F);

        let lo = _mm512_and_si512(v, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), low_mask);
        let cnt_lo = _mm512_shuffle_epi8(pop_lut, lo);
        let cnt_hi = _mm512_shuffle_epi8(pop_lut, hi);
        _mm512_add_epi8(cnt_lo, cnt_hi)
    }

    // -----------------------------------------------------------------------
    // Helper: AVX512 batched 4-bit dot product (SQ).
    // Processes 64 bytes per iteration using LUT-based popcount.
    // -----------------------------------------------------------------------
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn avx512_4bit_dot_product_batch<const BATCH_SIZE: usize>(
        query_ptr: *const u8,
        data_vecs: &[*mut u8],
        binary_code_bytes: i32,
        results: &mut [f32],
    ) {
        let bcb = binary_code_bytes as usize;
        let plane0 = query_ptr;
        let plane1 = query_ptr.add(bcb);
        let plane2 = query_ptr.add(2 * bcb);
        let plane3 = query_ptr.add(3 * bcb);

        // 64-bit accumulators per vector
        let mut acc: [__m512i; 8] = [_mm512_setzero_si512(); 8];

        let mut i: usize = 0;
        while i + 64 <= bcb {
            let q0 = _mm512_loadu_si512(plane0.add(i) as *const i32);
            let q1 = _mm512_loadu_si512(plane1.add(i) as *const i32);
            let q2 = _mm512_loadu_si512(plane2.add(i) as *const i32);
            let q3 = _mm512_loadu_si512(plane3.add(i) as *const i32);

            for b in 0..BATCH_SIZE {
                let d = _mm512_loadu_si512(data_vecs[b].add(i) as *const i32);

                // AND each plane with data, then per-byte popcount
                let p0 = avx512_popcnt_epi8(_mm512_and_si512(q0, d));
                let p1 = avx512_popcnt_epi8(_mm512_and_si512(q1, d));
                let p2 = avx512_popcnt_epi8(_mm512_and_si512(q2, d));
                let p3 = avx512_popcnt_epi8(_mm512_and_si512(q3, d));

                // Weight: p0*1 + p1*2 + p2*4 + p3*8
                // Max per byte: 8*1 + 8*2 + 8*4 + 8*8 = 120, fits in u8
                let mut weighted = _mm512_add_epi8(p0, _mm512_slli_epi16(p1, 1));
                weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(p2, 2));
                weighted = _mm512_add_epi8(weighted, _mm512_slli_epi16(p3, 3));

                // Horizontal sum: u8 -> u64 via sad_epu8 against zero
                let sad = _mm512_sad_epu8(weighted, _mm512_setzero_si512());
                acc[b] = _mm512_add_epi64(acc[b], sad);
            }

            i += 64;
        }

        // Horizontal sum of 64-bit accumulators into results
        for b in 0..BATCH_SIZE {
            // _mm512_reduce_add_epi64 is not available in Rust stable intrinsics,
            // so we extract and sum manually.
            let mut total: i64 = 0;
            let arr: [i64; 8] = std::mem::transmute(acc[b]);
            for val in arr.iter() {
                total += val;
            }
            results[b] = total as f32;
        }

        // Scalar tail for remaining bytes (< 64)
        while i < bcb {
            let q0b = *plane0.add(i);
            let q1b = *plane1.add(i);
            let q2b = *plane2.add(i);
            let q3b = *plane3.add(i);
            for b in 0..BATCH_SIZE {
                let db = *data_vecs[b].add(i);
                results[b] += ((q0b & db).count_ones() * 1
                    + (q1b & db).count_ones() * 2
                    + (q2b & db).count_ones() * 4
                    + (q3b & db).count_ones() * 8) as f32;
            }
            i += 1;
        }
    }

    /// AVX512 FP16 Max Inner Product bulk computation.
    pub struct Avx512Fp16MaxIp;

    impl SimilarityFunction for Avx512Fp16MaxIp {
        fn calculate_similarity_in_bulk(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_ids: &[i32],
            scores: &mut [f32],
            num_vectors: i32,
        ) {
            // Safety: this module is only compiled when avx512f is enabled
            unsafe {
                avx512_fp16_max_ip_bulk(srch_context, internal_vector_ids, scores, num_vectors);
            }
        }

        fn calculate_similarity(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_id: i32,
        ) -> f32 {
            // Falls back to Faiss distance computer for single vector
            let faiss_func = srch_context.faiss_function as *mut ffi::FaissSQDistanceComputer;
            let vector_ptr = srch_context
                .get_vector_pointer(internal_vector_id)
                .expect("Failed to get vector pointer");
            let score = unsafe {
                ffi::faiss_sq_distance_computer_query_to_code(faiss_func, vector_ptr)
            };
            ip_to_max_ip_transform(score)
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,f16c")]
    unsafe fn avx512_fp16_max_ip_bulk(
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_ids: &[i32],
        scores: &mut [f32],
        num_vectors: i32,
    ) {
        let num = num_vectors as usize;
        let query_ptr = srch_context.query_vector_simd_aligned as *const f32;
        let dim = srch_context.dimension;

        const VEC_BLOCK: usize = 8;
        const ELEM_PER_LOAD: i32 = 16;

        let simd_dim = (dim / ELEM_PER_LOAD) * ELEM_PER_LOAD;
        let tail_dim = dim - simd_dim;
        let tail_mask: u16 = if tail_dim > 0 {
            (1u32 << tail_dim) as u16 - 1
        } else {
            0
        };

        let mut processed_count: usize = 0;

        // Batch of 8 vectors
        while processed_count + VEC_BLOCK <= num {
            let mut vectors: [*mut u8; VEC_BLOCK] = [ptr::null_mut(); VEC_BLOCK];
            srch_context
                .get_vector_pointers_in_bulk(
                    &mut vectors,
                    &internal_vector_ids[processed_count..],
                    VEC_BLOCK as i32,
                )
                .expect("Failed to get vector pointers in bulk");

            let mut sum: [__m512; 8] = [_mm512_setzero_ps(); 8];

            // Main loop: process 16 dimensions per iteration
            let mut i: i32 = 0;
            while i < simd_dim {
                let q0 = _mm512_loadu_ps(query_ptr.add(i as usize));

                let mut v_regs: [__m512; 8] = [_mm512_setzero_ps(); 8];
                for v in 0..VEC_BLOCK {
                    v_regs[v] = _mm512_cvtph_ps(_mm256_loadu_si256(
                        vectors[v].add((2 * i) as usize) as *const __m256i,
                    ));
                }

                for v in 0..VEC_BLOCK {
                    sum[v] = _mm512_fmadd_ps(q0, v_regs[v], sum[v]);
                }

                i += ELEM_PER_LOAD;
            }

            // Masked tail
            if tail_dim > 0 {
                let q0 = _mm512_maskz_loadu_ps(tail_mask, query_ptr.add(simd_dim as usize));

                let mut v_regs: [__m512; 8] = [_mm512_setzero_ps(); 8];
                for v in 0..VEC_BLOCK {
                    v_regs[v] = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(
                        tail_mask,
                        vectors[v].add((2 * simd_dim) as usize) as *const i32,
                    ));
                }

                for v in 0..VEC_BLOCK {
                    sum[v] = _mm512_fmadd_ps(q0, v_regs[v], sum[v]);
                }
            }

            // Reduce and store
            for v in 0..VEC_BLOCK {
                scores[processed_count + v] = _mm512_reduce_add_ps(sum[v]);
            }

            processed_count += VEC_BLOCK;
        }

        // Tail loop: remaining vectors one at a time
        while processed_count < num {
            let vec_ptr = srch_context
                .get_vector_pointer(internal_vector_ids[processed_count])
                .expect("Failed to get vector pointer");

            let mut sum_scalar = _mm512_setzero_ps();

            let mut i: i32 = 0;
            while i < simd_dim {
                let q = _mm512_loadu_ps(query_ptr.add(i as usize));
                let v = _mm512_cvtph_ps(_mm256_loadu_si256(
                    vec_ptr.add((2 * i) as usize) as *const __m256i,
                ));
                sum_scalar = _mm512_fmadd_ps(q, v, sum_scalar);
                i += ELEM_PER_LOAD;
            }

            if tail_dim > 0 {
                let q = _mm512_maskz_loadu_ps(tail_mask, query_ptr.add(simd_dim as usize));
                let v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(
                    tail_mask,
                    vec_ptr.add((2 * simd_dim) as usize) as *const i32,
                ));
                sum_scalar = _mm512_fmadd_ps(q, v, sum_scalar);
            }

            scores[processed_count] = _mm512_reduce_add_ps(sum_scalar);
            processed_count += 1;
        }

        // Apply score transform
        ip_to_max_ip_transform_bulk(&mut scores[..num]);
    }

    /// AVX512 FP16 L2 bulk computation.
    pub struct Avx512Fp16L2;

    impl SimilarityFunction for Avx512Fp16L2 {
        fn calculate_similarity_in_bulk(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_ids: &[i32],
            scores: &mut [f32],
            num_vectors: i32,
        ) {
            unsafe {
                avx512_fp16_l2_bulk(srch_context, internal_vector_ids, scores, num_vectors);
            }
        }

        fn calculate_similarity(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_id: i32,
        ) -> f32 {
            let faiss_func = srch_context.faiss_function as *mut ffi::FaissSQDistanceComputer;
            let vector_ptr = srch_context
                .get_vector_pointer(internal_vector_id)
                .expect("Failed to get vector pointer");
            let score = unsafe {
                ffi::faiss_sq_distance_computer_query_to_code(faiss_func, vector_ptr)
            };
            l2_transform(score)
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,f16c")]
    unsafe fn avx512_fp16_l2_bulk(
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_ids: &[i32],
        scores: &mut [f32],
        num_vectors: i32,
    ) {
        let num = num_vectors as usize;
        let query_ptr = srch_context.query_vector_simd_aligned as *const f32;
        let dim = srch_context.dimension;

        const VEC_BLOCK: usize = 8;
        const ELEM_PER_LOAD: i32 = 16;

        let simd_dim = (dim / ELEM_PER_LOAD) * ELEM_PER_LOAD;
        let tail_dim = dim - simd_dim;
        let tail_mask: u16 = if tail_dim > 0 {
            (1u32 << tail_dim) as u16 - 1
        } else {
            0
        };

        let mut processed_count: usize = 0;

        // Batch of 8 vectors
        while processed_count + VEC_BLOCK <= num {
            let mut vectors: [*mut u8; VEC_BLOCK] = [ptr::null_mut(); VEC_BLOCK];
            srch_context
                .get_vector_pointers_in_bulk(
                    &mut vectors,
                    &internal_vector_ids[processed_count..],
                    VEC_BLOCK as i32,
                )
                .expect("Failed to get vector pointers in bulk");

            let mut sum: [__m512; 8] = [_mm512_setzero_ps(); 8];

            // Main loop: process 16 dimensions per iteration
            let mut i: i32 = 0;
            while i < simd_dim {
                let q0 = _mm512_loadu_ps(query_ptr.add(i as usize));

                let mut v_regs: [__m512; 8] = [_mm512_setzero_ps(); 8];
                for v in 0..VEC_BLOCK {
                    v_regs[v] = _mm512_cvtph_ps(_mm256_loadu_si256(
                        vectors[v].add((2 * i) as usize) as *const __m256i,
                    ));
                }

                // L2: sum += (q - v)^2
                for v in 0..VEC_BLOCK {
                    let diff = _mm512_sub_ps(q0, v_regs[v]);
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
                }

                i += ELEM_PER_LOAD;
            }

            // Masked tail
            if tail_dim > 0 {
                let q0 = _mm512_maskz_loadu_ps(tail_mask, query_ptr.add(simd_dim as usize));

                let mut v_regs: [__m512; 8] = [_mm512_setzero_ps(); 8];
                for v in 0..VEC_BLOCK {
                    v_regs[v] = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(
                        tail_mask,
                        vectors[v].add((2 * simd_dim) as usize) as *const i32,
                    ));
                }

                for v in 0..VEC_BLOCK {
                    let diff = _mm512_sub_ps(q0, v_regs[v]);
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
                }
            }

            // Reduce and store
            for v in 0..VEC_BLOCK {
                scores[processed_count + v] = _mm512_reduce_add_ps(sum[v]);
            }

            processed_count += VEC_BLOCK;
        }

        // Tail loop: remaining vectors one at a time
        while processed_count < num {
            let vec_ptr = srch_context
                .get_vector_pointer(internal_vector_ids[processed_count])
                .expect("Failed to get vector pointer");

            let mut sum_scalar = _mm512_setzero_ps();

            let mut i: i32 = 0;
            while i < simd_dim {
                let q = _mm512_loadu_ps(query_ptr.add(i as usize));
                let v = _mm512_cvtph_ps(_mm256_loadu_si256(
                    vec_ptr.add((2 * i) as usize) as *const __m256i,
                ));
                let diff = _mm512_sub_ps(q, v);
                sum_scalar = _mm512_fmadd_ps(diff, diff, sum_scalar);
                i += ELEM_PER_LOAD;
            }

            if tail_dim > 0 {
                let q = _mm512_maskz_loadu_ps(tail_mask, query_ptr.add(simd_dim as usize));
                let v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(
                    tail_mask,
                    vec_ptr.add((2 * simd_dim) as usize) as *const i32,
                ));
                let diff = _mm512_sub_ps(q, v);
                sum_scalar = _mm512_fmadd_ps(diff, diff, sum_scalar);
            }

            scores[processed_count] = _mm512_reduce_add_ps(sum_scalar);
            processed_count += 1;
        }

        // Apply score transform
        l2_transform_bulk(&mut scores[..num]);
    }

    /// AVX512 SQ similarity function (4-bit query x 1-bit data).
    pub struct Avx512SqSimilarityFunction {
        pub is_max_ip: bool,
    }

    impl SimilarityFunction for Avx512SqSimilarityFunction {
        fn calculate_similarity_in_bulk(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_ids: &[i32],
            scores: &mut [f32],
            num_vectors: i32,
        ) {
            unsafe {
                avx512_sq_bulk(
                    self.is_max_ip,
                    srch_context,
                    internal_vector_ids,
                    scores,
                    num_vectors,
                );
            }
        }

        fn calculate_similarity(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_id: i32,
        ) -> f32 {
            // Reuse scalar implementation for single vector
            let default_sq = DefaultSqSimilarityFunction {
                is_max_ip: self.is_max_ip,
            };
            default_sq.calculate_similarity(srch_context, internal_vector_id)
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn avx512_sq_bulk(
        is_max_ip: bool,
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_ids: &[i32],
        scores: &mut [f32],
        num_vectors: i32,
    ) {
        let num = num_vectors as usize;
        let query_ptr = srch_context.query_vector_simd_aligned;
        let dim = srch_context.dimension;
        let binary_code_bytes = (dim + 7) / 8;

        // Read query correction factors from tmp_buffer
        let query_correction_ptr = srch_context.tmp_buffer.as_ptr() as *const f32;
        let ay = *query_correction_ptr;
        let ly = (*query_correction_ptr.add(1) - *query_correction_ptr) * FOUR_BIT_SCALE;
        let query_additional = *query_correction_ptr.add(2);
        let y1_raw: i32 = ptr::read_unaligned(query_correction_ptr.add(3) as *const i32);
        let y1 = y1_raw as f32;
        let centroid_dp = *query_correction_ptr.add(4);

        let dim_f32 = dim as f32;
        let mut processed_count: usize = 0;
        const VEC_BLOCK: usize = 8;
        const VEC_HALF_BLOCK: usize = 4;

        // Batch size 8
        while processed_count + VEC_BLOCK <= num {
            let mut vectors: [*mut u8; VEC_BLOCK] = [ptr::null_mut(); VEC_BLOCK];
            srch_context
                .get_vector_pointers_in_bulk(
                    &mut vectors,
                    &internal_vector_ids[processed_count..],
                    VEC_BLOCK as i32,
                )
                .expect("Failed to get vector pointers in bulk");

            avx512_4bit_dot_product_batch::<VEC_BLOCK>(
                query_ptr,
                &vectors[..VEC_BLOCK],
                binary_code_bytes,
                &mut scores[processed_count..],
            );

            for i in 0..VEC_BLOCK {
                let corrections_ptr = vectors[i].add(binary_code_bytes as usize);
                let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

                scores[processed_count + i] = ax * ay * dim_f32
                    + ay * lx * x1
                    + ax * ly * y1
                    + lx * ly * scores[processed_count + i];

                if is_max_ip {
                    scores[processed_count + i] +=
                        query_additional + additional - centroid_dp;
                } else {
                    scores[processed_count + i] = f32::max(
                        0.0,
                        query_additional + additional
                            - 2.0 * scores[processed_count + i],
                    );
                }
            }

            processed_count += VEC_BLOCK;
        }

        // Batch size 4
        while processed_count + VEC_HALF_BLOCK <= num {
            let mut vectors: [*mut u8; VEC_HALF_BLOCK] = [ptr::null_mut(); VEC_HALF_BLOCK];
            srch_context
                .get_vector_pointers_in_bulk(
                    &mut vectors,
                    &internal_vector_ids[processed_count..],
                    VEC_HALF_BLOCK as i32,
                )
                .expect("Failed to get vector pointers in bulk");

            avx512_4bit_dot_product_batch::<VEC_HALF_BLOCK>(
                query_ptr,
                &vectors[..VEC_HALF_BLOCK],
                binary_code_bytes,
                &mut scores[processed_count..],
            );

            for i in 0..VEC_HALF_BLOCK {
                let corrections_ptr = vectors[i].add(binary_code_bytes as usize);
                let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

                scores[processed_count + i] = ax * ay * dim_f32
                    + ay * lx * x1
                    + ax * ly * y1
                    + lx * ly * scores[processed_count + i];

                if is_max_ip {
                    scores[processed_count + i] +=
                        query_additional + additional - centroid_dp;
                } else {
                    scores[processed_count + i] = f32::max(
                        0.0,
                        query_additional + additional
                            - 2.0 * scores[processed_count + i],
                    );
                }
            }

            processed_count += VEC_HALF_BLOCK;
        }

        // Tail: remaining vectors (scalar)
        while processed_count < num {
            let data_vec = srch_context
                .get_vector_pointer(internal_vector_ids[processed_count])
                .expect("Failed to get vector pointer");
            let qc_dist =
                int4_bit_dot_product(query_ptr, data_vec, binary_code_bytes) as f32;

            let corrections_ptr = data_vec.add(binary_code_bytes as usize);
            let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

            scores[processed_count] = ax * ay * dim_f32
                + ay * lx * x1
                + ax * ly * y1
                + lx * ly * qc_dist;

            if is_max_ip {
                scores[processed_count] += query_additional + additional - centroid_dp;
            } else {
                scores[processed_count] = f32::max(
                    0.0,
                    query_additional + additional - 2.0 * scores[processed_count],
                );
            }

            processed_count += 1;
        }

        // Final score transform
        if is_max_ip {
            ip_to_max_ip_transform_bulk(&mut scores[..num]);
        } else {
            l2_transform_bulk(&mut scores[..num]);
        }
    }
}

// ---------------------------------------------------------------------------
// NEON FP16 similarity functions (stub - requires aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    //! ARM NEON SIMD implementations for FP16 and SQ similarity functions.
    //! Uses NEON intrinsics for FP16->FP32 conversion, FMA, and popcount.
    use super::*;

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    // -----------------------------------------------------------------------
    // Helper: NEON batched 4-bit dot product (SQ).
    // Processes 16 bytes per iteration using vcntq_u8 for per-byte popcount.
    // -----------------------------------------------------------------------
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn neon_4bit_dot_product_batch<const BATCH_SIZE: usize>(
        query_ptr: *const u8,
        data_vecs: &[*mut u8],
        binary_code_bytes: i32,
        results: &mut [f32],
    ) {
        let bcb = binary_code_bytes as usize;
        let plane0 = query_ptr;
        let plane1 = query_ptr.add(bcb);
        let plane2 = query_ptr.add(2 * bcb);
        let plane3 = query_ptr.add(3 * bcb);

        // u32x4 accumulators per vector (max BATCH_SIZE = 8)
        let mut acc: [uint32x4_t; 8] = [vdupq_n_u32(0); 8];

        let mut i: usize = 0;
        while i + 16 <= bcb {
            // Load 16 bytes from each query plane
            let q0 = vld1q_u8(plane0.add(i));
            let q1 = vld1q_u8(plane1.add(i));
            let q2 = vld1q_u8(plane2.add(i));
            let q3 = vld1q_u8(plane3.add(i));

            for b in 0..BATCH_SIZE {
                // Load 16 bytes of data vector's binary code
                let d = vld1q_u8(data_vecs[b].add(i));

                // AND each plane with data, then per-byte popcount
                let p0 = vcntq_u8(vandq_u8(q0, d));
                let p1 = vcntq_u8(vandq_u8(q1, d));
                let p2 = vcntq_u8(vandq_u8(q2, d));
                let p3 = vcntq_u8(vandq_u8(q3, d));

                // Weight: p0*1 + p1*2 + p2*4 + p3*8
                // Max per byte: 8*1 + 8*2 + 8*4 + 8*8 = 120, fits in u8
                let mut weighted = vaddq_u8(p0, vshlq_n_u8(p1, 1));
                weighted = vaddq_u8(weighted, vshlq_n_u8(p2, 2));
                weighted = vaddq_u8(weighted, vshlq_n_u8(p3, 3));

                // Widen and accumulate: u8 -> u16 -> u32
                acc[b] = vaddq_u32(acc[b], vpaddlq_u16(vpaddlq_u8(weighted)));
            }

            i += 16;
        }

        // Horizontal sum into results
        for b in 0..BATCH_SIZE {
            results[b] = vaddvq_u32(acc[b]) as f32;
        }

        // Scalar tail for remaining bytes (< 16)
        while i < bcb {
            let q0b = *plane0.add(i);
            let q1b = *plane1.add(i);
            let q2b = *plane2.add(i);
            let q3b = *plane3.add(i);
            for b in 0..BATCH_SIZE {
                let db = *data_vecs[b].add(i);
                results[b] += ((q0b & db).count_ones() * 1
                    + (q1b & db).count_ones() * 2
                    + (q2b & db).count_ones() * 4
                    + (q3b & db).count_ones() * 8) as f32;
            }
            i += 1;
        }
    }

    /// NEON FP16 Max Inner Product bulk computation.
    pub struct NeonFp16MaxIp;

    impl SimilarityFunction for NeonFp16MaxIp {
        fn calculate_similarity_in_bulk(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_ids: &[i32],
            scores: &mut [f32],
            num_vectors: i32,
        ) {
            unsafe {
                neon_fp16_max_ip_bulk(srch_context, internal_vector_ids, scores, num_vectors);
            }
        }

        fn calculate_similarity(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_id: i32,
        ) -> f32 {
            let faiss_func = srch_context.faiss_function as *mut ffi::FaissSQDistanceComputer;
            let vector_ptr = srch_context
                .get_vector_pointer(internal_vector_id)
                .expect("Failed to get vector pointer");
            let score = unsafe {
                ffi::faiss_sq_distance_computer_query_to_code(faiss_func, vector_ptr)
            };
            ip_to_max_ip_transform(score)
        }
    }

    /// Load 16 bytes (8 FP16 values) from ptr, reinterpret as float16x8_t,
    /// and convert to two float32x4_t (low and high halves).
    #[inline]
    #[target_feature(enable = "neon,fp16")]
    unsafe fn load_fp16x8_to_f32(ptr: *const u8) -> (float32x4_t, float32x4_t) {
        // Load 16 raw bytes and transmute to float16x8_t
        let raw: float16x8_t = std::mem::transmute(vld1q_u8(ptr));
        let lo = vcvt_f32_f16(vget_low_f16(raw));
        let hi = vcvt_f32_f16(vget_high_f16(raw));
        (lo, hi)
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn neon_fp16_max_ip_bulk(
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_ids: &[i32],
        scores: &mut [f32],
        num_vectors: i32,
    ) {
        let num = num_vectors as usize;
        let query_ptr = srch_context.query_vector_simd_aligned as *const f32;
        let dim = srch_context.dimension;

        const VEC_BLOCK: usize = 4;
        const DIM_BATCH: i32 = 8;

        let mut processed_count: usize = 0;

        // Batch of 4 vectors
        while processed_count + VEC_BLOCK <= num {
            let mut vectors: [*mut u8; VEC_BLOCK] = [ptr::null_mut(); VEC_BLOCK];
            srch_context
                .get_vector_pointers_in_bulk(
                    &mut vectors,
                    &internal_vector_ids[processed_count..],
                    VEC_BLOCK as i32,
                )
                .expect("Failed to get vector pointers in bulk");

            // Score accumulators
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            // Batch inner product: 8 dimensions per iteration
            let mut i: i32 = 0;
            while i + DIM_BATCH <= dim {
                // Load 8 FP32 query elements (2 x float32x4)
                let q0 = vld1q_f32(query_ptr.add(i as usize));
                let q1 = vld1q_f32(query_ptr.add((i + 4) as usize));

                // Load 8 FP16 elements from each vector, convert to FP32
                // Each FP16 element is 2 bytes, so byte offset = i * 2
                let (d0_lo, d0_hi) = load_fp16x8_to_f32(vectors[0].add((i * 2) as usize));
                let (d1_lo, d1_hi) = load_fp16x8_to_f32(vectors[1].add((i * 2) as usize));
                let (d2_lo, d2_hi) = load_fp16x8_to_f32(vectors[2].add((i * 2) as usize));
                let (d3_lo, d3_hi) = load_fp16x8_to_f32(vectors[3].add((i * 2) as usize));

                // FMA: acc += q * d
                acc0 = vfmaq_f32(acc0, q0, d0_lo);
                acc0 = vfmaq_f32(acc0, q1, d0_hi);
                acc1 = vfmaq_f32(acc1, q0, d1_lo);
                acc1 = vfmaq_f32(acc1, q1, d1_hi);
                acc2 = vfmaq_f32(acc2, q0, d2_lo);
                acc2 = vfmaq_f32(acc2, q1, d2_hi);
                acc3 = vfmaq_f32(acc3, q0, d3_lo);
                acc3 = vfmaq_f32(acc3, q1, d3_hi);

                i += DIM_BATCH;
            }

            // Horizontal sum
            scores[processed_count] = vaddvq_f32(acc0);
            scores[processed_count + 1] = vaddvq_f32(acc1);
            scores[processed_count + 2] = vaddvq_f32(acc2);
            scores[processed_count + 3] = vaddvq_f32(acc3);

            // Scalar tail for remaining dimensions
            while i < dim {
                let qv = *query_ptr.add(i as usize);
                // Read FP16 value (2 bytes) and convert to f32
                let h0_val = ptr::read_unaligned(vectors[0].add((i * 2) as usize) as *const u16);
                let h1_val = ptr::read_unaligned(vectors[1].add((i * 2) as usize) as *const u16);
                let h2_val = ptr::read_unaligned(vectors[2].add((i * 2) as usize) as *const u16);
                let h3_val = ptr::read_unaligned(vectors[3].add((i * 2) as usize) as *const u16);

                scores[processed_count] += qv * f16_to_f32(h0_val);
                scores[processed_count + 1] += qv * f16_to_f32(h1_val);
                scores[processed_count + 2] += qv * f16_to_f32(h2_val);
                scores[processed_count + 3] += qv * f16_to_f32(h3_val);
                i += 1;
            }

            processed_count += VEC_BLOCK;
        }

        // Tail loop for remaining vectors (one at a time)
        while processed_count < num {
            let vec_ptr = srch_context
                .get_vector_pointer(internal_vector_ids[processed_count])
                .expect("Failed to get vector pointer");

            let mut acc = vdupq_n_f32(0.0);
            let mut i: i32 = 0;
            while i + DIM_BATCH <= dim {
                let q0 = vld1q_f32(query_ptr.add(i as usize));
                let q1 = vld1q_f32(query_ptr.add((i + 4) as usize));
                let (d0_lo, d0_hi) = load_fp16x8_to_f32(vec_ptr.add((i * 2) as usize));
                acc = vfmaq_f32(acc, q0, d0_lo);
                acc = vfmaq_f32(acc, q1, d0_hi);
                i += DIM_BATCH;
            }

            let mut final_sum = vaddvq_f32(acc);
            // Scalar tail
            while i < dim {
                let h_val = ptr::read_unaligned(vec_ptr.add((i * 2) as usize) as *const u16);
                final_sum += *query_ptr.add(i as usize) * f16_to_f32(h_val);
                i += 1;
            }
            scores[processed_count] = final_sum;
            processed_count += 1;
        }

        // Apply score transform
        ip_to_max_ip_transform_bulk(&mut scores[..num]);
    }

    /// NEON FP16 L2 bulk computation.
    /// Note: The C++ NEON L2 implementation delegates to Faiss SQDistanceComputer
    /// for the actual distance computation, so we mirror that behavior here.
    pub struct NeonFp16L2;

    impl SimilarityFunction for NeonFp16L2 {
        fn calculate_similarity_in_bulk(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_ids: &[i32],
            scores: &mut [f32],
            num_vectors: i32,
        ) {
            // The C++ NEON L2 implementation delegates to Faiss distance computer
            // for bulk computation (unlike MaxIP which uses custom SIMD).
            let num = num_vectors as usize;
            let faiss_func = srch_context.faiss_function as *mut ffi::FaissSQDistanceComputer;
            assert!(
                !faiss_func.is_null(),
                "Faiss distance computer is null in NeonFp16L2"
            );

            for i in 0..num {
                let vector_ptr = srch_context
                    .get_vector_pointer(internal_vector_ids[i])
                    .expect("Failed to get vector pointer");
                scores[i] = unsafe {
                    ffi::faiss_sq_distance_computer_query_to_code(faiss_func, vector_ptr)
                };
            }

            l2_transform_bulk(&mut scores[..num]);
        }

        fn calculate_similarity(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_id: i32,
        ) -> f32 {
            let faiss_func = srch_context.faiss_function as *mut ffi::FaissSQDistanceComputer;
            let vector_ptr = srch_context
                .get_vector_pointer(internal_vector_id)
                .expect("Failed to get vector pointer");
            let score = unsafe {
                ffi::faiss_sq_distance_computer_query_to_code(faiss_func, vector_ptr)
            };
            l2_transform(score)
        }
    }

    /// NEON SQ similarity function.
    pub struct NeonSqSimilarityFunction {
        pub is_max_ip: bool,
    }

    impl SimilarityFunction for NeonSqSimilarityFunction {
        fn calculate_similarity_in_bulk(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_ids: &[i32],
            scores: &mut [f32],
            num_vectors: i32,
        ) {
            unsafe {
                neon_sq_bulk(
                    self.is_max_ip,
                    srch_context,
                    internal_vector_ids,
                    scores,
                    num_vectors,
                );
            }
        }

        fn calculate_similarity(
            &self,
            srch_context: &mut SimdVectorSearchContext,
            internal_vector_id: i32,
        ) -> f32 {
            let default_sq = DefaultSqSimilarityFunction {
                is_max_ip: self.is_max_ip,
            };
            default_sq.calculate_similarity(srch_context, internal_vector_id)
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn neon_sq_bulk(
        is_max_ip: bool,
        srch_context: &mut SimdVectorSearchContext,
        internal_vector_ids: &[i32],
        scores: &mut [f32],
        num_vectors: i32,
    ) {
        let num = num_vectors as usize;
        let query_ptr = srch_context.query_vector_simd_aligned;
        let dim = srch_context.dimension;
        let binary_code_bytes = (dim + 7) / 8;

        // Read query correction factors from tmp_buffer
        let query_correction_ptr = srch_context.tmp_buffer.as_ptr() as *const f32;
        let ay = *query_correction_ptr;
        let ly = (*query_correction_ptr.add(1) - *query_correction_ptr) * FOUR_BIT_SCALE;
        let query_additional = *query_correction_ptr.add(2);
        let y1_raw: i32 = ptr::read_unaligned(query_correction_ptr.add(3) as *const i32);
        let y1 = y1_raw as f32;
        let centroid_dp = *query_correction_ptr.add(4);

        let dim_f32 = dim as f32;
        let mut processed_count: usize = 0;
        const VEC_BLOCK: usize = 8;
        const VEC_HALF_BLOCK: usize = 4;

        // Batch size 8
        while processed_count + VEC_BLOCK <= num {
            let mut vectors: [*mut u8; VEC_BLOCK] = [ptr::null_mut(); VEC_BLOCK];
            srch_context
                .get_vector_pointers_in_bulk(
                    &mut vectors,
                    &internal_vector_ids[processed_count..],
                    VEC_BLOCK as i32,
                )
                .expect("Failed to get vector pointers in bulk");

            neon_4bit_dot_product_batch::<VEC_BLOCK>(
                query_ptr,
                &vectors[..VEC_BLOCK],
                binary_code_bytes,
                &mut scores[processed_count..],
            );

            for i in 0..VEC_BLOCK {
                let corrections_ptr = vectors[i].add(binary_code_bytes as usize);
                let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

                scores[processed_count + i] = ax * ay * dim_f32
                    + ay * lx * x1
                    + ax * ly * y1
                    + lx * ly * scores[processed_count + i];

                if is_max_ip {
                    scores[processed_count + i] +=
                        query_additional + additional - centroid_dp;
                } else {
                    scores[processed_count + i] = f32::max(
                        0.0,
                        query_additional + additional
                            - 2.0 * scores[processed_count + i],
                    );
                }
            }

            processed_count += VEC_BLOCK;
        }

        // Batch size 4
        while processed_count + VEC_HALF_BLOCK <= num {
            let mut vectors: [*mut u8; VEC_HALF_BLOCK] = [ptr::null_mut(); VEC_HALF_BLOCK];
            srch_context
                .get_vector_pointers_in_bulk(
                    &mut vectors,
                    &internal_vector_ids[processed_count..],
                    VEC_HALF_BLOCK as i32,
                )
                .expect("Failed to get vector pointers in bulk");

            neon_4bit_dot_product_batch::<VEC_HALF_BLOCK>(
                query_ptr,
                &vectors[..VEC_HALF_BLOCK],
                binary_code_bytes,
                &mut scores[processed_count..],
            );

            for i in 0..VEC_HALF_BLOCK {
                let corrections_ptr = vectors[i].add(binary_code_bytes as usize);
                let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

                scores[processed_count + i] = ax * ay * dim_f32
                    + ay * lx * x1
                    + ax * ly * y1
                    + lx * ly * scores[processed_count + i];

                if is_max_ip {
                    scores[processed_count + i] +=
                        query_additional + additional - centroid_dp;
                } else {
                    scores[processed_count + i] = f32::max(
                        0.0,
                        query_additional + additional
                            - 2.0 * scores[processed_count + i],
                    );
                }
            }

            processed_count += VEC_HALF_BLOCK;
        }

        // Tail: remaining vectors (scalar)
        while processed_count < num {
            let data_vec = srch_context
                .get_vector_pointer(internal_vector_ids[processed_count])
                .expect("Failed to get vector pointer");
            let qc_dist =
                int4_bit_dot_product(query_ptr, data_vec, binary_code_bytes) as f32;

            let corrections_ptr = data_vec.add(binary_code_bytes as usize);
            let (ax, lx, additional, x1) = read_data_corrections(corrections_ptr);

            scores[processed_count] = ax * ay * dim_f32
                + ay * lx * x1
                + ax * ly * y1
                + lx * ly * qc_dist;

            if is_max_ip {
                scores[processed_count] += query_additional + additional - centroid_dp;
            } else {
                scores[processed_count] = f32::max(
                    0.0,
                    query_additional + additional - 2.0 * scores[processed_count],
                );
            }

            processed_count += 1;
        }

        // Final score transform
        if is_max_ip {
            ip_to_max_ip_transform_bulk(&mut scores[..num]);
        } else {
            l2_transform_bulk(&mut scores[..num]);
        }
    }

    /// Convert a u16 IEEE 754 half-precision value to f32.
    #[inline]
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let frac = (bits & 0x3FF) as u32;

        if exp == 0 {
            if frac == 0 {
                // Zero
                f32::from_bits(sign << 31)
            } else {
                // Subnormal: value = (-1)^sign * 2^(-14) * (frac / 1024)
                let val = (frac as f32) / 1024.0 * (1.0 / 16384.0);
                if sign == 1 { -val } else { val }
            }
        } else if exp == 31 {
            // Inf or NaN
            if frac == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            // Normalized
            let f32_exp = (exp as i32 - 15 + 127) as u32;
            let f32_frac = frac << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac)
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD dispatch: select appropriate implementation based on CPU features
// ---------------------------------------------------------------------------

/// Select the appropriate similarity function implementation based on the function
/// type and available CPU features. This mirrors the C++ compile-time dispatch via
/// preprocessor macros (KNN_HAVE_AVX512, KNN_HAVE_ARM_FP16, default).
///
/// In Rust, we use runtime CPU feature detection combined with cfg attributes.
pub fn select_similarity_function(
    function_type: NativeSimilarityFunctionType,
) -> Box<dyn SimilarityFunction> {
    // On x86_64 with AVX512 available at compile time
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        match function_type {
            NativeSimilarityFunctionType::Fp16MaximumInnerProduct => {
                Box::new(avx512::Avx512Fp16MaxIp)
            }
            NativeSimilarityFunctionType::Fp16L2 => Box::new(avx512::Avx512Fp16L2),
            NativeSimilarityFunctionType::SqIp => {
                Box::new(avx512::Avx512SqSimilarityFunction { is_max_ip: true })
            }
            NativeSimilarityFunctionType::SqL2 => {
                Box::new(avx512::Avx512SqSimilarityFunction { is_max_ip: false })
            }
        }
    }

    // On aarch64 with NEON (always available on aarch64)
    #[cfg(target_arch = "aarch64")]
    {
        match function_type {
            NativeSimilarityFunctionType::Fp16MaximumInnerProduct => {
                Box::new(neon::NeonFp16MaxIp)
            }
            NativeSimilarityFunctionType::Fp16L2 => Box::new(neon::NeonFp16L2),
            NativeSimilarityFunctionType::SqIp => {
                Box::new(neon::NeonSqSimilarityFunction { is_max_ip: true })
            }
            NativeSimilarityFunctionType::SqL2 => {
                Box::new(neon::NeonSqSimilarityFunction { is_max_ip: false })
            }
        }
    }

    // Default (scalar) fallback for other architectures or when SIMD is not available
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx512f"),
        target_arch = "aarch64"
    )))]
    {
        match function_type {
            NativeSimilarityFunctionType::Fp16MaximumInnerProduct => {
                Box::new(DefaultFp16SimilarityFunction { is_max_ip: true })
            }
            NativeSimilarityFunctionType::Fp16L2 => {
                Box::new(DefaultFp16SimilarityFunction { is_max_ip: false })
            }
            NativeSimilarityFunctionType::SqIp => {
                Box::new(DefaultSqSimilarityFunction { is_max_ip: true })
            }
            NativeSimilarityFunctionType::SqL2 => {
                Box::new(DefaultSqSimilarityFunction { is_max_ip: false })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API: save_search_context / get_search_context
// Mirrors the C++ static methods on SimilarityFunction
// ---------------------------------------------------------------------------

/// Save search context into thread-local storage.
/// This prepares the query vector, mmap pages, and selects the similarity function.
///
/// # Safety
/// - `query_ptr` must point to `query_byte_size` valid bytes.
/// - `mmap_address_and_size` must point to `num_address_and_size` valid i64 values.
pub unsafe fn save_search_context(
    query_ptr: *const u8,
    query_byte_size: i32,
    dimension: i32,
    mmap_address_and_size: *const i64,
    num_address_and_size: i32,
    native_function_type_ord: i32,
) -> Result<(), SimdError> {
    let function_type = NativeSimilarityFunctionType::from_ordinal(native_function_type_ord)?;

    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.with(|cell| {
        let ctx = &mut *cell.borrow_mut();

        // Free tmp buffer
        ctx.tmp_buffer.clear();
        ctx.tmp_buffer.shrink_to_fit();

        // Allocate query vector space if needed
        if ctx.query_vector_byte_size < query_byte_size {
            let rounded_up = ((query_byte_size as usize + 63) / 64) * 64;
            let layout = Layout::from_size_align(rounded_up, 64)
                .map_err(|_| SimdError::AllocationFailed(rounded_up))?;

            // Free previously allocated space
            if !ctx.query_vector_simd_aligned.is_null() && ctx.query_vector_byte_size > 0 {
                let old_size = ((ctx.query_vector_byte_size as usize + 63) / 64) * 64;
                let old_layout = Layout::from_size_align_unchecked(old_size, 64);
                alloc::dealloc(ctx.query_vector_simd_aligned, old_layout);
            }

            let aligned_ptr = alloc::alloc(layout);
            if aligned_ptr.is_null() {
                return Err(SimdError::AllocationFailed(rounded_up));
            }

            ctx.query_vector_simd_aligned = aligned_ptr;
            ctx.query_vector_byte_size = query_byte_size;
        }

        // Copy query bytes
        ptr::copy_nonoverlapping(
            query_ptr,
            ctx.query_vector_simd_aligned,
            query_byte_size as usize,
        );

        // Set similarity function and configure based on type
        match function_type {
            NativeSimilarityFunctionType::Fp16MaximumInnerProduct => {
                ctx.similarity_function =
                    Some(select_similarity_function(function_type));
                // FP16 vector bytes = 2 * dimension
                ctx.one_vector_byte_size = 2 * dimension as i64;

                // Reset Faiss function for single vector similarity calculation
                if !ctx.faiss_function.is_null() {
                    ffi::faiss_distance_computer_free(ctx.faiss_function);
                }
                // QT_fp16 = 0, METRIC_INNER_PRODUCT = 0
                ctx.faiss_function =
                    ffi::faiss_sq_get_distance_computer(dimension as usize, 0, 0);
                ffi::faiss_distance_computer_set_query(
                    ctx.faiss_function,
                    ctx.query_vector_simd_aligned as *const f32,
                );
            }
            NativeSimilarityFunctionType::Fp16L2 => {
                ctx.similarity_function =
                    Some(select_similarity_function(function_type));
                // FP16 vector bytes = 2 * dimension
                ctx.one_vector_byte_size = 2 * dimension as i64;

                // Reset Faiss function
                if !ctx.faiss_function.is_null() {
                    ffi::faiss_distance_computer_free(ctx.faiss_function);
                }
                // QT_fp16 = 0, METRIC_L2 = 1
                ctx.faiss_function =
                    ffi::faiss_sq_get_distance_computer(dimension as usize, 0, 1);
                ffi::faiss_distance_computer_set_query(
                    ctx.faiss_function,
                    ctx.query_vector_simd_aligned as *const f32,
                );
            }
            NativeSimilarityFunctionType::SqIp | NativeSimilarityFunctionType::SqL2 => {
                ctx.similarity_function =
                    Some(select_similarity_function(function_type));
                // oneVectorByteSize = quantized vector bytes + 3 floats + 1 int (correction factors)
                // Lucene's docPackedLength for SINGLE_BIT_QUERY_NIBBLE: (dim + 7) / 8
                ctx.one_vector_byte_size = ((dimension + 7) / 8) as i64
                    + 3 * std::mem::size_of::<f32>() as i64
                    + std::mem::size_of::<i32>() as i64;
            }
        }

        // Assign native function ord number
        ctx.native_function_type_ord = native_function_type_ord;

        // Set dimension
        ctx.dimension = dimension;

        // Set mmap pages
        ctx.mmap_pages.clear();
        ctx.mmap_page_sizes.clear();
        let mut i = 0i32;
        while i < num_address_and_size {
            let addr = *mmap_address_and_size.offset(i as isize);
            let size = *mmap_address_and_size.offset((i + 1) as isize);
            ctx.mmap_pages.push(addr as *mut u8);
            ctx.mmap_page_sizes.push(size);
            i += 2;
        }

        // Build prefix sum table for page sizes
        for i in 1..ctx.mmap_page_sizes.len() {
            let prev = ctx.mmap_page_sizes[i - 1];
            ctx.mmap_page_sizes[i] += prev;
        }

        Ok(())
    })
}

/// Get the thread-local search context.
/// Returns a pointer to the thread-local SimdVectorSearchContext.
///
/// # Safety
/// The returned reference borrows from thread-local storage.
/// It must not outlive the current thread or be used across an await point.
pub fn with_search_context<F, R>(f: F) -> R
where
    F: FnOnce(&mut SimdVectorSearchContext) -> R,
{
    THREAD_LOCAL_SIMD_VEC_SRCH_CTX.with(|cell| {
        let ctx = &mut *cell.borrow_mut();
        f(ctx)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ip_to_max_ip_transform_positive() {
        assert_eq!(ip_to_max_ip_transform(1.0), 2.0);
        assert_eq!(ip_to_max_ip_transform(0.0), 1.0);
        assert_eq!(ip_to_max_ip_transform(0.5), 1.5);
    }

    #[test]
    fn test_ip_to_max_ip_transform_negative() {
        let result = ip_to_max_ip_transform(-1.0);
        assert!((result - 0.5).abs() < 1e-6);
        let result = ip_to_max_ip_transform(-3.0);
        assert!((result - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_l2_transform() {
        assert_eq!(l2_transform(0.0), 1.0);
        assert!((l2_transform(1.0) - 0.5).abs() < 1e-6);
        assert!((l2_transform(3.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_ip_to_max_ip_transform_bulk() {
        let mut scores = vec![1.0, -1.0, 0.0, 0.5, -3.0];
        ip_to_max_ip_transform_bulk(&mut scores);
        assert_eq!(scores[0], 2.0);
        assert!((scores[1] - 0.5).abs() < 1e-6);
        assert_eq!(scores[2], 1.0);
        assert_eq!(scores[3], 1.5);
        assert!((scores[4] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_l2_transform_bulk() {
        let mut scores = vec![0.0, 1.0, 3.0];
        l2_transform_bulk(&mut scores);
        assert_eq!(scores[0], 1.0);
        assert!((scores[1] - 0.5).abs() < 1e-6);
        assert!((scores[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_native_similarity_function_type_from_ordinal() {
        assert_eq!(
            NativeSimilarityFunctionType::from_ordinal(0).unwrap(),
            NativeSimilarityFunctionType::Fp16MaximumInnerProduct
        );
        assert_eq!(
            NativeSimilarityFunctionType::from_ordinal(1).unwrap(),
            NativeSimilarityFunctionType::Fp16L2
        );
        assert_eq!(
            NativeSimilarityFunctionType::from_ordinal(2).unwrap(),
            NativeSimilarityFunctionType::SqIp
        );
        assert_eq!(
            NativeSimilarityFunctionType::from_ordinal(3).unwrap(),
            NativeSimilarityFunctionType::SqL2
        );
        assert!(NativeSimilarityFunctionType::from_ordinal(99).is_err());
    }

    #[test]
    fn test_int4_bit_dot_product_basic() {
        // Simple test: all zeros
        let q = vec![0u8; 32]; // 4 planes * 8 bytes
        let d = vec![0u8; 8];
        let result = int4_bit_dot_product(q.as_ptr(), d.as_ptr(), 8);
        assert_eq!(result, 0);

        // All ones in plane0, all ones in data
        let mut q2 = vec![0u8; 32];
        for i in 0..8 {
            q2[i] = 0xFF; // plane0 all ones
        }
        let d2 = vec![0xFFu8; 8];
        let result2 = int4_bit_dot_product(q2.as_ptr(), d2.as_ptr(), 8);
        // plane0: popcount(0xFF & 0xFF) * 8 words... = 64 bits * weight 1 = 64
        assert_eq!(result2, 64);
    }

    #[test]
    fn test_read_data_corrections() {
        let mut data = vec![0u8; 16];
        // lower = 1.0
        let lower: f32 = 1.0;
        unsafe {
            ptr::copy_nonoverlapping(
                &lower as *const f32 as *const u8,
                data.as_mut_ptr(),
                4,
            );
        }
        // upper = 3.0
        let upper: f32 = 3.0;
        unsafe {
            ptr::copy_nonoverlapping(
                &upper as *const f32 as *const u8,
                data.as_mut_ptr().add(4),
                4,
            );
        }
        // additional = 5.0
        let additional: f32 = 5.0;
        unsafe {
            ptr::copy_nonoverlapping(
                &additional as *const f32 as *const u8,
                data.as_mut_ptr().add(8),
                4,
            );
        }
        // componentSum = 7
        let comp_sum: i32 = 7;
        unsafe {
            ptr::copy_nonoverlapping(
                &comp_sum as *const i32 as *const u8,
                data.as_mut_ptr().add(12),
                4,
            );
        }

        let (ax, lx, add, x1) = read_data_corrections(data.as_ptr());
        assert_eq!(ax, 1.0); // lower
        assert_eq!(lx, 2.0); // upper - lower
        assert_eq!(add, 5.0); // additional
        assert_eq!(x1, 7.0); // componentSum as f32
    }

    #[test]
    fn test_simd_vector_search_context_single_page() {
        let mut data = vec![0u8; 1024];
        // Write some known pattern
        for i in 0..1024 {
            data[i] = (i % 256) as u8;
        }

        let mut ctx = SimdVectorSearchContext::default();
        ctx.one_vector_byte_size = 16;
        ctx.mmap_pages = vec![data.as_mut_ptr()];
        ctx.mmap_page_sizes = vec![1024];

        // Get vector at id 0
        let ptr = ctx.get_vector_pointer(0).unwrap();
        assert_eq!(ptr, data.as_mut_ptr());

        // Get vector at id 2 (offset = 32)
        let ptr2 = ctx.get_vector_pointer(2).unwrap();
        assert_eq!(ptr2, unsafe { data.as_mut_ptr().add(32) });

        // Clean up - prevent double free
        ctx.mmap_pages.clear();
    }
}
