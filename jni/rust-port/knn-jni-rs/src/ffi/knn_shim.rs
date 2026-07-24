// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! FFI declarations for the C++ knn_shim library (csrc/knn_shim.cpp).
//!
//! This shim provides C-callable wrappers around C++ operations that require
//! RTTI (dynamic_cast), C++ virtual classes, and template specializations
//! that cannot be performed through pure Rust FFI.

use std::ffi::c_void;
use std::os::raw::c_int;

// ---------------------------------------------------------------------------
// Callback types matching the C++ typedefs in knn_shim.cpp
// ---------------------------------------------------------------------------

/// Read callback: `size_t (*)(void* ctx, void* dest, size_t nbytes)`
pub type ReadCb = unsafe extern "C" fn(ctx: *mut c_void, dest: *mut c_void, nbytes: usize) -> usize;

/// Write callback: `size_t (*)(void* ctx, const void* src, size_t nbytes)`
pub type WriteCb = unsafe extern "C" fn(ctx: *mut c_void, src: *const c_void, nbytes: usize) -> usize;

/// Flush callback: `void (*)(void* ctx)`
pub type FlushCb = unsafe extern "C" fn(ctx: *mut c_void);

// ---------------------------------------------------------------------------
// extern "C" declarations for knn_shim.cpp
// ---------------------------------------------------------------------------

extern "C" {
    // -----------------------------------------------------------------------
    // 1. RTTI helpers: detect index types and set parameters
    // -----------------------------------------------------------------------

    /// Check if the underlying index (unwrapping IndexIDMap) is IndexHNSW.
    /// Returns 1 if true, 0 otherwise.
    pub fn knn_shim_is_index_hnsw(index_ptr: *mut c_void) -> c_int;

    /// Check if the underlying index (unwrapping IndexIDMap) is IndexIVF.
    /// Returns 1 if true, 0 otherwise.
    pub fn knn_shim_is_index_ivf(index_ptr: *mut c_void) -> c_int;

    /// Check if the underlying index is IndexIVFPQ with L2 metric.
    /// Returns 1 if true, 0 otherwise.
    pub fn knn_shim_is_index_ivfpq_l2(index_ptr: *mut c_void) -> c_int;

    /// Set efConstruction on an HNSW index (unwraps IndexIDMap).
    pub fn knn_shim_set_hnsw_ef_construction(index_ptr: *mut c_void, ef_construction: c_int);

    /// Set efSearch on an HNSW index (unwraps IndexIDMap).
    pub fn knn_shim_set_hnsw_ef_search(index_ptr: *mut c_void, ef_search: c_int);

    /// Set nprobe on an IVF index (unwraps IndexIDMap).
    pub fn knn_shim_set_ivf_nprobe(index_ptr: *mut c_void, nprobe: c_int);

    /// Get efSearch from an HNSW index. Returns -1 if not HNSW.
    pub fn knn_shim_get_hnsw_ef_search(index_ptr: *mut c_void) -> c_int;

    /// Get nprobe from an IVF index. Returns -1 if not IVF.
    pub fn knn_shim_get_ivf_nprobe(index_ptr: *mut c_void) -> c_int;

    // -----------------------------------------------------------------------
    // 2. IVFPQ precomputed table
    // -----------------------------------------------------------------------

    /// Initialize the IVFPQ precomputed table. Returns a pointer to the
    /// allocated AlignedTable<float> (as void*), or nullptr on failure.
    pub fn knn_shim_init_ivfpq_precomputed_table(index_ptr: *mut c_void) -> *mut c_void;

    /// Set the precomputed table on an IVFPQ index.
    pub fn knn_shim_set_ivfpq_precomputed_table(index_ptr: *mut c_void, table_ptr: *mut c_void);

    /// Free an IVFPQ precomputed table.
    pub fn knn_shim_free_ivfpq_precomputed_table(table_ptr: *mut c_void);

    // -----------------------------------------------------------------------
    // 3. Faiss IOReader/IOWriter from callbacks
    // -----------------------------------------------------------------------

    /// Create a Faiss IOReader from a read callback + context pointer.
    pub fn knn_shim_create_io_reader(ctx: *mut c_void, read_fn: ReadCb) -> *mut c_void;

    /// Create a Faiss IOWriter from write/flush callbacks + context pointer.
    pub fn knn_shim_create_io_writer(
        ctx: *mut c_void,
        write_fn: WriteCb,
        flush_fn: FlushCb,
    ) -> *mut c_void;

    /// Free an IOReader created by knn_shim_create_io_reader.
    pub fn knn_shim_free_io_reader(reader_ptr: *mut c_void);

    /// Free an IOWriter created by knn_shim_create_io_writer.
    pub fn knn_shim_free_io_writer(writer_ptr: *mut c_void);

    /// Flush an IOWriter (calls the flush callback).
    pub fn knn_shim_flush_io_writer(writer_ptr: *mut c_void);

    /// Read an index from an IOReader. Returns the index pointer (caller owns).
    pub fn knn_shim_read_index_from_reader(reader_ptr: *mut c_void, io_flags: c_int) -> *mut c_void;

    /// Read a binary index from an IOReader.
    pub fn knn_shim_read_binary_index_from_reader(reader_ptr: *mut c_void, io_flags: c_int) -> *mut c_void;

    /// Write an index to an IOWriter.
    pub fn knn_shim_write_index_to_writer(index_ptr: *mut c_void, writer_ptr: *mut c_void);

    /// Write a binary index to an IOWriter.
    pub fn knn_shim_write_binary_index_to_writer(index_ptr: *mut c_void, writer_ptr: *mut c_void);

    // -----------------------------------------------------------------------
    // 4. NMSLIB stream support
    // -----------------------------------------------------------------------

    /// Create an ostream backed by a write callback (for NMSLIB SaveIndex).
    /// Returns a wrapper pointer (use knn_shim_get_ostream_ptr to get raw ostream*).
    pub fn knn_shim_create_ostream(ctx: *mut c_void, write_fn: WriteCb) -> *mut c_void;

    /// Get the raw ostream pointer from the wrapper (for passing to NMSLIB).
    pub fn knn_shim_get_ostream_ptr(wrapper_ptr: *mut c_void) -> *mut c_void;

    /// Free an ostream created by knn_shim_create_ostream.
    pub fn knn_shim_free_ostream(wrapper_ptr: *mut c_void);

    /// Create an istream backed by a read callback (for NMSLIB LoadIndex).
    /// Returns a wrapper pointer (use knn_shim_get_istream_ptr to get raw istream*).
    pub fn knn_shim_create_istream(ctx: *mut c_void, read_fn: ReadCb) -> *mut c_void;

    /// Get the raw istream pointer from the wrapper.
    pub fn knn_shim_get_istream_ptr(wrapper_ptr: *mut c_void) -> *mut c_void;

    /// Free an istream created by knn_shim_create_istream.
    pub fn knn_shim_free_istream(wrapper_ptr: *mut c_void);

    // -----------------------------------------------------------------------
    // 5. ADC index loading
    // -----------------------------------------------------------------------

    /// Load an ADC (Asymmetric Distance Computation) index from an IOReader.
    /// Returns an index pointer or nullptr on failure.
    pub fn knn_shim_load_index_adc(io_reader_ptr: *mut c_void, metric_type: c_int) -> *mut c_void;

    // -----------------------------------------------------------------------
    // 6. Merge interrupt callback
    // -----------------------------------------------------------------------

    /// Set the Faiss abort callback that checks Java's MergeAbortChecker.
    pub fn knn_shim_set_merge_interrupt_callback(jni_env_ptr: *mut c_void);

    // -----------------------------------------------------------------------
    // 7. SQ index operations
    // -----------------------------------------------------------------------

    /// Initialize a Faiss SQ index. Returns the index memory address as jlong.
    pub fn knn_shim_init_sq_index(
        total_live_docs: i32,
        dim: i32,
        centroid_dp: f32,
        quantized_vec_bytes: i32,
    ) -> i64;

    /// Add documents to an SQ index.
    pub fn knn_shim_sq_add_docs(
        index_memory_address: i64,
        num_docs: i32,
        num_added: i32,
    );

    /// Pass quantized vectors with correction factors to an SQ index.
    pub fn knn_shim_sq_pass_vectors(
        index_memory_address: i64,
        buffer: *const u8,
        num_elements: i32,
    );

    /// Release (free) an SQ index.
    pub fn knn_shim_sq_release_index(index_memory_address: i64);
}
