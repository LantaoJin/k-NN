// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Raw FFI declarations for the Faiss C API (libfaiss_c / libfaiss).
//!
//! We declare only the subset of functions and types actually used by the k-NN
//! JNI wrapper. Complex C++ classes are represented as opaque types -- we never
//! construct or inspect them from Rust, only pass pointers obtained from Faiss
//! factory functions.

use std::os::raw::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// Metric types
// ---------------------------------------------------------------------------

/// Faiss distance metric identifiers.
/// These mirror `faiss::MetricType` enum values.
pub type FaissMetricType = c_int;

pub const METRIC_INNER_PRODUCT: FaissMetricType = 0;
pub const METRIC_L2: FaissMetricType = 1;

// ---------------------------------------------------------------------------
// IO flags used when reading indices
// ---------------------------------------------------------------------------

pub const IO_FLAG_READ_ONLY: c_int = 1;
pub const IO_FLAG_PQ_SKIP_SDC_TABLE: c_int = 2;
pub const IO_FLAG_SKIP_PRECOMPUTE_TABLE: c_int = 16;
pub const IO_FLAG_SKIP_STORAGE: c_int = 32;

// ---------------------------------------------------------------------------
// Faiss idx_t -- the integer type used for vector IDs in Faiss
// ---------------------------------------------------------------------------

/// Faiss uses `int64_t` as its index type (`faiss::idx_t`).
pub type FaissIdx = i64;

// ---------------------------------------------------------------------------
// Opaque struct declarations
// ---------------------------------------------------------------------------
// We use empty enums to create non-instantiable opaque types, following the
// standard Rust FFI pattern. These are never constructed on the Rust side;
// we only hold `*mut` pointers returned by Faiss C functions.

/// Opaque handle for `faiss::Index` (the base float index class).
#[repr(C)]
pub struct FaissIndex { _private: [u8; 0] }

/// Opaque handle for `faiss::IndexBinary` (the base binary index class).
#[repr(C)]
pub struct FaissIndexBinary { _private: [u8; 0] }

/// Opaque handle for `faiss::IndexIDMap` (wraps an Index and adds an id mapping).
#[repr(C)]
pub struct FaissIndexIDMap { _private: [u8; 0] }

/// Opaque handle for `faiss::IndexBinaryIDMap`.
#[repr(C)]
pub struct FaissIndexBinaryIDMap { _private: [u8; 0] }

/// Opaque handle for `faiss::IndexHNSW`.
#[repr(C)]
pub struct FaissIndexHNSW { _private: [u8; 0] }

/// Opaque handle for `faiss::IndexIVF` (base class for IVF indices).
#[repr(C)]
pub struct FaissIndexIVF { _private: [u8; 0] }

/// Opaque handle for `faiss::IndexIVFPQ`.
#[repr(C)]
pub struct FaissIndexIVFPQ { _private: [u8; 0] }

/// Opaque handle for `faiss::IndexIVFFlat`.
#[repr(C)]
pub struct FaissIndexIVFFlat { _private: [u8; 0] }

/// Opaque handle for `faiss::IOReader`.
#[repr(C)]
pub struct FaissIOReader { _private: [u8; 0] }

/// Opaque handle for `faiss::IOWriter`.
#[repr(C)]
pub struct FaissIOWriter { _private: [u8; 0] }

/// Opaque handle for `faiss::VectorIOReader` (an in-memory IOReader).
#[repr(C)]
pub struct FaissVectorIOReader { _private: [u8; 0] }

/// Opaque handle for `faiss::VectorIOWriter` (an in-memory IOWriter).
#[repr(C)]
pub struct FaissVectorIOWriter { _private: [u8; 0] }

/// Opaque handle for `faiss::SearchParameters`.
#[repr(C)]
pub struct FaissSearchParameters { _private: [u8; 0] }

/// Opaque handle for `faiss::SearchParametersHNSW`.
#[repr(C)]
pub struct FaissSearchParametersHNSW { _private: [u8; 0] }

/// Opaque handle for `faiss::SearchParametersIVF`.
#[repr(C)]
pub struct FaissSearchParametersIVF { _private: [u8; 0] }

/// Opaque handle for `faiss::IDSelector`.
#[repr(C)]
pub struct FaissIDSelector { _private: [u8; 0] }

/// Opaque handle for `faiss::RangeSearchResult`.
#[repr(C)]
pub struct FaissRangeSearchResult { _private: [u8; 0] }

// ---------------------------------------------------------------------------
// extern "C" function declarations -- Faiss C API
// ---------------------------------------------------------------------------

extern "C" {
    // -----------------------------------------------------------------------
    // Index factory functions
    // -----------------------------------------------------------------------

    /// Create a float index from a description string.
    ///
    /// C API: `int faiss_index_factory(FaissIndex** p_index, int d, const char* description, FaissMetricType metric)`
    /// Returns 0 on success, non-zero on error. The index is written to *p_index.
    pub fn faiss_index_factory(
        p_index: *mut *mut FaissIndex,
        d: c_int,
        description: *const c_char,
        metric: FaissMetricType,
    ) -> c_int;

    /// Create a binary index from a description string.
    ///
    /// C API: `int faiss_index_binary_factory(FaissIndexBinary** p_index, int d, const char* description)`
    /// Returns 0 on success, non-zero on error.
    pub fn faiss_index_binary_factory(
        p_index: *mut *mut FaissIndexBinary,
        d: c_int,
        description: *const c_char,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // Index I/O -- reading
    // -----------------------------------------------------------------------

    /// Read a float index from a file path.
    ///
    /// C API: `int faiss_read_index(const char* fname, int io_flags, FaissIndex** p_out)`
    pub fn faiss_read_index(
        fname: *const c_char,
        io_flags: c_int,
        p_out: *mut *mut FaissIndex,
    ) -> c_int;

    /// Read a float index from an IOReader.
    ///
    /// Corresponds to `faiss::read_index(reader, io_flags)`.
    pub fn faiss_read_index_from_reader(
        reader: *mut FaissIOReader,
        io_flags: c_int,
    ) -> *mut FaissIndex;

    /// Read a binary index from a file path.
    ///
    /// C API: `int faiss_read_index_binary(const char* fname, int io_flags, FaissIndexBinary** p_out)`
    pub fn faiss_read_index_binary(
        fname: *const c_char,
        io_flags: c_int,
        p_out: *mut *mut FaissIndexBinary,
    ) -> c_int;

    /// Read a binary index from an IOReader.
    ///
    /// Corresponds to `faiss::read_index_binary(reader, io_flags)`.
    pub fn faiss_read_index_binary_from_reader(
        reader: *mut FaissIOReader,
        io_flags: c_int,
    ) -> *mut FaissIndexBinary;

    // -----------------------------------------------------------------------
    // Index I/O -- writing
    // -----------------------------------------------------------------------

    /// Write a float index to an IOWriter.
    ///
    /// Corresponds to `faiss::write_index(idx, writer)`.
    pub fn faiss_write_index(
        idx: *const FaissIndex,
        writer: *mut FaissIOWriter,
    );

    /// Write a float index to a file path.
    ///
    /// Corresponds to `faiss::write_index(idx, fname)`.
    pub fn faiss_write_index_to_file(
        idx: *const FaissIndex,
        fname: *const c_char,
    );

    /// Write a binary index to an IOWriter.
    ///
    /// Corresponds to `faiss::write_index_binary(idx, writer, io_flags)`.
    pub fn faiss_write_index_binary(
        idx: *const FaissIndexBinary,
        writer: *mut FaissIOWriter,
        io_flags: c_int,
    );

    // -----------------------------------------------------------------------
    // Index operations (search, train, add)
    // -----------------------------------------------------------------------

    /// Search the k nearest neighbors in a float index.
    ///
    /// Corresponds to `index->search(n, x, k, distances, labels)`.
    ///
    /// # Safety
    /// - `index` must be a valid pointer to a live Faiss index.
    /// - `x` must point to `n * d` floats.
    /// - `distances` must have space for `n * k` floats.
    /// - `labels` must have space for `n * k` idx values.
    pub fn faiss_index_search(
        index: *const FaissIndex,
        n: FaissIdx,
        x: *const f32,
        k: FaissIdx,
        distances: *mut f32,
        labels: *mut FaissIdx,
    );

    /// Search the k nearest neighbors with custom search parameters.
    ///
    /// Corresponds to `index->search(n, x, k, distances, labels, params)`.
    pub fn faiss_index_search_with_params(
        index: *const FaissIndex,
        n: FaissIdx,
        x: *const f32,
        k: FaissIdx,
        distances: *mut f32,
        labels: *mut FaissIdx,
        params: *const FaissSearchParameters,
    );

    /// Range search on a float index.
    ///
    /// Corresponds to `index->range_search(n, x, radius, result)`.
    pub fn faiss_index_range_search(
        index: *const FaissIndex,
        n: FaissIdx,
        x: *const f32,
        radius: f32,
        result: *mut FaissRangeSearchResult,
    );

    /// Train a float index with training vectors.
    ///
    /// Corresponds to `index->train(n, x)`.
    pub fn faiss_index_train(
        index: *mut FaissIndex,
        n: FaissIdx,
        x: *const f32,
    );

    /// Add vectors with IDs to a float index.
    ///
    /// Corresponds to `index->add_with_ids(n, x, xids)`.
    pub fn faiss_index_add_with_ids(
        index: *mut FaissIndex,
        n: FaissIdx,
        x: *const f32,
        xids: *const FaissIdx,
    );

    /// Search the k nearest neighbors in a binary index.
    ///
    /// Corresponds to `index->search(n, x, k, distances, labels)`.
    pub fn faiss_index_binary_search(
        index: *const FaissIndexBinary,
        n: FaissIdx,
        x: *const u8,
        k: FaissIdx,
        distances: *mut i32,
        labels: *mut FaissIdx,
    );

    /// Train a binary index.
    ///
    /// Corresponds to `index->train(n, x)`.
    pub fn faiss_index_binary_train(
        index: *mut FaissIndexBinary,
        n: FaissIdx,
        x: *const u8,
    );

    /// Add vectors with IDs to a binary index.
    ///
    /// Corresponds to `index->add_with_ids(n, x, xids)`.
    pub fn faiss_index_binary_add_with_ids(
        index: *mut FaissIndexBinary,
        n: FaissIdx,
        x: *const u8,
        xids: *const FaissIdx,
    );

    // -----------------------------------------------------------------------
    // Index property accessors
    // -----------------------------------------------------------------------

    /// Check if an index is trained.
    ///
    /// Corresponds to `index->is_trained`.
    pub fn faiss_index_is_trained(index: *const FaissIndex) -> c_int;

    /// Check if a binary index is trained.
    pub fn faiss_index_binary_is_trained(index: *const FaissIndexBinary) -> c_int;

    /// Get the dimension of a float index.
    pub fn faiss_index_d(index: *const FaissIndex) -> c_int;

    /// Get the metric type of a float index.
    pub fn faiss_index_metric_type(index: *const FaissIndex) -> FaissMetricType;

    // -----------------------------------------------------------------------
    // IndexIDMap construction
    // -----------------------------------------------------------------------

    /// Create an IndexIDMap wrapping a float index.
    ///
    /// Corresponds to `new faiss::IndexIDMap(index)`.
    /// The returned pointer owns the wrapper but NOT the underlying index
    /// (unless own_fields is set).
    pub fn faiss_index_id_map_new(index: *mut FaissIndex) -> *mut FaissIndexIDMap;

    /// Create an IndexBinaryIDMap wrapping a binary index.
    pub fn faiss_index_binary_id_map_new(
        index: *mut FaissIndexBinary,
    ) -> *mut FaissIndexBinaryIDMap;

    // -----------------------------------------------------------------------
    // Index destruction
    // -----------------------------------------------------------------------

    /// Free a float index and all owned resources.
    ///
    /// Corresponds to `delete index`.
    pub fn faiss_index_free(index: *mut FaissIndex);

    /// Free a binary index and all owned resources.
    pub fn faiss_index_binary_free(index: *mut FaissIndexBinary);

    // -----------------------------------------------------------------------
    // Downcasting / dynamic type checks (via Faiss C helpers)
    // -----------------------------------------------------------------------
    // Note: The actual libfaiss C API may name these differently. These
    // declarations represent the logical operations needed; the link-time
    // symbol names may need adjustment via `#[link_name = "..."]` attributes
    // when binding to a specific Faiss C build.

    /// Attempt to downcast a FaissIndex to FaissIndexHNSW.
    /// Returns null if the index is not an HNSW index.
    pub fn faiss_index_to_hnsw(index: *mut FaissIndex) -> *mut FaissIndexHNSW;

    /// Attempt to downcast a FaissIndex to FaissIndexIVF.
    /// Returns null if the index is not an IVF index.
    pub fn faiss_index_to_ivf(index: *mut FaissIndex) -> *mut FaissIndexIVF;

    /// Attempt to downcast a FaissIndex to FaissIndexIVFPQ.
    /// Returns null if the index is not an IVFPQ index.
    pub fn faiss_index_to_ivfpq(index: *mut FaissIndex) -> *mut FaissIndexIVFPQ;

    // -----------------------------------------------------------------------
    // HNSW parameters
    // -----------------------------------------------------------------------

    /// Set efSearch on an HNSW index.
    pub fn faiss_index_hnsw_set_ef_search(
        index: *mut FaissIndexHNSW,
        ef_search: c_int,
    );

    /// Set efConstruction on an HNSW index.
    pub fn faiss_index_hnsw_set_ef_construction(
        index: *mut FaissIndexHNSW,
        ef_construction: c_int,
    );

    /// Get efSearch from an HNSW index.
    pub fn faiss_index_hnsw_get_ef_search(
        index: *const FaissIndexHNSW,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // IVF parameters
    // -----------------------------------------------------------------------

    /// Set nprobe on an IVF index.
    pub fn faiss_index_ivf_set_nprobe(
        index: *mut FaissIndexIVF,
        nprobe: c_int,
    );

    // -----------------------------------------------------------------------
    // VectorIOReader / VectorIOWriter helpers
    // These are used for in-memory serialization (template index from bytes).
    // -----------------------------------------------------------------------

    /// Create a new empty VectorIOReader.
    pub fn faiss_vector_io_reader_new() -> *mut FaissVectorIOReader;

    /// Set the data buffer of a VectorIOReader.
    /// The reader does NOT take ownership of `data`.
    pub fn faiss_vector_io_reader_set_data(
        reader: *mut FaissVectorIOReader,
        data: *const u8,
        size: usize,
    );

    /// Free a VectorIOReader.
    pub fn faiss_vector_io_reader_free(reader: *mut FaissVectorIOReader);

    /// Create a new empty VectorIOWriter.
    pub fn faiss_vector_io_writer_new() -> *mut FaissVectorIOWriter;

    /// Get the written data from a VectorIOWriter.
    /// Returns a pointer to the internal buffer and sets `size` to its length.
    pub fn faiss_vector_io_writer_get_data(
        writer: *const FaissVectorIOWriter,
        data: *mut *const u8,
        size: *mut usize,
    );

    /// Free a VectorIOWriter.
    pub fn faiss_vector_io_writer_free(writer: *mut FaissVectorIOWriter);

    // -----------------------------------------------------------------------
    // Read from VectorIOReader (casted to IOReader)
    // -----------------------------------------------------------------------

    /// Read a float index from a VectorIOReader.
    /// The VectorIOReader is a subtype of IOReader, so this reuses the reader
    /// overload. Provided as a convenience binding.
    pub fn faiss_read_index_from_vector_io_reader(
        reader: *mut FaissVectorIOReader,
        io_flags: c_int,
    ) -> *mut FaissIndex;

    /// Read a binary index from a VectorIOReader.
    pub fn faiss_read_index_binary_from_vector_io_reader(
        reader: *mut FaissVectorIOReader,
        io_flags: c_int,
    ) -> *mut FaissIndexBinary;

    // -----------------------------------------------------------------------
    // Write to VectorIOWriter
    // -----------------------------------------------------------------------

    /// Write a float index to a VectorIOWriter.
    pub fn faiss_write_index_to_vector_io_writer(
        idx: *const FaissIndex,
        writer: *mut FaissVectorIOWriter,
    );

    /// Write a binary index to a VectorIOWriter.
    pub fn faiss_write_index_binary_to_vector_io_writer(
        idx: *const FaissIndexBinary,
        writer: *mut FaissVectorIOWriter,
    );
}

// ---------------------------------------------------------------------------
// OpenMP thread control
// ---------------------------------------------------------------------------

extern "C" {
    /// Set the number of threads for OpenMP parallel regions.
    ///
    /// Used to control parallelism during index creation and to limit threads
    /// to 1 during search (since OpenSearch manages its own thread pool).
    pub fn omp_set_num_threads(num_threads: c_int);

    /// Get the current maximum number of OpenMP threads.
    pub fn omp_get_max_threads() -> c_int;
}

// ---------------------------------------------------------------------------
// Helper: cast opaque pointer types for IDMap usage
// ---------------------------------------------------------------------------
// The IndexIDMap wraps an Index, and the C API often requires passing it
// as a plain Index pointer for search/train. These helpers provide safe
// (well, as safe as FFI gets) conversions.

impl FaissIndexIDMap {
    /// Treat this IndexIDMap as a generic FaissIndex for search operations.
    ///
    /// # Safety
    /// The pointer must be valid and non-null.
    #[inline]
    pub unsafe fn as_index(ptr: *mut Self) -> *mut FaissIndex {
        ptr as *mut FaissIndex
    }
}

impl FaissIndexBinaryIDMap {
    /// Treat this IndexBinaryIDMap as a generic FaissIndexBinary.
    ///
    /// # Safety
    /// The pointer must be valid and non-null.
    #[inline]
    pub unsafe fn as_index_binary(ptr: *mut Self) -> *mut FaissIndexBinary {
        ptr as *mut FaissIndexBinary
    }
}
