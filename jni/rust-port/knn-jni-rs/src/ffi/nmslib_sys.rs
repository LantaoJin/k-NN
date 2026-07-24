// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Raw FFI declarations for NMSLIB (Non-Metric Space Library).
//!
//! NMSLIB is a C++ library with no official C API, so these declarations
//! correspond to a thin C wrapper that must be provided (or linked against
//! the C++ symbols with appropriate mangling). The k-NN plugin uses NMSLIB
//! exclusively with the HNSW method and float vectors.
//!
//! All NMSLIB C++ types are represented as opaque `*mut c_void` pointers
//! since we never inspect their internals from Rust.

use std::os::raw::{c_char, c_float, c_int, c_void};

// ---------------------------------------------------------------------------
// Opaque type aliases
// ---------------------------------------------------------------------------
// NMSLIB has deeply templated C++ types (Space<float>, Index<float>, etc.)
// that cannot be represented as Rust structs. We use `c_void` pointers and
// give them semantic aliases for readability.

/// Opaque handle to an NMSLIB space (e.g., `similarity::Space<float>`).
pub type NmslibSpace = c_void;

/// Opaque handle to an NMSLIB index (e.g., `similarity::Index<float>`).
pub type NmslibIndex = c_void;

/// Opaque handle to an NMSLIB query object (e.g., `similarity::KNNQuery<float>`).
pub type NmslibQuery = c_void;

/// Opaque handle to an NMSLIB query result / priority queue
/// (e.g., `similarity::KNNQueue<float>`).
pub type NmslibResult = c_void;

/// Opaque handle to an NMSLIB Object (data point).
pub type NmslibObject = c_void;

// ---------------------------------------------------------------------------
// extern "C" function declarations -- NMSLIB C wrapper
// ---------------------------------------------------------------------------
// These functions correspond to the operations performed in
// `nmslib_wrapper.cpp`. A C shim library must export these symbols.

extern "C" {
    // -----------------------------------------------------------------------
    // Library initialization
    // -----------------------------------------------------------------------

    /// Initialize the NMSLIB library. Must be called once before any other
    /// NMSLIB operations.
    ///
    /// Corresponds to `similarity::initLibrary()`.
    pub fn nmslib_init_library();

    // -----------------------------------------------------------------------
    // Space management
    // -----------------------------------------------------------------------

    /// Create a float space by name (e.g., "l2", "cosinesimil", "negdotprod").
    ///
    /// Corresponds to:
    /// ```cpp
    /// similarity::SpaceFactoryRegistry<float>::Instance()
    ///     .CreateSpace(space_name, similarity::AnyParams())
    /// ```
    ///
    /// Returns an opaque pointer to the created space. The caller owns it.
    pub fn nmslib_create_space(space_name: *const c_char) -> *mut NmslibSpace;

    /// Free a previously created space.
    pub fn nmslib_free_space(space: *mut NmslibSpace);

    // -----------------------------------------------------------------------
    // Index creation and lifecycle
    // -----------------------------------------------------------------------

    /// Create an HNSW index for float vectors in the given space.
    ///
    /// Corresponds to:
    /// ```cpp
    /// similarity::MethodFactoryRegistry<float>::Instance()
    ///     .CreateMethod(false, "hnsw", space_name, *space, data)
    /// ```
    ///
    /// `space` must be a valid space pointer. The returned index is owned by
    /// the caller.
    pub fn nmslib_create_index(
        space: *mut NmslibSpace,
        space_name: *const c_char,
    ) -> *mut NmslibIndex;

    /// Build (create) the HNSW graph with the given parameters.
    ///
    /// `params` is a null-terminated array of null-terminated C strings,
    /// each in the format "key=value" (e.g., "efConstruction=512", "M=16",
    /// "indexThreadQty=4").
    /// `num_params` is the number of parameter strings.
    ///
    /// Corresponds to `index->CreateIndex(similarity::AnyParams(params))`.
    pub fn nmslib_build_index(
        index: *mut NmslibIndex,
        params: *const *const c_char,
        num_params: c_int,
    );

    /// Add a single data point to the index's dataset before building.
    ///
    /// `id` is the integer ID for this vector.
    /// `vector` points to `dim` floats.
    ///
    /// This must be called for all vectors before `nmslib_build_index`.
    pub fn nmslib_add_data_point(
        index: *mut NmslibIndex,
        id: c_int,
        vector: *const c_float,
        dim: c_int,
    );

    // -----------------------------------------------------------------------
    // Index persistence (save / load)
    // -----------------------------------------------------------------------

    /// Save an index to a file path.
    ///
    /// Corresponds to `index->SaveIndex(path)`.
    pub fn nmslib_save_index(
        index: *mut NmslibIndex,
        path: *const c_char,
    );

    /// Save an index via an opaque writer callback.
    ///
    /// `writer_ctx` is a user-provided context pointer passed to the writer
    /// callback. This supports the stream-based I/O used in the k-NN plugin.
    ///
    /// Corresponds to `hnswIndex->SaveIndexWithStream(writer)`.
    pub fn nmslib_save_index_with_stream(
        index: *mut NmslibIndex,
        writer_ctx: *mut c_void,
    );

    /// Load an index from a file path.
    ///
    /// Corresponds to `index->LoadIndex(path)`.
    pub fn nmslib_load_index(
        index: *mut NmslibIndex,
        path: *const c_char,
    );

    /// Load an index via an opaque reader callback.
    ///
    /// `reader_ctx` is a user-provided context pointer passed to the reader
    /// callback. This supports the stream-based I/O used in the k-NN plugin.
    ///
    /// Corresponds to `hnswIndex->LoadIndexWithStream(reader)`.
    pub fn nmslib_load_index_with_stream(
        index: *mut NmslibIndex,
        reader_ctx: *mut c_void,
    );

    // -----------------------------------------------------------------------
    // Query-time configuration
    // -----------------------------------------------------------------------

    /// Set query-time parameters on an index.
    ///
    /// `params` is a null-terminated array of "key=value" C strings.
    /// Typically used to set "efSearch=N".
    ///
    /// Corresponds to `index->SetQueryTimeParams(similarity::AnyParams(params))`.
    pub fn nmslib_set_query_time_params(
        index: *mut NmslibIndex,
        params: *const *const c_char,
        num_params: c_int,
    );

    // -----------------------------------------------------------------------
    // Search / query
    // -----------------------------------------------------------------------

    /// Execute a k-NN query against the index.
    ///
    /// `query_vector` points to `dim` floats representing the query point.
    /// `k` is the number of nearest neighbors to retrieve.
    /// `result_ids` must point to a buffer of at least `k` ints.
    /// `result_distances` must point to a buffer of at least `k` floats.
    ///
    /// Returns the actual number of results found (may be less than k).
    ///
    /// Corresponds to creating a `KNNQuery`, calling `index->Search(query)`,
    /// and extracting results from `query->Result()`.
    pub fn nmslib_query_index(
        index: *mut NmslibIndex,
        space: *mut NmslibSpace,
        query_vector: *const c_float,
        dim: c_int,
        k: c_int,
        result_ids: *mut i64,
        result_distances: *mut c_float,
    ) -> c_int;

    /// Execute a k-NN query with a custom efSearch value.
    ///
    /// Same as `nmslib_query_index` but uses the provided `ef_search`
    /// parameter instead of the one configured on the index.
    ///
    /// Corresponds to creating an `HNSWQuery` with the ef_search override.
    pub fn nmslib_query_index_with_ef(
        index: *mut NmslibIndex,
        space: *mut NmslibSpace,
        query_vector: *const c_float,
        dim: c_int,
        k: c_int,
        ef_search: c_int,
        result_ids: *mut i64,
        result_distances: *mut c_float,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // Index destruction
    // -----------------------------------------------------------------------

    /// Free an index and all associated resources.
    pub fn nmslib_free_index(index: *mut NmslibIndex);
}
