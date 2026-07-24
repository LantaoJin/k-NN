// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! IndexService trait and implementations for float, binary, and byte Faiss indices.
//!
//! Ported from: jni/src/faiss_index_service.cpp
//! Header:      jni/include/faiss_index_service.h

use jni::sys::{jint, jlong};
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_int;

// ---------------------------------------------------------------------------
// Constants (mirrors extern const std::string in jni_util.h)
// ---------------------------------------------------------------------------
pub const NPROBES: &str = "nprobes";
pub const COARSE_QUANTIZER: &str = "coarse_quantizer";
pub const M: &str = "m";
pub const EF_CONSTRUCTION: &str = "ef_construction";
pub const EF_SEARCH: &str = "ef_search";

// ---------------------------------------------------------------------------
// Faiss metric type (mirrors faiss::MetricType enum)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaissMetricType {
    MetricInnerProduct = 0,
    MetricL2 = 1,
    MetricL1 = 2,
    MetricLinf = 3,
    MetricLp = 4,
    MetricCanberra = 20,
    MetricBrayCurtis = 21,
    MetricJensenShannon = 22,
}

// ---------------------------------------------------------------------------
// Opaque FFI types representing Faiss C++ objects
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct FaissIndex {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct FaissIndexBinary {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct FaissIndexIDMap {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct FaissIndexBinaryIDMap {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct FaissIOWriter {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct FaissIndexHNSW {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct FaissIndexIVF {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// FFI declarations to Faiss C API
// These will be moved to a dedicated ffi/ module in a later phase.
// ---------------------------------------------------------------------------
mod ffi {
    use super::*;

    extern "C" {
        // faiss::index_factory
        pub fn faiss_index_factory(
            p_index: *mut *mut FaissIndex,
            d: c_int,
            description: *const libc::c_char,
            metric: FaissMetricType,
        ) -> c_int;

        // faiss::index_binary_factory
        pub fn faiss_index_binary_factory(
            p_index: *mut *mut FaissIndexBinary,
            d: c_int,
            description: *const libc::c_char,
        ) -> c_int;

        // Create IndexIDMap wrapping an index
        pub fn faiss_IndexIDMap_new(
            p_index: *mut *mut FaissIndexIDMap,
            index: *mut FaissIndex,
        ) -> c_int;

        // Create IndexBinaryIDMap wrapping an index
        pub fn faiss_IndexBinaryIDMap_new(
            p_index: *mut *mut FaissIndexBinaryIDMap,
            index: *mut FaissIndexBinary,
        ) -> c_int;

        // Set own_fields on IndexIDMap
        pub fn faiss_IndexIDMap_set_own_fields(
            index: *mut FaissIndexIDMap,
            own: c_int,
        );

        // Set own_fields on IndexBinaryIDMap
        pub fn faiss_IndexBinaryIDMap_set_own_fields(
            index: *mut FaissIndexBinaryIDMap,
            own: c_int,
        );

        // Check if index is trained
        pub fn faiss_Index_is_trained(index: *const FaissIndex) -> c_int;

        // Check if binary index is trained
        pub fn faiss_IndexBinary_is_trained(index: *const FaissIndexBinary) -> c_int;

        // Add vectors with IDs (works on any Index including IndexIDMap)
        pub fn faiss_Index_add_with_ids(
            index: *mut FaissIndex,
            n: i64,
            x: *const f32,
            xids: *const i64,
        ) -> c_int;

        // Add vectors with IDs (binary index)
        pub fn faiss_IndexBinaryIDMap_add_with_ids(
            index: *mut FaissIndexBinaryIDMap,
            n: i64,
            x: *const u8,
            xids: *const i64,
        ) -> c_int;

        // Write index to IOWriter
        pub fn faiss_write_index_to_IOWriter(
            index: *const FaissIndex,
            writer: *mut FaissIOWriter,
        ) -> c_int;

        // Write binary index to IOWriter
        pub fn faiss_write_index_binary_to_IOWriter(
            index: *const FaissIndexBinary,
            writer: *mut FaissIOWriter,
        ) -> c_int;

        // Free an index
        pub fn faiss_Index_free(index: *mut FaissIndex);

        // Free a binary index
        pub fn faiss_IndexBinary_free(index: *mut FaissIndexBinary);

        // Free IndexIDMap
        pub fn faiss_IndexIDMap_free(index: *mut FaissIndexIDMap);

        // Free IndexBinaryIDMap
        pub fn faiss_IndexBinaryIDMap_free(index: *mut FaissIndexBinaryIDMap);

        // Get the sub-index from an IndexIDMap
        pub fn faiss_IndexIDMap_sub_index(index: *mut FaissIndexIDMap) -> *mut FaissIndex;

        // OMP thread count
        pub fn omp_set_num_threads(num_threads: c_int);
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------
#[derive(Debug)]
pub enum IndexServiceError {
    IndexNotTrained,
    FaissError(String),
    InvalidArgument(String),
    WriteError(String),
}

impl std::fmt::Display for IndexServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexServiceError::IndexNotTrained => write!(f, "Index is not trained"),
            IndexServiceError::FaissError(msg) => write!(f, "Faiss error: {}", msg),
            IndexServiceError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            IndexServiceError::WriteError(msg) => write!(f, "Write error: {}", msg),
        }
    }
}

impl std::error::Error for IndexServiceError {}

pub type Result<T> = std::result::Result<T, IndexServiceError>;

// ---------------------------------------------------------------------------
// Extra parameters map type
// In the Rust port, parameters are pre-converted from Java objects to native values.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub enum ParamValue {
    Int(i32),
    SubParams(HashMap<String, ParamValue>),
}

/// Set extra parameters that cannot be configured with the index factory.
///
/// Uses the knn_shim RTTI helpers to detect the runtime index type (HNSW or IVF)
/// and set the appropriate parameters. Unwraps IndexIDMap automatically.
pub fn set_extra_parameters(
    parameters: &HashMap<String, ParamValue>,
    index: *mut FaissIndex,
) {
    use crate::ffi::knn_shim;
    use std::ffi::c_void;

    let index_ptr = index as *mut c_void;

    unsafe {
        // Check if the index is HNSW and set HNSW-specific params
        if knn_shim::knn_shim_is_index_hnsw(index_ptr) != 0 {
            if let Some(ParamValue::Int(ef_construction)) = parameters.get(EF_CONSTRUCTION) {
                knn_shim::knn_shim_set_hnsw_ef_construction(index_ptr, *ef_construction as c_int);
            }
            if let Some(ParamValue::Int(ef_search)) = parameters.get(EF_SEARCH) {
                knn_shim::knn_shim_set_hnsw_ef_search(index_ptr, *ef_search as c_int);
            }
        }

        // Check if the index is IVF and set IVF-specific params
        if knn_shim::knn_shim_is_index_ivf(index_ptr) != 0 {
            if let Some(ParamValue::Int(nprobes)) = parameters.get(NPROBES) {
                knn_shim::knn_shim_set_ivf_nprobe(index_ptr, *nprobes as c_int);
            }
            // Recurse into coarse_quantizer sub-params if present
            if let Some(ParamValue::SubParams(sub_params)) = parameters.get(COARSE_QUANTIZER) {
                // The coarse quantizer is typically an HNSW index inside the IVF.
                // The shim functions already unwrap IDMap, so we pass the same pointer.
                // In C++, set_extra_parameters is called recursively on the coarse_quantizer.
                // The shim handles unwrapping, so HNSW params on the top-level pointer
                // will target the HNSW coarse quantizer if the index is IVF+HNSW.
                if let Some(ParamValue::Int(ef_search)) = sub_params.get(EF_SEARCH) {
                    knn_shim::knn_shim_set_hnsw_ef_search(index_ptr, *ef_search as c_int);
                }
                if let Some(ParamValue::Int(ef_construction)) = sub_params.get(EF_CONSTRUCTION) {
                    knn_shim::knn_shim_set_hnsw_ef_construction(index_ptr, *ef_construction as c_int);
                }
            }
        }
    }
}

/// Binary index variant of set_extra_parameters.
///
/// Binary indices in the k-NN plugin do not currently support extra runtime parameters
/// (no HNSW efSearch/efConstruction or IVF nprobe on binary indices via this path).
/// This is intentionally a no-op.
pub fn set_extra_parameters_binary(
    _parameters: &HashMap<String, ParamValue>,
    _index: *mut FaissIndexBinary,
) {
    // Binary indices don't have the same parameter surface as float indices.
    // The C++ code also effectively no-ops for binary index extra parameters.
}

// ---------------------------------------------------------------------------
// IndexService trait
// ---------------------------------------------------------------------------
/// Trait mirroring the C++ IndexService virtual class hierarchy.
pub trait IndexService {
    /// Initialize a Faiss index and return its memory address as jlong.
    ///
    /// # Safety
    /// The returned jlong is a raw pointer cast. The caller is responsible for
    /// eventually freeing it via `write_index` (which takes ownership) or explicit free.
    fn init_index(
        &self,
        metric: FaissMetricType,
        index_description: &str,
        dim: i32,
        num_vectors: i32,
        thread_count: i32,
        parameters: &HashMap<String, ParamValue>,
    ) -> Result<jlong>;

    /// Insert vectors into the index.
    ///
    /// # Safety
    /// `vectors_address` must be a valid pointer to a Vec of the appropriate element type.
    /// `id_map_address` must be a valid pointer to the index's IDMap.
    unsafe fn insert_to_index(
        &self,
        dim: i32,
        num_ids: i32,
        thread_count: i32,
        vectors_address: i64,
        ids: &[i64],
        id_map_address: jlong,
    ) -> Result<()>;

    /// Write the index using the provided IOWriter, consuming (freeing) the index.
    ///
    /// # Safety
    /// `id_map_address` must be a valid pointer previously returned by `init_index`.
    /// `writer` must be a valid pointer to a Faiss IOWriter.
    unsafe fn write_index(
        &self,
        writer: *mut FaissIOWriter,
        id_map_address: jlong,
        skip_flat: bool,
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------
// FloatIndexService (corresponds to C++ IndexService base class for float vectors)
// ---------------------------------------------------------------------------
pub struct FloatIndexService;

impl FloatIndexService {
    pub fn new() -> Self {
        FloatIndexService
    }

    /// Pre-allocate storage for HNSW indices (mirrors C++ IndexService::allocIndex).
    /// Stubbed: requires access to internal Faiss storage pointers.
    fn alloc_index(&self, _index: *mut FaissIndex, _dim: usize, _num_vectors: usize) {
        // TODO: dynamic_cast equivalent to detect IndexHNSWSQ/IndexHNSW and reserve storage.
        // In C++:
        //   - IndexHNSWSQ -> storage (IndexScalarQuantizer) -> codes.reserve(code_size * numVectors)
        //   - IndexHNSW  -> storage (IndexFlat) -> codes.reserve(code_size * numVectors)
        // This requires Faiss C API helpers that expose storage pointers.
    }
}

impl Default for FloatIndexService {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexService for FloatIndexService {
    fn init_index(
        &self,
        metric: FaissMetricType,
        index_description: &str,
        dim: i32,
        num_vectors: i32,
        thread_count: i32,
        parameters: &HashMap<String, ParamValue>,
    ) -> Result<jlong> {
        let desc_cstr = CString::new(index_description)
            .map_err(|e| IndexServiceError::InvalidArgument(e.to_string()))?;

        unsafe {
            // Set thread count
            if thread_count != 0 {
                ffi::omp_set_num_threads(thread_count as c_int);
            }

            // Create index using Faiss factory method
            let mut index_ptr: *mut FaissIndex = std::ptr::null_mut();
            let ret = ffi::faiss_index_factory(
                &mut index_ptr,
                dim as c_int,
                desc_cstr.as_ptr(),
                metric,
            );
            if ret != 0 {
                return Err(IndexServiceError::FaissError(
                    "faiss_index_factory failed".to_string(),
                ));
            }

            // Set extra parameters (currently stubbed for complex indices)
            if !parameters.is_empty() {
                // set_extra_parameters(parameters, index_ptr);
                // Skipping for now - will be enabled once RTTI helpers are available
            }

            // Check that the index is trained
            if ffi::faiss_Index_is_trained(index_ptr) == 0 {
                ffi::faiss_Index_free(index_ptr);
                return Err(IndexServiceError::IndexNotTrained);
            }

            // Create IndexIDMap wrapping the index
            let mut id_map_ptr: *mut FaissIndexIDMap = std::ptr::null_mut();
            let ret = ffi::faiss_IndexIDMap_new(&mut id_map_ptr, index_ptr);
            if ret != 0 {
                ffi::faiss_Index_free(index_ptr);
                return Err(IndexServiceError::FaissError(
                    "faiss_IndexIDMap_new failed".to_string(),
                ));
            }

            // Set own_fields = true so the IDMap will free the underlying index
            ffi::faiss_IndexIDMap_set_own_fields(id_map_ptr, 1);

            // Allocate storage hints
            let sub_index = ffi::faiss_IndexIDMap_sub_index(id_map_ptr);
            self.alloc_index(sub_index, dim as usize, num_vectors as usize);

            Ok(id_map_ptr as jlong)
        }
    }

    unsafe fn insert_to_index(
        &self,
        dim: i32,
        num_ids: i32,
        thread_count: i32,
        vectors_address: i64,
        ids: &[i64],
        id_map_address: jlong,
    ) -> Result<()> {
        // Read vectors from memory address
        let input_vectors: *const Vec<f32> = vectors_address as *const Vec<f32>;
        if input_vectors.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "vectors_address pointer is null".to_string(),
            ));
        }
        let input_vectors_ref = &*input_vectors;

        let num_vectors = (input_vectors_ref.len() as u64 / dim as u64) as i32;
        if num_vectors == 0 {
            return Err(IndexServiceError::InvalidArgument(
                "Number of vectors cannot be 0".to_string(),
            ));
        }
        if num_ids != num_vectors {
            return Err(IndexServiceError::InvalidArgument(
                format!(
                    "Number of IDs does not match number of vectors: num_ids={}, num_vectors={}, vec.len()={}, dim={}",
                    num_ids, num_vectors, input_vectors_ref.len(), dim
                ),
            ));
        }

        // Set thread count
        if thread_count != 0 {
            ffi::omp_set_num_threads(thread_count as c_int);
        }

        let id_map = id_map_address as *mut FaissIndexIDMap;
        if id_map.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "id_map_address pointer is null".to_string(),
            ));
        }

        // Add vectors (IndexIDMap is-a Index, safe to cast)
        let ret = ffi::faiss_Index_add_with_ids(
            id_map as *mut FaissIndex,
            num_vectors as i64,
            input_vectors_ref.as_ptr(),
            ids.as_ptr(),
        );
        if ret != 0 {
            return Err(IndexServiceError::FaissError(
                format!(
                    "faiss_IndexIDMap_add_with_ids failed: ret={}, id_map={:?}, n={}, dim={}, vec_ptr={:?}",
                    ret, id_map, num_vectors, dim, input_vectors_ref.as_ptr()
                ),
            ));
        }

        Ok(())
    }

    unsafe fn write_index(
        &self,
        writer: *mut FaissIOWriter,
        id_map_address: jlong,
        _skip_flat: bool,
    ) -> Result<()> {
        let id_map = id_map_address as *mut FaissIndexIDMap;
        if id_map.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "id_map_address pointer is null".to_string(),
            ));
        }

        // Write the index via IOWriter
        let ret = ffi::faiss_write_index_to_IOWriter(id_map as *const FaissIndex, writer);
        if ret != 0 {
            // Free even on error since we took ownership
            ffi::faiss_IndexIDMap_free(id_map);
            return Err(IndexServiceError::WriteError(
                "Failed to write index to disk".to_string(),
            ));
        }

        // TODO: flush if writer is FaissOpenSearchIOWriter
        // In C++ this uses dynamic_cast to detect the OpenSearchIOWriter subclass.
        // In Rust FFI, we would need a type tag or vtable inspection.

        // Free the index (we took ownership, mirroring unique_ptr semantics in C++)
        ffi::faiss_IndexIDMap_free(id_map);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BinaryIndexService
// ---------------------------------------------------------------------------
pub struct BinaryIndexService;

impl BinaryIndexService {
    pub fn new() -> Self {
        BinaryIndexService
    }

    /// Pre-allocate storage for binary HNSW indices.
    fn alloc_index(&self, _index: *mut FaissIndex, _dim: usize, _num_vectors: usize) {
        // TODO: dynamic_cast equivalent to detect IndexBinaryHNSW and reserve storage.
        // In C++:
        //   - IndexBinaryHNSW -> storage (IndexBinaryFlat) -> xb.reserve(dim * numVectors / 8)
    }

    /// Initialize a Faiss SQ (scalar quantized) HNSW binary index.
    ///
    /// # Safety
    /// Returns a raw pointer as jlong. Caller must free via write_index or explicit free.
    pub unsafe fn init_faiss_sq_index(
        &self,
        _metric: FaissMetricType,
        _index_description: &str,
        _dim: i32,
        _num_vectors: i32,
        _thread_count: i32,
        _parameters: &HashMap<String, ParamValue>,
        _centroid_dp: f32,
        _quantized_vector_bytes: i32,
    ) -> Result<jlong> {
        // Delegates to the C++ shim which handles FaissSQHnsw/FaissSQFlat template instantiation.
        let result = unsafe {
            crate::ffi::knn_shim::knn_shim_init_sq_index(
                _num_vectors,
                _dim,
                _centroid_dp,
                _quantized_vector_bytes,
            )
        };
        if result == 0 {
            Err(IndexServiceError::FaissError("Failed to initialize SQ index".to_string()))
        } else {
            Ok(result)
        }
    }
}

impl Default for BinaryIndexService {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexService for BinaryIndexService {
    fn init_index(
        &self,
        _metric: FaissMetricType,
        index_description: &str,
        dim: i32,
        num_vectors: i32,
        thread_count: i32,
        parameters: &HashMap<String, ParamValue>,
    ) -> Result<jlong> {
        let desc_cstr = CString::new(index_description)
            .map_err(|e| IndexServiceError::InvalidArgument(e.to_string()))?;

        unsafe {
            // Set thread count
            if thread_count != 0 {
                ffi::omp_set_num_threads(thread_count as c_int);
            }

            // Create binary index using Faiss factory method (metric not used for binary)
            let mut index_ptr: *mut FaissIndexBinary = std::ptr::null_mut();
            let ret = ffi::faiss_index_binary_factory(
                &mut index_ptr,
                dim as c_int,
                desc_cstr.as_ptr(),
            );
            if ret != 0 {
                return Err(IndexServiceError::FaissError(
                    "faiss_index_binary_factory failed".to_string(),
                ));
            }

            // Set extra parameters (currently stubbed)
            if !parameters.is_empty() {
                // set_extra_parameters_binary(parameters, index_ptr);
            }

            // Check that the index is trained
            if ffi::faiss_IndexBinary_is_trained(index_ptr) == 0 {
                ffi::faiss_IndexBinary_free(index_ptr);
                return Err(IndexServiceError::IndexNotTrained);
            }

            // Create IndexBinaryIDMap wrapping the index
            let mut id_map_ptr: *mut FaissIndexBinaryIDMap = std::ptr::null_mut();
            let ret = ffi::faiss_IndexBinaryIDMap_new(&mut id_map_ptr, index_ptr);
            if ret != 0 {
                ffi::faiss_IndexBinary_free(index_ptr);
                return Err(IndexServiceError::FaissError(
                    "faiss_IndexBinaryIDMap_new failed".to_string(),
                ));
            }

            // Set own_fields = true
            ffi::faiss_IndexBinaryIDMap_set_own_fields(id_map_ptr, 1);

            // alloc_index for storage pre-allocation (stubbed)
            // In C++ this casts to faiss::Index* via dynamic_cast on idMap->index
            // which is a binary index. The alloc is a no-op unless it's HNSW.
            let _ = (dim, num_vectors); // suppress unused warnings

            Ok(id_map_ptr as jlong)
        }
    }

    unsafe fn insert_to_index(
        &self,
        dim: i32,
        num_ids: i32,
        thread_count: i32,
        vectors_address: i64,
        ids: &[i64],
        id_map_address: jlong,
    ) -> Result<()> {
        // Read vectors from memory address (binary vectors are uint8)
        let input_vectors: *const Vec<u8> = vectors_address as *const Vec<u8>;
        if input_vectors.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "vectors_address pointer is null".to_string(),
            ));
        }
        let input_vectors_ref = &*input_vectors;

        // For binary indices, each vector has dim/8 bytes
        let bytes_per_vector = (dim / 8) as u64;
        let num_vectors = (input_vectors_ref.len() as u64 / bytes_per_vector) as i32;
        if num_vectors == 0 {
            return Err(IndexServiceError::InvalidArgument(
                "Number of vectors cannot be 0".to_string(),
            ));
        }
        if num_ids != num_vectors {
            return Err(IndexServiceError::InvalidArgument(
                "Number of IDs does not match number of vectors".to_string(),
            ));
        }

        // Set thread count
        if thread_count != 0 {
            ffi::omp_set_num_threads(thread_count as c_int);
        }

        let id_map = id_map_address as *mut FaissIndexBinaryIDMap;
        if id_map.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "id_map_address pointer is null".to_string(),
            ));
        }

        // Add vectors
        let ret = ffi::faiss_IndexBinaryIDMap_add_with_ids(
            id_map,
            num_vectors as i64,
            input_vectors_ref.as_ptr(),
            ids.as_ptr(),
        );
        if ret != 0 {
            return Err(IndexServiceError::FaissError(
                "faiss_IndexBinaryIDMap_add_with_ids failed".to_string(),
            ));
        }

        Ok(())
    }

    unsafe fn write_index(
        &self,
        writer: *mut FaissIOWriter,
        id_map_address: jlong,
        _skip_flat: bool,
    ) -> Result<()> {
        let id_map = id_map_address as *mut FaissIndexBinaryIDMap;
        if id_map.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "id_map_address pointer is null".to_string(),
            ));
        }

        // Write the binary index via IOWriter
        let ret =
            ffi::faiss_write_index_binary_to_IOWriter(id_map as *const FaissIndexBinary, writer);
        if ret != 0 {
            ffi::faiss_IndexBinaryIDMap_free(id_map);
            return Err(IndexServiceError::WriteError(
                "Failed to write index to disk".to_string(),
            ));
        }

        // TODO: flush if writer is FaissOpenSearchIOWriter

        // Free the index (took ownership)
        ffi::faiss_IndexBinaryIDMap_free(id_map);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ByteIndexService
// ---------------------------------------------------------------------------
pub struct ByteIndexService;

impl ByteIndexService {
    pub fn new() -> Self {
        ByteIndexService
    }

    /// Pre-allocate storage for byte (SQ) HNSW indices.
    fn alloc_index(&self, _index: *mut FaissIndex, _dim: usize, _num_vectors: usize) {
        // TODO: dynamic_cast equivalent to detect IndexHNSWSQ and reserve storage.
        // In C++:
        //   - IndexHNSWSQ -> storage (IndexScalarQuantizer) -> codes.reserve(code_size * numVectors)
    }
}

impl Default for ByteIndexService {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexService for ByteIndexService {
    fn init_index(
        &self,
        metric: FaissMetricType,
        index_description: &str,
        dim: i32,
        num_vectors: i32,
        thread_count: i32,
        parameters: &HashMap<String, ParamValue>,
    ) -> Result<jlong> {
        let desc_cstr = CString::new(index_description)
            .map_err(|e| IndexServiceError::InvalidArgument(e.to_string()))?;

        unsafe {
            // Set thread count
            if thread_count != 0 {
                ffi::omp_set_num_threads(thread_count as c_int);
            }

            // Create index using Faiss factory method
            let mut index_ptr: *mut FaissIndex = std::ptr::null_mut();
            let ret = ffi::faiss_index_factory(
                &mut index_ptr,
                dim as c_int,
                desc_cstr.as_ptr(),
                metric,
            );
            if ret != 0 {
                return Err(IndexServiceError::FaissError(
                    "faiss_index_factory failed".to_string(),
                ));
            }

            // Set extra parameters (currently stubbed)
            if !parameters.is_empty() {
                // set_extra_parameters(parameters, index_ptr);
            }

            // Check that the index is trained
            if ffi::faiss_Index_is_trained(index_ptr) == 0 {
                ffi::faiss_Index_free(index_ptr);
                return Err(IndexServiceError::IndexNotTrained);
            }

            // Create IndexIDMap wrapping the index
            let mut id_map_ptr: *mut FaissIndexIDMap = std::ptr::null_mut();
            let ret = ffi::faiss_IndexIDMap_new(&mut id_map_ptr, index_ptr);
            if ret != 0 {
                ffi::faiss_Index_free(index_ptr);
                return Err(IndexServiceError::FaissError(
                    "faiss_IndexIDMap_new failed".to_string(),
                ));
            }

            // Set own_fields = true
            ffi::faiss_IndexIDMap_set_own_fields(id_map_ptr, 1);

            // Allocate storage hints
            let sub_index = ffi::faiss_IndexIDMap_sub_index(id_map_ptr);
            self.alloc_index(sub_index, dim as usize, num_vectors as usize);

            Ok(id_map_ptr as jlong)
        }
    }

    /// Insert byte (int8) vectors by converting them to float in batches of 1000.
    /// This mirrors the C++ implementation which avoids memory spikes.
    unsafe fn insert_to_index(
        &self,
        dim: i32,
        num_ids: i32,
        thread_count: i32,
        vectors_address: i64,
        ids: &[i64],
        id_map_address: jlong,
    ) -> Result<()> {
        // Read vectors from memory address (byte vectors are i8)
        let input_vectors: *const Vec<i8> = vectors_address as *const Vec<i8>;
        if input_vectors.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "vectors_address pointer is null".to_string(),
            ));
        }
        let input_vectors_ref = &*input_vectors;

        let num_vectors = (input_vectors_ref.len() / dim as usize) as i32;
        if num_vectors == 0 {
            return Err(IndexServiceError::InvalidArgument(
                "Number of vectors cannot be 0".to_string(),
            ));
        }
        if num_ids != num_vectors {
            return Err(IndexServiceError::InvalidArgument(
                "Number of IDs does not match number of vectors".to_string(),
            ));
        }

        // Set thread count
        if thread_count != 0 {
            ffi::omp_set_num_threads(thread_count as c_int);
        }

        let id_map = id_map_address as *mut FaissIndexIDMap;
        if id_map.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "id_map_address pointer is null".to_string(),
            ));
        }
        let dim_usize = dim as usize;

        // Add vectors in batches by casting int8 vectors into float with a batch size of 1000
        // to avoid additional memory spike.
        // Refer to: https://github.com/opensearch-project/k-NN/issues/1659#issuecomment-2307390255
        let batch_size_max: usize = 1000;
        let mut input_float_vectors: Vec<f32> = vec![0.0; batch_size_max * dim_usize];
        let mut float_vectors_ids: Vec<i64> = vec![0; batch_size_max];

        let mut iter_offset: usize = 0;
        let mut id: usize = 0;

        while id < num_vectors as usize {
            let batch_size = std::cmp::min(batch_size_max, num_vectors as usize - id);

            for i in 0..batch_size {
                float_vectors_ids[i] = ids[id + i];
                for j in 0..dim_usize {
                    input_float_vectors[i * dim_usize + j] =
                        input_vectors_ref[iter_offset] as f32;
                    iter_offset += 1;
                }
            }

            let ret = ffi::faiss_Index_add_with_ids(
                id_map as *mut FaissIndex,
                batch_size as i64,
                input_float_vectors.as_ptr(),
                float_vectors_ids.as_ptr(),
            );
            if ret != 0 {
                return Err(IndexServiceError::FaissError(
                    "faiss_IndexIDMap_add_with_ids failed during batch insert".to_string(),
                ));
            }

            id += batch_size;
        }

        Ok(())
    }

    unsafe fn write_index(
        &self,
        writer: *mut FaissIOWriter,
        id_map_address: jlong,
        _skip_flat: bool,
    ) -> Result<()> {
        let id_map = id_map_address as *mut FaissIndexIDMap;
        if id_map.is_null() {
            return Err(IndexServiceError::InvalidArgument(
                "id_map_address pointer is null".to_string(),
            ));
        }

        // Write the index via IOWriter
        let ret = ffi::faiss_write_index_to_IOWriter(id_map as *const FaissIndex, writer);
        if ret != 0 {
            ffi::faiss_IndexIDMap_free(id_map);
            return Err(IndexServiceError::WriteError(
                "Failed to write index to disk".to_string(),
            ));
        }

        // TODO: flush if writer is FaissOpenSearchIOWriter

        // Free the index (took ownership)
        ffi::faiss_IndexIDMap_free(id_map);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: create the appropriate IndexService implementation based on type tag
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    Float,
    Binary,
    Byte,
}

/// Factory function to create an IndexService boxed trait object.
pub fn create_index_service(index_type: IndexType) -> Box<dyn IndexService> {
    match index_type {
        IndexType::Float => Box::new(FloatIndexService::new()),
        IndexType::Binary => Box::new(BinaryIndexService::new()),
        IndexType::Byte => Box::new(ByteIndexService::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_index_service() {
        // Verify we can instantiate each service type
        let _ = create_index_service(IndexType::Float);
        let _ = create_index_service(IndexType::Binary);
        let _ = create_index_service(IndexType::Byte);
    }

    #[test]
    fn test_param_value() {
        let mut params = HashMap::new();
        params.insert(EF_CONSTRUCTION.to_string(), ParamValue::Int(128));
        params.insert(EF_SEARCH.to_string(), ParamValue::Int(64));

        let mut sub = HashMap::new();
        sub.insert(NPROBES.to_string(), ParamValue::Int(8));
        params.insert(
            COARSE_QUANTIZER.to_string(),
            ParamValue::SubParams(sub),
        );

        assert_eq!(params.len(), 3);
    }
}
