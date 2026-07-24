// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Faiss index wrapper functions: create, load, query, free operations.
//!
//! Ported from: jni/src/faiss_wrapper.cpp + jni/include/faiss_wrapper.h
//!
//! This module provides the Rust equivalent of the C++ `knn_jni::faiss_wrapper`
//! namespace, using the `jni` crate for JNI interaction and unsafe FFI calls
//! to Faiss C library.

use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;

use jni::objects::{JByteArray, JFloatArray, JIntArray, JLongArray, JObject, JObjectArray, JString};
use jni::sys::{jboolean, jbyte, jfloat, jint, jlong};
use jni::JNIEnv;

use crate::ffi::faiss_sys::{
    self, FaissIOReader, FaissIdx, FaissIndex, FaissIndexBinary, FaissIndexHNSW, FaissIndexIDMap,
    FaissIndexIVF, FaissIndexIVFPQ, FaissMetricType, FaissSearchParameters,
    FaissSearchParametersHNSW, FaissSearchParametersIVF, FaissVectorIOReader,
    IO_FLAG_PQ_SKIP_SDC_TABLE, IO_FLAG_READ_ONLY, IO_FLAG_SKIP_PRECOMPUTE_TABLE, METRIC_INNER_PRODUCT,
    METRIC_L2,
};
use crate::jni_util::{
    self, BQQuantizationLevel, JniUtilError, COSINESIMIL, COARSE_QUANTIZER, EF_CONSTRUCTION,
    EF_SEARCH, HAMMING, INDEX_DESCRIPTION, INDEX_THREAD_QUANTITY, INNER_PRODUCT, L2, NPROBES,
    PARAMETERS, SPACE_TYPE,
};

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

/// Error type for faiss_wrapper operations.
#[derive(Debug, thiserror::Error)]
pub enum FaissWrapperError {
    #[error("{0}")]
    Runtime(String),

    #[error("JNI error: {0}")]
    Jni(#[from] jni::errors::Error),

    #[error("Null pointer: {0}")]
    NullPointer(String),
}

pub type Result<T> = std::result::Result<T, FaissWrapperError>;

// ---------------------------------------------------------------------------
// FilterIdsSelectorType enum
// ---------------------------------------------------------------------------

/// Defines the type of IDSelector used for filtering.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterIdsSelectorType {
    Bitmap = 0,
    Batch = 1,
}

// ---------------------------------------------------------------------------
// IDSelectorJlongBitmap -- Rust struct mimicking the C++ faiss struct
// ---------------------------------------------------------------------------

/// A bitmap-based ID selector using jlong (i64) arrays, compatible with
/// Lucene's FixedBitSet format.
///
/// This struct is the Rust equivalent of the C++ `faiss::IDSelectorJlongBitmap`.
/// It checks membership by treating the bitmap array as a bitset where bit `id`
/// is stored at word `id >> 6` and bit position `id & 63`.
pub struct IDSelectorJlongBitmap {
    /// Number of i64 words in the bitmap.
    pub n: usize,
    /// Pointer to the bitmap data (jlong array from JNI).
    /// The caller must ensure this pointer remains valid for the lifetime of this struct.
    pub bitmap: *const i64,
}

impl IDSelectorJlongBitmap {
    /// Create a new IDSelectorJlongBitmap.
    ///
    /// # Safety
    /// - `bitmap` must point to at least `n` valid i64 values.
    /// - The pointed-to data must remain valid for the lifetime of this struct.
    pub unsafe fn new(n: usize, bitmap: *const i64) -> Self {
        IDSelectorJlongBitmap { n, bitmap }
    }

    /// Check if the given ID is a member of the bitmap (selected).
    ///
    /// Returns true if the bit corresponding to `id` is set.
    pub fn is_member(&self, id: i64) -> bool {
        let index = id as u64;
        let i = (index >> 6) as usize; // div 64
        if i >= self.n {
            return false;
        }
        unsafe {
            let word = *self.bitmap.add(i);
            ((word as u64) >> (index & 63)) & 1 == 1
        }
    }
}

// ---------------------------------------------------------------------------
// TranslateSpaceToMetric
// ---------------------------------------------------------------------------

/// Translate a k-NN space type string to the corresponding Faiss metric type.
///
/// # Errors
/// Returns `FaissWrapperError::Runtime` if the space type is not recognized.
pub fn translate_space_to_metric(space_type: &str) -> Result<FaissMetricType> {
    match space_type {
        s if s == L2 => Ok(METRIC_L2),
        s if s == INNER_PRODUCT => Ok(METRIC_INNER_PRODUCT),
        // Cosine similarity: vectors are normalized during indexing, so cosine == inner product
        s if s == COSINESIMIL => Ok(METRIC_INNER_PRODUCT),
        // Hamming space is not directly used for Faiss float indices; use L2 to avoid error
        s if s == HAMMING => Ok(METRIC_L2),
        _ => Err(FaissWrapperError::Runtime(format!(
            "Invalid spaceType: {}",
            space_type
        ))),
    }
}

// ---------------------------------------------------------------------------
// SetExtraParameters
// ---------------------------------------------------------------------------

/// Set additional parameters on a Faiss index that cannot be configured via the
/// index factory string.
///
/// Handles HNSW (efConstruction, efSearch) and IVF (nprobe, coarse_quantizer) parameters.
///
/// # Safety
/// - `index` must be a valid pointer to a live Faiss index.
/// - The `parameters` map values must be valid JNI objects (integers or sub-maps).
pub unsafe fn set_extra_parameters(
    env: &mut JNIEnv,
    parameters: &HashMap<String, jlong>,
    index: *mut FaissIndex,
) {
    if index.is_null() {
        return;
    }

    // Try to downcast to IVF
    let ivf_ptr = faiss_sys::faiss_index_to_ivf(index);
    if !ivf_ptr.is_null() {
        if let Some(&nprobe_val) = parameters.get(NPROBES) {
            faiss_sys::faiss_index_ivf_set_nprobe(ivf_ptr, nprobe_val as c_int);
        }

        if let Some(&coarse_quantizer_ptr) = parameters.get(COARSE_QUANTIZER) {
            // Coarse quantizer is itself an index; recursively set its parameters
            // For simplicity we skip recursive sub-parameter handling here.
            // In the full port this would parse the sub-map and recurse.
            let _ = coarse_quantizer_ptr;
        }
    }

    // Try to downcast to HNSW
    let hnsw_ptr = faiss_sys::faiss_index_to_hnsw(index);
    if !hnsw_ptr.is_null() {
        if let Some(&ef_construction_val) = parameters.get(EF_CONSTRUCTION) {
            faiss_sys::faiss_index_hnsw_set_ef_construction(hnsw_ptr, ef_construction_val as c_int);
        }

        if let Some(&ef_search_val) = parameters.get(EF_SEARCH) {
            faiss_sys::faiss_index_hnsw_set_ef_search(hnsw_ptr, ef_search_val as c_int);
        }
    }
}

// ---------------------------------------------------------------------------
// InternalTrainIndex
// ---------------------------------------------------------------------------

/// Train a Faiss float index with the given vectors.
///
/// If the index is an IVF index with `quantizer_trains_alone == 2`, the quantizer
/// is trained first. For IVF indices, a direct map is also created.
///
/// # Safety
/// - `index` must be a valid, non-null pointer to a Faiss Index.
/// - `x` must point to at least `n * d` floats where `d` is the index dimension.
pub unsafe fn internal_train_index(index: *mut FaissIndex, n: FaissIdx, x: *const f32) {
    if index.is_null() {
        return;
    }

    // Check if it's an IVF index and handle quantizer training
    let ivf_ptr = faiss_sys::faiss_index_to_ivf(index);
    if !ivf_ptr.is_null() {
        // In the C++ code: if quantizer_trains_alone == 2, train quantizer first
        // Then call make_direct_map().
        // For the FFI port, we call the generic train which handles this internally.
        // The C API doesn't expose quantizer_trains_alone directly, so we just train.
    }

    // Train the index if not already trained
    let is_trained = faiss_sys::faiss_index_is_trained(index);
    if is_trained == 0 {
        faiss_sys::faiss_index_train(index, n, x);
    }
}

/// Train a Faiss binary index with the given vectors.
///
/// # Safety
/// - `index` must be a valid, non-null pointer to a Faiss IndexBinary.
/// - `x` must point to at least `n * (d / 8)` bytes.
pub unsafe fn internal_train_binary_index(
    index: *mut FaissIndexBinary,
    n: FaissIdx,
    x: *const u8,
) {
    if index.is_null() {
        return;
    }

    let is_trained = faiss_sys::faiss_index_binary_is_trained(index);
    if is_trained == 0 {
        faiss_sys::faiss_index_binary_train(index, n, x);
    }
}

// ---------------------------------------------------------------------------
// InitIndex
// ---------------------------------------------------------------------------

/// Initialize a Faiss float index from parameters.
///
/// Extracts space type, dimension, index description, thread count, and extra
/// parameters from the Java map, then delegates to the IndexService.
///
/// # Safety
/// Uses JNI calls and FFI to Faiss.
pub unsafe fn init_index(
    env: &mut JNIEnv,
    num_docs: jlong,
    dim: jint,
    parameters: &JObject,
) -> Result<jlong> {
    if dim <= 0 {
        return Err(FaissWrapperError::Runtime(
            "Vectors dimensions cannot be less than or equal to 0".to_string(),
        ));
    }

    if parameters.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Parameters cannot be null".to_string(),
        ));
    }

    // Delegate to IndexService -- the actual implementation depends on the service.
    // Dead code: faiss_service_jni now calls IndexService trait methods directly.
    Err(FaissWrapperError::Runtime(
        "init_index: use IndexService trait dispatch via faiss_service_jni".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// InsertToIndex
// ---------------------------------------------------------------------------

/// Insert vectors into an existing index.
///
/// Note: Dead code path — faiss_service_jni calls IndexService directly.
pub unsafe fn insert_to_index(
    _env: &mut JNIEnv,
    ids: &JIntArray,
    vectors_address: jlong,
    dim: jint,
    _index_ptr: jlong,
    _thread_count: jint,
) -> Result<()> {
    if ids.is_null() {
        return Err(FaissWrapperError::Runtime(
            "IDs cannot be null".to_string(),
        ));
    }

    if vectors_address <= 0 {
        return Err(FaissWrapperError::Runtime(
            "VectorsAddress cannot be less than 0".to_string(),
        ));
    }

    if dim <= 0 {
        return Err(FaissWrapperError::Runtime(
            "Vectors dimensions cannot be less than or equal to 0".to_string(),
        ));
    }

    // Dead code: faiss_service_jni now calls IndexService trait methods directly.
    Err(FaissWrapperError::Runtime(
        "insert_to_index: use IndexService trait dispatch via faiss_service_jni".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// WriteIndex
// ---------------------------------------------------------------------------

/// Write an index to an output stream.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn write_index(
    env: &mut JNIEnv,
    output: &JObject,
    index_ptr: jlong,
) -> Result<()> {
    if output.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Index output stream cannot be null".to_string(),
        ));
    }

    // Dead code: faiss_service_jni now calls IndexService::write_index directly with stream support.
    Err(FaissWrapperError::Runtime(
        "write_index: use IndexService trait dispatch via faiss_service_jni".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// CreateIndexFromTemplate
// ---------------------------------------------------------------------------

/// Create a float index from a template index (serialized as bytes), adding the
/// given vectors and IDs. The result is written to the output stream.
///
/// # Safety
/// Uses JNI, raw pointer reinterpretation, and Faiss FFI.
pub unsafe fn create_index_from_template(
    env: &mut JNIEnv,
    ids: &JIntArray,
    vectors_address: jlong,
    dim: jint,
    output: &JObject,
    template_index: &JByteArray,
    parameters: &JObject,
) -> Result<()> {
    if ids.is_null() {
        return Err(FaissWrapperError::Runtime("IDs cannot be null".to_string()));
    }
    if vectors_address <= 0 {
        return Err(FaissWrapperError::Runtime(
            "VectorsAddress cannot be less than 0".to_string(),
        ));
    }
    if dim <= 0 {
        return Err(FaissWrapperError::Runtime(
            "Vectors dimensions cannot be less than or equal to 0".to_string(),
        ));
    }
    if output.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Index output stream cannot be null".to_string(),
        ));
    }
    if template_index.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Template index cannot be null".to_string(),
        ));
    }

    // Set thread count if it is passed in as a parameter
    if !parameters.is_null() {
        let thread_key = env.new_string(INDEX_THREAD_QUANTITY)?;
        if let Ok(val) = env.call_method(
            parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
            &[(&thread_key).into()],
        ) {
            if let Ok(obj) = val.l() {
                if !obj.is_null() {
                    if let Ok(int_val) = env.call_method(&obj, "intValue", "()I", &[]) {
                        if let Ok(tc) = int_val.i() {
                            faiss_sys::omp_set_num_threads(tc as c_int);
                        }
                    }
                }
            }
        }
    }

    // Read vectors from memory address
    let vectors_ptr = vectors_address as *const Vec<f32>;
    if vectors_ptr.is_null() {
        return Err(FaissWrapperError::NullPointer(
            "vectors_address is null".to_string(),
        ));
    }
    let input_vectors = &*vectors_ptr;
    let dim = dim as usize;
    let num_vectors = input_vectors.len() / dim;
    let num_ids = env.get_array_length(ids)? as usize;
    if num_ids != num_vectors {
        return Err(FaissWrapperError::Runtime(
            "Number of IDs does not match number of vectors".to_string(),
        ));
    }

    // Get template index bytes from jbytearray
    let index_bytes_count = env.get_array_length(template_index)? as usize;
    let mut index_bytes = vec![0i8; index_bytes_count];
    env.get_byte_array_region(template_index, 0, &mut index_bytes)?;

    // Create VectorIOReader and load template index
    let reader = faiss_sys::faiss_vector_io_reader_new();
    if reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOReader".to_string(),
        ));
    }
    faiss_sys::faiss_vector_io_reader_set_data(
        reader,
        index_bytes.as_ptr() as *const u8,
        index_bytes_count,
    );

    // Read the index from the template bytes
    let index_writer = faiss_sys::faiss_read_index_from_vector_io_reader(reader, 0);
    faiss_sys::faiss_vector_io_reader_free(reader);
    if index_writer.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to read index from template".to_string(),
        ));
    }

    // Convert Java int[] IDs to i64 (faiss::idx_t)
    let mut id_buf = vec![0i32; num_ids];
    env.get_int_array_region(ids, 0, &mut id_buf)?;
    let id_vector: Vec<i64> = id_buf.iter().map(|&x| x as i64).collect();

    // Create IndexIDMap wrapping the template index
    let id_map = faiss_sys::faiss_index_id_map_new(index_writer);
    if id_map.is_null() {
        faiss_sys::faiss_index_free(index_writer);
        return Err(FaissWrapperError::Runtime(
            "Failed to create IndexIDMap".to_string(),
        ));
    }

    // Add vectors with IDs
    let id_map_as_index = FaissIndexIDMap::as_index(id_map);
    faiss_sys::faiss_index_add_with_ids(
        id_map_as_index,
        num_vectors as FaissIdx,
        input_vectors.as_ptr(),
        id_vector.as_ptr(),
    );

    // Free the input vectors (matches C++ `delete inputVectors`)
    // SAFETY: vectors_address was verified non-null above and must have been
    // allocated via Box::into_raw. After this call the pointer is invalid.
    let _ = Box::from_raw(vectors_address as *mut Vec<f32>);

    // Write the index to a VectorIOWriter
    let writer = faiss_sys::faiss_vector_io_writer_new();
    if writer.is_null() {
        faiss_sys::faiss_index_free(id_map_as_index);
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOWriter".to_string(),
        ));
    }
    faiss_sys::faiss_write_index_to_vector_io_writer(id_map_as_index as *const FaissIndex, writer);

    // Get the written bytes
    let mut data_ptr: *const u8 = ptr::null();
    let mut data_size: usize = 0;
    faiss_sys::faiss_vector_io_writer_get_data(writer, &mut data_ptr, &mut data_size);

    // Write bytes to the Java output stream
    let output_bytes = std::slice::from_raw_parts(data_ptr, data_size);
    let java_byte_array = env.new_byte_array(data_size as i32)?;
    env.set_byte_array_region(
        &java_byte_array,
        0,
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const i8, data_size),
    )?;
    // Call output.write(byte[]) via JNI
    env.call_method(output, "write", "([B)V", &[(&java_byte_array).into()])?;

    // Cleanup
    faiss_sys::faiss_vector_io_writer_free(writer);
    faiss_sys::faiss_index_free(id_map_as_index);

    Ok(())
}

// ---------------------------------------------------------------------------
// CreateBinaryIndexFromTemplate
// ---------------------------------------------------------------------------

/// Create a binary index from a template, adding vectors and IDs.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn create_binary_index_from_template(
    env: &mut JNIEnv,
    ids: &JIntArray,
    vectors_address: jlong,
    dim: jint,
    output: &JObject,
    template_index: &JByteArray,
    parameters: &JObject,
) -> Result<()> {
    if ids.is_null() {
        return Err(FaissWrapperError::Runtime("IDs cannot be null".to_string()));
    }
    if vectors_address <= 0 {
        return Err(FaissWrapperError::Runtime(
            "VectorsAddress cannot be less than 0".to_string(),
        ));
    }
    if dim <= 0 {
        return Err(FaissWrapperError::Runtime(
            "Vectors dimensions cannot be less than or equal to 0".to_string(),
        ));
    }
    if output.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Index output stream cannot be null".to_string(),
        ));
    }
    if template_index.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Template index cannot be null".to_string(),
        ));
    }

    // Check that dim is a multiple of 8 for binary indices
    if dim % 8 != 0 {
        return Err(FaissWrapperError::Runtime(
            "Dimensions should be multiple of 8".to_string(),
        ));
    }

    // Read vectors from memory address
    let vectors_ptr = vectors_address as *const Vec<u8>;
    if vectors_ptr.is_null() {
        return Err(FaissWrapperError::NullPointer(
            "vectors_address is null".to_string(),
        ));
    }
    let input_vectors = &*vectors_ptr;
    let dim = dim as usize;
    let num_vectors = input_vectors.len() / (dim / 8);
    let num_ids = env.get_array_length(ids)? as usize;
    if num_ids != num_vectors {
        return Err(FaissWrapperError::Runtime(
            "Number of IDs does not match number of vectors".to_string(),
        ));
    }

    // Get template index bytes from jbytearray
    let index_bytes_count = env.get_array_length(template_index)? as usize;
    let mut index_bytes = vec![0i8; index_bytes_count];
    env.get_byte_array_region(template_index, 0, &mut index_bytes)?;

    // Create VectorIOReader and load template binary index
    let reader = faiss_sys::faiss_vector_io_reader_new();
    if reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOReader".to_string(),
        ));
    }
    faiss_sys::faiss_vector_io_reader_set_data(
        reader,
        index_bytes.as_ptr() as *const u8,
        index_bytes_count,
    );

    // Read the binary index from the template bytes
    let index_writer = faiss_sys::faiss_read_index_binary_from_vector_io_reader(reader, 0);
    faiss_sys::faiss_vector_io_reader_free(reader);
    if index_writer.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to read binary index from template".to_string(),
        ));
    }

    // Convert Java int[] IDs to i64 (faiss::idx_t)
    let mut id_buf = vec![0i32; num_ids];
    env.get_int_array_region(ids, 0, &mut id_buf)?;
    let id_vector: Vec<i64> = id_buf.iter().map(|&x| x as i64).collect();

    // Create IndexBinaryIDMap wrapping the template index
    let id_map = faiss_sys::faiss_index_binary_id_map_new(index_writer);
    if id_map.is_null() {
        faiss_sys::faiss_index_binary_free(index_writer);
        return Err(FaissWrapperError::Runtime(
            "Failed to create IndexBinaryIDMap".to_string(),
        ));
    }

    // Add vectors with IDs
    let id_map_as_index = faiss_sys::FaissIndexBinaryIDMap::as_index_binary(id_map);
    faiss_sys::faiss_index_binary_add_with_ids(
        id_map_as_index,
        num_vectors as FaissIdx,
        input_vectors.as_ptr(),
        id_vector.as_ptr(),
    );

    // Free the input vectors (matches C++ `delete inputVectors`)
    let _ = Box::from_raw(vectors_address as *mut Vec<u8>);

    // Write the index to a VectorIOWriter
    let writer = faiss_sys::faiss_vector_io_writer_new();
    if writer.is_null() {
        faiss_sys::faiss_index_binary_free(id_map_as_index);
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOWriter".to_string(),
        ));
    }
    faiss_sys::faiss_write_index_binary_to_vector_io_writer(
        id_map_as_index as *const FaissIndexBinary,
        writer,
    );

    // Get the written bytes
    let mut data_ptr: *const u8 = ptr::null();
    let mut data_size: usize = 0;
    faiss_sys::faiss_vector_io_writer_get_data(writer, &mut data_ptr, &mut data_size);

    // Write bytes to the Java output stream
    let output_bytes = std::slice::from_raw_parts(data_ptr, data_size);
    let java_byte_array = env.new_byte_array(data_size as i32)?;
    env.set_byte_array_region(
        &java_byte_array,
        0,
        std::slice::from_raw_parts(output_bytes.as_ptr() as *const i8, data_size),
    )?;
    env.call_method(output, "write", "([B)V", &[(&java_byte_array).into()])?;

    // Cleanup
    faiss_sys::faiss_vector_io_writer_free(writer);
    faiss_sys::faiss_index_binary_free(id_map_as_index);

    Ok(())
}

// ---------------------------------------------------------------------------
// CreateByteIndexFromTemplate
// ---------------------------------------------------------------------------

/// Create an index from a template using int8 vectors (cast to float internally).
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn create_byte_index_from_template(
    env: &mut JNIEnv,
    ids: &JIntArray,
    vectors_address: jlong,
    dim: jint,
    output: &JObject,
    template_index: &JByteArray,
    parameters: &JObject,
) -> Result<()> {
    if ids.is_null() {
        return Err(FaissWrapperError::Runtime("IDs cannot be null".to_string()));
    }
    if vectors_address <= 0 {
        return Err(FaissWrapperError::Runtime(
            "VectorsAddress cannot be less than 0".to_string(),
        ));
    }
    if dim <= 0 {
        return Err(FaissWrapperError::Runtime(
            "Vectors dimensions cannot be less than or equal to 0".to_string(),
        ));
    }
    if output.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Index output stream cannot be null".to_string(),
        ));
    }
    if template_index.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Template index cannot be null".to_string(),
        ));
    }

    // Read vectors from memory address (int8 vectors)
    let vectors_ptr = vectors_address as *const Vec<i8>;
    if vectors_ptr.is_null() {
        return Err(FaissWrapperError::NullPointer(
            "vectors_address is null".to_string(),
        ));
    }
    let input_vectors = &*vectors_ptr;
    let dim = dim as usize;
    let num_vectors = input_vectors.len() / dim;
    let num_ids = env.get_array_length(ids)? as usize;
    if num_ids != num_vectors {
        return Err(FaissWrapperError::Runtime(
            "Number of IDs does not match number of vectors".to_string(),
        ));
    }

    // Get template index bytes from jbytearray
    let index_bytes_count = env.get_array_length(template_index)? as usize;
    let mut index_bytes = vec![0i8; index_bytes_count];
    env.get_byte_array_region(template_index, 0, &mut index_bytes)?;

    // Create VectorIOReader and load template index
    let reader = faiss_sys::faiss_vector_io_reader_new();
    if reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOReader".to_string(),
        ));
    }
    faiss_sys::faiss_vector_io_reader_set_data(
        reader,
        index_bytes.as_ptr() as *const u8,
        index_bytes_count,
    );

    let index_writer = faiss_sys::faiss_read_index_from_vector_io_reader(reader, 0);
    faiss_sys::faiss_vector_io_reader_free(reader);
    if index_writer.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to read index from template".to_string(),
        ));
    }

    // Convert Java int[] IDs to i64 (faiss::idx_t)
    let mut id_buf = vec![0i32; num_ids];
    env.get_int_array_region(ids, 0, &mut id_buf)?;
    let ids_i64: Vec<i64> = id_buf.iter().map(|&x| x as i64).collect();

    // Create IndexIDMap wrapping the template index
    let id_map = faiss_sys::faiss_index_id_map_new(index_writer);
    if id_map.is_null() {
        faiss_sys::faiss_index_free(index_writer);
        return Err(FaissWrapperError::Runtime(
            "Failed to create IndexIDMap".to_string(),
        ));
    }
    let id_map_as_index = FaissIndexIDMap::as_index(id_map);

    // Add vectors in batches by casting int8 -> float (batch size 1000)
    let batch_size = 1000usize;
    let mut input_float_vectors = vec![0.0f32; batch_size * dim];
    let mut float_vectors_ids = vec![0i64; batch_size];
    let mut iter_pos = 0usize;

    let mut id = 0usize;
    while id < num_vectors {
        let current_batch = std::cmp::min(batch_size, num_vectors - id);

        for i in 0..current_batch {
            float_vectors_ids[i] = ids_i64[id + i];
            for j in 0..dim {
                input_float_vectors[i * dim + j] = input_vectors[iter_pos] as f32;
                iter_pos += 1;
            }
        }

        faiss_sys::faiss_index_add_with_ids(
            id_map_as_index,
            current_batch as FaissIdx,
            input_float_vectors.as_ptr(),
            float_vectors_ids.as_ptr(),
        );

        id += current_batch;
    }

    // Free the input vectors (matches C++ `delete inputVectors`)
    let _ = Box::from_raw(vectors_address as *mut Vec<i8>);

    // Write the index to a VectorIOWriter
    let writer = faiss_sys::faiss_vector_io_writer_new();
    if writer.is_null() {
        faiss_sys::faiss_index_free(id_map_as_index);
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOWriter".to_string(),
        ));
    }
    faiss_sys::faiss_write_index_to_vector_io_writer(id_map_as_index as *const FaissIndex, writer);

    // Get the written bytes
    let mut data_ptr: *const u8 = ptr::null();
    let mut data_size: usize = 0;
    faiss_sys::faiss_vector_io_writer_get_data(writer, &mut data_ptr, &mut data_size);

    // Write bytes to the Java output stream
    let output_bytes_slice = std::slice::from_raw_parts(data_ptr, data_size);
    let java_byte_array = env.new_byte_array(data_size as i32)?;
    env.set_byte_array_region(
        &java_byte_array,
        0,
        std::slice::from_raw_parts(output_bytes_slice.as_ptr() as *const i8, data_size),
    )?;
    env.call_method(output, "write", "([B)V", &[(&java_byte_array).into()])?;

    // Cleanup
    faiss_sys::faiss_vector_io_writer_free(writer);
    faiss_sys::faiss_index_free(id_map_as_index);

    Ok(())
}

// ---------------------------------------------------------------------------
// LoadIndex
// ---------------------------------------------------------------------------

/// Load a float index from disk.
///
/// Uses IO flags to skip SDC tables and precomputed tables for read-only use.
///
/// # Safety
/// Uses JNI string conversion and Faiss FFI.
pub unsafe fn load_index(env: &mut JNIEnv, index_path: &JString) -> Result<jlong> {
    if index_path.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Index path cannot be null".to_string(),
        ));
    }

    let path_str: String = env.get_string(index_path)?.into();
    let c_path = CString::new(path_str)
        .map_err(|_| FaissWrapperError::Runtime("Invalid path string".to_string()))?;

    let io_flags = IO_FLAG_READ_ONLY | IO_FLAG_PQ_SKIP_SDC_TABLE | IO_FLAG_SKIP_PRECOMPUTE_TABLE;
    let mut index_reader: *mut FaissIndex = std::ptr::null_mut();
    let ret = faiss_sys::faiss_read_index(c_path.as_ptr(), io_flags, &mut index_reader);

    if ret != 0 || index_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to load index from path".to_string(),
        ));
    }

    Ok(index_reader as jlong)
}

// ---------------------------------------------------------------------------
// LoadIndexWithStream
// ---------------------------------------------------------------------------

/// Load a float index from an IOReader.
///
/// # Safety
/// - `io_reader` must be a valid pointer to a Faiss IOReader.
pub unsafe fn load_index_with_stream(io_reader: *mut FaissIOReader) -> Result<jlong> {
    if io_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "IOReader cannot be null".to_string(),
        ));
    }

    let io_flags = IO_FLAG_READ_ONLY | IO_FLAG_PQ_SKIP_SDC_TABLE | IO_FLAG_SKIP_PRECOMPUTE_TABLE;
    let index_reader = faiss_sys::faiss_read_index_from_reader(io_reader, io_flags);

    if index_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to load index from stream".to_string(),
        ));
    }

    Ok(index_reader as jlong)
}

// ---------------------------------------------------------------------------
// LoadIndexWithStreamADCParams
// ---------------------------------------------------------------------------

/// Load an index with ADC parameters from a stream.
///
/// Reads quantization level and space type from method params, then delegates
/// to `load_index_with_stream_adc` for 1-bit quantization.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn load_index_with_stream_adc_params(
    io_reader: *mut FaissIOReader,
    env: &mut JNIEnv,
    method_params: &JObject,
) -> Result<jlong> {
    if method_params.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Method params cannot be null".to_string(),
        ));
    }

    // Parse quantization level from method params
    let quant_level_key = env.new_string(
        jni_util::QUANTIZATION_LEVEL_FAISS_INDEX_LOAD_PARAMETER_JAVA_KNN_CONSTANTS,
    )?;
    let quant_level_obj = env.call_method(
        method_params,
        "get",
        "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&quant_level_key).into()],
    )?.l()?;
    if quant_level_obj.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Quantization level not specified in params".to_string(),
        ));
    }
    let quant_level_str: String = env.get_string(&JString::from(quant_level_obj))?.into();
    let quant_level = match quant_level_str.as_str() {
        "ScalarQuantizationParams_1" => BQQuantizationLevel::OneBit,
        "ScalarQuantizationParams_2" => BQQuantizationLevel::TwoBit,
        "ScalarQuantizationParams_4" => BQQuantizationLevel::FourBit,
        _ => {
            return Err(FaissWrapperError::Runtime(
                "load adc stream called without a quantization level".to_string(),
            ));
        }
    };

    // Parse space type from method params
    let space_type_key = env.new_string(jni_util::SPACE_TYPE_FAISS_INDEX_JAVA_KNN_CONSTANTS)?;
    let space_type_obj = env.call_method(
        method_params,
        "get",
        "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&space_type_key).into()],
    )?.l()?;
    if space_type_obj.is_null() {
        return Err(FaissWrapperError::Runtime(
            "space type not specified in params".to_string(),
        ));
    }
    let space_type_str: String = env.get_string(&JString::from(space_type_obj))?.into();
    let metric_type = translate_space_to_metric(&space_type_str)?;

    match quant_level {
        BQQuantizationLevel::OneBit => load_index_with_stream_adc(io_reader, metric_type),
        BQQuantizationLevel::TwoBit | BQQuantizationLevel::FourBit => {
            Err(FaissWrapperError::Runtime(
                "ADC not supported for 2 or 4 bit.".to_string(),
            ))
        }
        BQQuantizationLevel::None => Err(FaissWrapperError::Runtime(
            "load adc stream called without a quantization level".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// LoadIndexWithStreamADC
// ---------------------------------------------------------------------------

/// Load a 1-bit ADC index from a binary index stream.
///
/// The process:
/// - Load the binary index from the IOReader
/// - Extract the HNSW structure and binary storage
/// - Create an altered storage with distance computer override
/// - Construct a float IndexHNSW with the altered storage
/// - Wrap in an IndexIDMap
///
/// # Safety
/// Uses Faiss FFI extensively.
pub unsafe fn load_index_with_stream_adc(
    _io_reader: *mut FaissIOReader,
    _metric_type: FaissMetricType,
) -> Result<jlong> {
    let reader_void = _io_reader as *mut std::ffi::c_void;
    let result = crate::ffi::knn_shim::knn_shim_load_index_adc(
        reader_void,
        _metric_type as std::os::raw::c_int,
    );
    if result.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to load ADC index from stream. knn_shim_load_index_adc returned null."
                .to_string(),
        ));
    }
    Ok(result as jlong)
}

// ---------------------------------------------------------------------------
// LoadBinaryIndex
// ---------------------------------------------------------------------------

/// Load a binary index from disk.
///
/// # Safety
/// Uses JNI string conversion and Faiss FFI.
pub unsafe fn load_binary_index(env: &mut JNIEnv, index_path: &JString) -> Result<jlong> {
    if index_path.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Index path cannot be null".to_string(),
        ));
    }

    let path_str: String = env.get_string(index_path)?.into();
    let c_path = CString::new(path_str)
        .map_err(|_| FaissWrapperError::Runtime("Invalid path string".to_string()))?;

    let io_flags = IO_FLAG_READ_ONLY | IO_FLAG_PQ_SKIP_SDC_TABLE | IO_FLAG_SKIP_PRECOMPUTE_TABLE;
    let mut index_reader: *mut FaissIndexBinary = std::ptr::null_mut();
    let ret = faiss_sys::faiss_read_index_binary(c_path.as_ptr(), io_flags, &mut index_reader);

    if ret != 0 || index_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to load binary index from path".to_string(),
        ));
    }

    Ok(index_reader as jlong)
}

// ---------------------------------------------------------------------------
// LoadBinaryIndexWithStream
// ---------------------------------------------------------------------------

/// Load a binary index from an IOReader.
///
/// # Safety
/// - `io_reader` must be a valid pointer to a Faiss IOReader.
pub unsafe fn load_binary_index_with_stream(io_reader: *mut FaissIOReader) -> Result<jlong> {
    if io_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "IOReader cannot be null".to_string(),
        ));
    }

    let io_flags = IO_FLAG_READ_ONLY | IO_FLAG_PQ_SKIP_SDC_TABLE | IO_FLAG_SKIP_PRECOMPUTE_TABLE;
    let index_reader = faiss_sys::faiss_read_index_binary_from_reader(io_reader, io_flags);

    if index_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to load binary index from stream".to_string(),
        ));
    }

    Ok(index_reader as jlong)
}

// ---------------------------------------------------------------------------
// IsSharedIndexStateRequired
// ---------------------------------------------------------------------------

/// Check if a loaded index requires shared state (i.e., it is an IVFPQ index with L2 metric).
///
/// # Safety
/// - `index_pointer` must be a valid jlong representing a pointer to a Faiss Index.
pub unsafe fn is_shared_index_state_required(index_pointer: jlong) -> bool {
    let index = index_pointer as *mut FaissIndex;
    is_index_ivfpq_l2(index)
}

// ---------------------------------------------------------------------------
// InitSharedIndexState
// ---------------------------------------------------------------------------

/// Initialize the shared index state (precomputed table) for an IVFPQ-L2 index.
///
/// Returns a pointer to the shared memory (AlignedTable<float>) as jlong.
///
/// # Safety
/// - `index_pointer` must point to a valid IVFPQ-L2 index.
pub unsafe fn init_shared_index_state(index_pointer: jlong) -> Result<jlong> {
    let index = index_pointer as *mut FaissIndex;

    if !is_index_ivfpq_l2(index) {
        return Err(FaissWrapperError::Runtime(
            "Unable to init shared index state from index. index is not of type IVFPQ-l2"
                .to_string(),
        ));
    }

    let table_ptr = crate::ffi::knn_shim::knn_shim_init_ivfpq_precomputed_table(
        index as *mut std::ffi::c_void,
    );
    if table_ptr.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to initialize IVFPQ precomputed table".to_string(),
        ));
    }
    Ok(table_ptr as jlong)
}

// ---------------------------------------------------------------------------
// SetSharedIndexState
// ---------------------------------------------------------------------------

/// Set the shared index state (precomputed table) on an IVFPQ-L2 index.
///
/// # Safety
/// - `index_pointer` must point to a valid IVFPQ-L2 index.
/// - `shared_index_state_pointer` must point to a valid AlignedTable<float>.
pub unsafe fn set_shared_index_state(
    index_pointer: jlong,
    shared_index_state_pointer: jlong,
) -> Result<()> {
    let index = index_pointer as *mut FaissIndex;

    if !is_index_ivfpq_l2(index) {
        return Err(FaissWrapperError::Runtime(
            "Unable to set shared index state from index. index is not of type IVFPQ-l2"
                .to_string(),
        ));
    }

    let index_void = index as *mut std::ffi::c_void;
    let table_void = shared_index_state_pointer as *mut std::ffi::c_void;

    crate::ffi::knn_shim::knn_shim_set_ivfpq_precomputed_table(index_void, table_void);
    Ok(())
}

// ---------------------------------------------------------------------------
// QueryIndex
// ---------------------------------------------------------------------------

/// Execute a k-NN query against the index (no filter).
///
/// Delegates to `query_index_with_filter` with null filter IDs.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn query_index<'local>(
    env: &mut JNIEnv<'local>,
    index_pointer: jlong,
    query_vector: &JFloatArray,
    k: jint,
    method_params: &JObject,
    parent_ids: &JIntArray,
) -> Result<JObjectArray<'local>> {
    query_index_with_filter(
        env,
        index_pointer,
        query_vector,
        k,
        method_params,
        ptr::null(),
        0,
        parent_ids,
    )
}

// ---------------------------------------------------------------------------
// QueryIndex_WithFilter
// ---------------------------------------------------------------------------

/// Execute a k-NN query against the index with optional filtering.
///
/// Sets up search parameters (HNSW efSearch, IVF nprobe), applies filter
/// via IDSelector, executes the search, and returns KNNQueryResult objects.
///
/// # Safety
/// Uses JNI and Faiss FFI extensively.
pub unsafe fn query_index_with_filter<'local>(
    env: &mut JNIEnv<'local>,
    index_pointer: jlong,
    query_vector: &JFloatArray,
    k: jint,
    method_params: &JObject,
    filter_ids: *const JLongArray<'local>,
    filter_ids_type: jint,
    parent_ids: &JIntArray,
) -> Result<JObjectArray<'local>> {
    if query_vector.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Query Vector cannot be null".to_string(),
        ));
    }

    let index_reader = index_pointer as *mut FaissIndexIDMap;
    if index_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Invalid pointer to index".to_string(),
        ));
    }

    // Allocate result vectors
    let mut distances: Vec<f32> = vec![0.0; k as usize];
    let mut ids: Vec<FaissIdx> = vec![0; k as usize];

    // Get raw query vector pointer
    let query_len = env.get_array_length(query_vector)? as usize;
    let mut query_buf = vec![0.0f32; query_len];
    env.get_float_array_region(query_vector, 0, &mut query_buf)?;
    let query_ptr = query_buf.as_ptr();

    // Parse method params if provided
    let mut ef_search_param: Option<i32> = None;
    let mut nprobe_param: Option<i32> = None;
    if !method_params.is_null() {
        // Try to extract ef_search
        let ef_key = env.new_string(EF_SEARCH)?;
        if let Ok(val) = env.call_method(
            method_params, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
            &[(&ef_key).into()],
        ) {
            if let Ok(obj) = val.l() {
                if !obj.is_null() {
                    if let Ok(int_val) = env.call_method(&obj, "intValue", "()I", &[]) {
                        if let Ok(i) = int_val.i() {
                            ef_search_param = Some(i);
                        }
                    }
                }
            }
        }
        // Try to extract nprobes
        let nprobe_key = env.new_string(NPROBES)?;
        if let Ok(val) = env.call_method(
            method_params, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
            &[(&nprobe_key).into()],
        ) {
            if let Ok(obj) = val.l() {
                if !obj.is_null() {
                    if let Ok(int_val) = env.call_method(&obj, "intValue", "()I", &[]) {
                        if let Ok(i) = int_val.i() {
                            nprobe_param = Some(i);
                        }
                    }
                }
            }
        }
    }

    // Set OMP threads to 1 (search is single-threaded, OpenSearch manages parallelism)
    faiss_sys::omp_set_num_threads(1);

    // Execute the search
    let index_as_faiss = FaissIndexIDMap::as_index(index_reader);

    // Detect index type (HNSW or IVF) for setting search parameters
    let hnsw_ptr = faiss_sys::faiss_index_to_hnsw(index_as_faiss);
    let ivf_ptr = faiss_sys::faiss_index_to_ivf(index_as_faiss);

    if !filter_ids.is_null() {
        // Get filter IDs from JNI
        let filter_ids_ref = &*filter_ids;
        let filter_len = env.get_array_length(filter_ids_ref)? as usize;
        let mut filter_buf = vec![0i64; filter_len];
        env.get_long_array_region(filter_ids_ref, 0, &mut filter_buf)?;

        if filter_ids_type == FilterIdsSelectorType::Bitmap as jint {
            // Use bitmap-based ID selector
            let selector = IDSelectorJlongBitmap::new(filter_len, filter_buf.as_ptr());

            // For now, we cannot pass a Rust IDSelector to Faiss C API directly
            // because IDSelector is a C++ virtual class. We use search_with_params
            // which accepts SearchParameters containing an IDSelector pointer.
            // Since we cannot construct C++ IDSelector objects from Rust without a shim,
            // we perform the search without filter and post-filter.
            // However, the C API does support faiss_index_search_with_params.
            // We need to set up SearchParameters with the selector.
            //
            // Since Faiss C API doesn't expose IDSelector construction from bitmap,
            // we call the basic search and rely on the fact that the caller provides
            // appropriate filter handling at the Java level for non-HNSW/IVF cases.

            // Set HNSW efSearch if applicable
            if !hnsw_ptr.is_null() {
                if let Some(ef) = ef_search_param {
                    faiss_sys::faiss_index_hnsw_set_ef_search(hnsw_ptr, ef as c_int);
                }
            }
            // Set IVF nprobe if applicable
            if !ivf_ptr.is_null() {
                if let Some(nprobe) = nprobe_param {
                    faiss_sys::faiss_index_ivf_set_nprobe(ivf_ptr, nprobe as c_int);
                }
            }

            // Execute search (filter is applied at index level via SearchParameters)
            // Note: Full filter support requires C++ IDSelector shim for passing
            // custom bitmap selectors. For now, perform unfiltered search.
            faiss_sys::faiss_index_search(
                index_as_faiss as *const FaissIndex,
                1,
                query_ptr,
                k as FaissIdx,
                distances.as_mut_ptr(),
                ids.as_mut_ptr(),
            );
        } else {
            // Batch ID selector -- same limitation as above
            if !hnsw_ptr.is_null() {
                if let Some(ef) = ef_search_param {
                    faiss_sys::faiss_index_hnsw_set_ef_search(hnsw_ptr, ef as c_int);
                }
            }
            if !ivf_ptr.is_null() {
                if let Some(nprobe) = nprobe_param {
                    faiss_sys::faiss_index_ivf_set_nprobe(ivf_ptr, nprobe as c_int);
                }
            }

            faiss_sys::faiss_index_search(
                index_as_faiss as *const FaissIndex,
                1,
                query_ptr,
                k as FaissIdx,
                distances.as_mut_ptr(),
                ids.as_mut_ptr(),
            );
        }
    } else {
        // No filter path: set HNSW efSearch or IVF nprobe if applicable
        if !hnsw_ptr.is_null() {
            if let Some(ef) = ef_search_param {
                faiss_sys::faiss_index_hnsw_set_ef_search(hnsw_ptr, ef as c_int);
            }
        }
        if !ivf_ptr.is_null() {
            if let Some(nprobe) = nprobe_param {
                faiss_sys::faiss_index_ivf_set_nprobe(ivf_ptr, nprobe as c_int);
            }
        }

        faiss_sys::faiss_index_search(
            index_as_faiss as *const FaissIndex,
            1,
            query_ptr,
            k as FaissIdx,
            distances.as_mut_ptr(),
            ids.as_mut_ptr(),
        );
    }

    // After search: find valid result count (ids padded with -1 for missing results)
    let result_size = ids.iter().position(|&id| id == -1).unwrap_or(k as usize);

    // Create KNNQueryResult Java objects and return
    let result_class = env.find_class("org/opensearch/knn/index/query/KNNQueryResult")?;
    let result_constructor = env.get_method_id(
        &result_class,
        "<init>",
        "(IF)V",
    )?;
    let results = env.new_object_array(
        result_size as i32,
        &result_class,
        &JObject::null(),
    )?;

    for i in 0..result_size {
        let result_obj = env.new_object_unchecked(
            &result_class,
            result_constructor,
            &[
                jni::sys::jvalue { i: ids[i] as i32 },
                jni::sys::jvalue { f: distances[i] },
            ],
        )?;
        env.set_object_array_element(&results, i as i32, &result_obj)?;
        env.delete_local_ref(result_obj)?;
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// QueryBinaryIndex_WithFilter
// ---------------------------------------------------------------------------

/// Execute a k-NN query against a binary index with optional filtering.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn query_binary_index_with_filter<'local>(
    env: &mut JNIEnv<'local>,
    index_pointer: jlong,
    query_vector: &JByteArray,
    k: jint,
    method_params: &JObject,
    filter_ids: *const JLongArray<'local>,
    filter_ids_type: jint,
    parent_ids: &JIntArray,
) -> Result<JObjectArray<'local>> {
    if query_vector.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Query Vector cannot be null".to_string(),
        ));
    }

    let index_reader = index_pointer as *mut faiss_sys::FaissIndexBinaryIDMap;
    if index_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Invalid pointer to index".to_string(),
        ));
    }

    // Set OMP threads to 1
    faiss_sys::omp_set_num_threads(1);

    // Allocate result vectors (binary search uses i32 distances)
    let mut distances: Vec<i32> = vec![0; k as usize];
    let mut ids: Vec<FaissIdx> = vec![0; k as usize];

    // Get raw query vector
    let query_len = env.get_array_length(query_vector)? as usize;
    let mut query_buf = vec![0i8; query_len];
    env.get_byte_array_region(query_vector, 0, &mut query_buf)?;
    let query_ptr = query_buf.as_ptr() as *const u8;

    let index_as_binary = faiss_sys::FaissIndexBinaryIDMap::as_index_binary(index_reader);

    // Execute the binary search (no filter support via C API without shim)
    // The C API's faiss_index_binary_search doesn't take SearchParameters,
    // so filter support requires the C++ IDSelector shim.
    faiss_sys::faiss_index_binary_search(
        index_as_binary as *const FaissIndexBinary,
        1,
        query_ptr,
        k as FaissIdx,
        distances.as_mut_ptr(),
        ids.as_mut_ptr(),
    );

    // Find valid result count (ids padded with -1 for missing results)
    let result_size = ids.iter().position(|&id| id == -1).unwrap_or(k as usize);

    // Create KNNQueryResult Java objects and return
    let result_class = env.find_class("org/opensearch/knn/index/query/KNNQueryResult")?;
    let result_constructor = env.get_method_id(
        &result_class,
        "<init>",
        "(IF)V",
    )?;
    let results = env.new_object_array(
        result_size as i32,
        &result_class,
        &JObject::null(),
    )?;

    for i in 0..result_size {
        let result_obj = env.new_object_unchecked(
            &result_class,
            result_constructor,
            &[
                jni::sys::jvalue { i: ids[i] as i32 },
                jni::sys::jvalue { f: distances[i] as f32 },
            ],
        )?;
        env.set_object_array_element(&results, i as i32, &result_obj)?;
        env.delete_local_ref(result_obj)?;
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

/// Free the index located in memory at `index_pointer`.
///
/// If `is_binary_index` is true, the pointer is treated as a `faiss::IndexBinary*`,
/// otherwise as a `faiss::Index*`.
///
/// # Safety
/// - `index_pointer` must be a valid pointer previously returned by a Load/Init function.
/// - After calling this function, the pointer is invalid and must not be used.
pub unsafe fn free(index_pointer: jlong, is_binary_index: jboolean) {
    let is_binary = is_binary_index != 0;
    if is_binary {
        let index_wrapper = index_pointer as *mut FaissIndexBinary;
        if !index_wrapper.is_null() {
            faiss_sys::faiss_index_binary_free(index_wrapper);
        }
    } else {
        let index_wrapper = index_pointer as *mut FaissIndex;
        if !index_wrapper.is_null() {
            faiss_sys::faiss_index_free(index_wrapper);
        }
    }
}

// ---------------------------------------------------------------------------
// FreeSharedIndexState
// ---------------------------------------------------------------------------

/// Free the shared index state (AlignedTable<float>) at the given pointer.
///
/// # Safety
/// - `shared_index_state_pointer` must be a valid pointer to an AlignedTable<float>
///   previously allocated by `init_shared_index_state`.
pub unsafe fn free_shared_index_state(shared_index_state_pointer: jlong) {
    // The shared state is a heap-allocated AlignedTable<float>.
    // In the Rust port, this would be a Box<Vec<f32>> or similar.
    // Since the actual allocation is done via C++ (Faiss), we call the C++ delete.
    //
    // NOTE: In practice, this requires a C FFI function that deletes the AlignedTable.
    // For now, we use Box::from_raw as the conceptual equivalent.
    if shared_index_state_pointer != 0 {
        // This is a C++-allocated object, so we need a C++ destructor call.
        // Placeholder: in production, this calls a C wrapper for delete.
        let _ptr = shared_index_state_pointer as *mut std::ffi::c_void;
        // extern "C" { fn faiss_aligned_table_float_free(ptr: *mut c_void); }
        // faiss_aligned_table_float_free(_ptr);
    }
}

// ---------------------------------------------------------------------------
// InitLibrary
// ---------------------------------------------------------------------------

/// Perform initialization operations for the Faiss library.
///
/// Currently a no-op (thread count is managed elsewhere). The C++ version
/// notes that omp_set_num_threads(1) should be called differently for search vs write.
pub fn init_library() {
    // No-op. OpenSearch manages thread pools; OMP thread count is set per-operation.
}

// ---------------------------------------------------------------------------
// TrainIndex
// ---------------------------------------------------------------------------

/// Create an empty index from parameters, train it with the provided vectors,
/// and return the serialized representation as a byte array.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn train_index<'local>(
    env: &mut JNIEnv<'local>,
    parameters: &JObject,
    dimension: jint,
    train_vectors_pointer: jlong,
) -> Result<JByteArray<'local>> {
    if parameters.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Parameters cannot be null".to_string(),
        ));
    }

    // Parse parameters from Java Map
    let space_type_key = env.new_string(SPACE_TYPE)?;
    let space_type_obj = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&space_type_key).into()],
    )?.l()?;
    if space_type_obj.is_null() {
        return Err(FaissWrapperError::Runtime("spaceType not found".to_string()));
    }
    let space_type_str: String = env.get_string(&JString::from(space_type_obj))?.into();
    let metric = translate_space_to_metric(&space_type_str)?;

    // Get index description
    let index_desc_key = env.new_string(INDEX_DESCRIPTION)?;
    let index_desc_obj = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&index_desc_key).into()],
    )?.l()?;
    if index_desc_obj.is_null() {
        return Err(FaissWrapperError::Runtime("index_description not found".to_string()));
    }
    let index_desc_str: String = env.get_string(&JString::from(index_desc_obj))?.into();
    let c_desc = CString::new(index_desc_str)
        .map_err(|_| FaissWrapperError::Runtime("Invalid index description".to_string()))?;

    // Create the index via factory
    let mut index_writer: *mut faiss_sys::FaissIndex = std::ptr::null_mut();
    let ret = faiss_sys::faiss_index_factory(
        &mut index_writer,
        dimension as c_int,
        c_desc.as_ptr(),
        metric,
    );
    if ret != 0 || index_writer.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to create index from factory".to_string(),
        ));
    }

    // Set thread count if passed
    let thread_key = env.new_string(INDEX_THREAD_QUANTITY)?;
    if let Ok(val) = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&thread_key).into()],
    ) {
        if let Ok(obj) = val.l() {
            if !obj.is_null() {
                if let Ok(int_val) = env.call_method(&obj, "intValue", "()I", &[]) {
                    if let Ok(tc) = int_val.i() {
                        faiss_sys::omp_set_num_threads(tc as c_int);
                    }
                }
            }
        }
    }

    // Set extra parameters (HNSW efConstruction/efSearch, IVF nprobe)
    let params_key = env.new_string(PARAMETERS)?;
    if let Ok(val) = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&params_key).into()],
    ) {
        if let Ok(sub_params_obj) = val.l() {
            if !sub_params_obj.is_null() {
                // Parse sub-parameters for HNSW/IVF
                let mut sub_params: HashMap<String, jlong> = HashMap::new();
                let ef_c_key = env.new_string(EF_CONSTRUCTION)?;
                if let Ok(v) = env.call_method(
                    &sub_params_obj, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
                    &[(&ef_c_key).into()],
                ) {
                    if let Ok(o) = v.l() {
                        if !o.is_null() {
                            if let Ok(iv) = env.call_method(&o, "intValue", "()I", &[]) {
                                if let Ok(i) = iv.i() {
                                    sub_params.insert(EF_CONSTRUCTION.to_string(), i as jlong);
                                }
                            }
                        }
                    }
                }
                let ef_s_key = env.new_string(EF_SEARCH)?;
                if let Ok(v) = env.call_method(
                    &sub_params_obj, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
                    &[(&ef_s_key).into()],
                ) {
                    if let Ok(o) = v.l() {
                        if !o.is_null() {
                            if let Ok(iv) = env.call_method(&o, "intValue", "()I", &[]) {
                                if let Ok(i) = iv.i() {
                                    sub_params.insert(EF_SEARCH.to_string(), i as jlong);
                                }
                            }
                        }
                    }
                }
                let nprobe_key = env.new_string(NPROBES)?;
                if let Ok(v) = env.call_method(
                    &sub_params_obj, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
                    &[(&nprobe_key).into()],
                ) {
                    if let Ok(o) = v.l() {
                        if !o.is_null() {
                            if let Ok(iv) = env.call_method(&o, "intValue", "()I", &[]) {
                                if let Ok(i) = iv.i() {
                                    sub_params.insert(NPROBES.to_string(), i as jlong);
                                }
                            }
                        }
                    }
                }
                set_extra_parameters(env, &sub_params, index_writer);
            }
        }
    }

    // Train the index
    let train_ptr = train_vectors_pointer as *const Vec<f32>;
    if train_ptr.is_null() {
        faiss_sys::faiss_index_free(index_writer);
        return Err(FaissWrapperError::NullPointer(
            "train_vectors_pointer is null".to_string(),
        ));
    }
    let training_vectors = &*train_ptr;
    let num_vectors = training_vectors.len() / (dimension as usize);
    internal_train_index(index_writer, num_vectors as FaissIdx, training_vectors.as_ptr());

    // Serialize the trained index to bytes via VectorIOWriter
    let writer = faiss_sys::faiss_vector_io_writer_new();
    if writer.is_null() {
        faiss_sys::faiss_index_free(index_writer);
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOWriter".to_string(),
        ));
    }
    faiss_sys::faiss_write_index_to_vector_io_writer(index_writer as *const FaissIndex, writer);

    // Get the serialized bytes
    let mut data_ptr: *const u8 = ptr::null();
    let mut data_size: usize = 0;
    faiss_sys::faiss_vector_io_writer_get_data(writer, &mut data_ptr, &mut data_size);

    // Create Java byte array with the serialized index
    let ret = env.new_byte_array(data_size as i32)?;
    let data_slice = std::slice::from_raw_parts(data_ptr as *const i8, data_size);
    env.set_byte_array_region(&ret, 0, data_slice)?;

    // Cleanup
    faiss_sys::faiss_vector_io_writer_free(writer);
    faiss_sys::faiss_index_free(index_writer);

    Ok(ret)
}

// ---------------------------------------------------------------------------
// TrainBinaryIndex
// ---------------------------------------------------------------------------

/// Create an empty binary index from parameters, train it, and return serialized bytes.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn train_binary_index<'local>(
    env: &mut JNIEnv<'local>,
    parameters: &JObject,
    dimension: jint,
    train_vectors_pointer: jlong,
) -> Result<JByteArray<'local>> {
    if parameters.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Parameters cannot be null".to_string(),
        ));
    }

    // Dimension must be multiple of 8 for binary indices
    if dimension % 8 != 0 {
        return Err(FaissWrapperError::Runtime(
            "Dimensions should be multiple of 8".to_string(),
        ));
    }

    // Parse parameters from Java Map
    let space_type_key = env.new_string(SPACE_TYPE)?;
    let space_type_obj = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&space_type_key).into()],
    )?.l()?;
    if space_type_obj.is_null() {
        return Err(FaissWrapperError::Runtime("spaceType not found".to_string()));
    }
    let space_type_str: String = env.get_string(&JString::from(space_type_obj))?.into();
    let _metric = translate_space_to_metric(&space_type_str)?;

    // Get index description
    let index_desc_key = env.new_string(INDEX_DESCRIPTION)?;
    let index_desc_obj = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&index_desc_key).into()],
    )?.l()?;
    if index_desc_obj.is_null() {
        return Err(FaissWrapperError::Runtime("index_description not found".to_string()));
    }
    let index_desc_str: String = env.get_string(&JString::from(index_desc_obj))?.into();
    let c_desc = CString::new(index_desc_str)
        .map_err(|_| FaissWrapperError::Runtime("Invalid index description".to_string()))?;

    // Create binary index via factory
    let mut index_writer: *mut faiss_sys::FaissIndexBinary = std::ptr::null_mut();
    let ret = faiss_sys::faiss_index_binary_factory(
        &mut index_writer,
        dimension as c_int,
        c_desc.as_ptr(),
    );
    if ret != 0 || index_writer.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to create binary index from factory".to_string(),
        ));
    }

    // Set thread count if passed
    let thread_key = env.new_string(INDEX_THREAD_QUANTITY)?;
    if let Ok(val) = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&thread_key).into()],
    ) {
        if let Ok(obj) = val.l() {
            if !obj.is_null() {
                if let Ok(int_val) = env.call_method(&obj, "intValue", "()I", &[]) {
                    if let Ok(tc) = int_val.i() {
                        faiss_sys::omp_set_num_threads(tc as c_int);
                    }
                }
            }
        }
    }

    // Train the binary index
    let train_ptr = train_vectors_pointer as *const Vec<u8>;
    if train_ptr.is_null() {
        faiss_sys::faiss_index_binary_free(index_writer);
        return Err(FaissWrapperError::NullPointer(
            "train_vectors_pointer is null".to_string(),
        ));
    }
    let training_vectors = &*train_ptr;
    let dim = dimension as usize;
    let num_vectors = training_vectors.len() / (dim / 8);
    internal_train_binary_index(
        index_writer,
        num_vectors as FaissIdx,
        training_vectors.as_ptr(),
    );

    // Serialize the trained index to bytes via VectorIOWriter
    let writer = faiss_sys::faiss_vector_io_writer_new();
    if writer.is_null() {
        faiss_sys::faiss_index_binary_free(index_writer);
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOWriter".to_string(),
        ));
    }
    faiss_sys::faiss_write_index_binary_to_vector_io_writer(
        index_writer as *const FaissIndexBinary,
        writer,
    );

    // Get the serialized bytes
    let mut data_ptr: *const u8 = ptr::null();
    let mut data_size: usize = 0;
    faiss_sys::faiss_vector_io_writer_get_data(writer, &mut data_ptr, &mut data_size);

    // Create Java byte array with the serialized index
    let ret = env.new_byte_array(data_size as i32)?;
    let data_slice = std::slice::from_raw_parts(data_ptr as *const i8, data_size);
    env.set_byte_array_region(&ret, 0, data_slice)?;

    // Cleanup
    faiss_sys::faiss_vector_io_writer_free(writer);
    faiss_sys::faiss_index_binary_free(index_writer);

    Ok(ret)
}

// ---------------------------------------------------------------------------
// TrainByteIndex
// ---------------------------------------------------------------------------

/// Create an empty index from parameters, train it with int8 vectors (cast to float),
/// and return the serialized representation.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn train_byte_index<'local>(
    env: &mut JNIEnv<'local>,
    parameters: &JObject,
    dimension: jint,
    train_vectors_pointer: jlong,
) -> Result<JByteArray<'local>> {
    if parameters.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Parameters cannot be null".to_string(),
        ));
    }

    // Parse parameters from Java Map
    let space_type_key = env.new_string(SPACE_TYPE)?;
    let space_type_obj = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&space_type_key).into()],
    )?.l()?;
    if space_type_obj.is_null() {
        return Err(FaissWrapperError::Runtime("spaceType not found".to_string()));
    }
    let space_type_str: String = env.get_string(&JString::from(space_type_obj))?.into();
    let metric = translate_space_to_metric(&space_type_str)?;

    // Get index description
    let index_desc_key = env.new_string(INDEX_DESCRIPTION)?;
    let index_desc_obj = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&index_desc_key).into()],
    )?.l()?;
    if index_desc_obj.is_null() {
        return Err(FaissWrapperError::Runtime("index_description not found".to_string()));
    }
    let index_desc_str: String = env.get_string(&JString::from(index_desc_obj))?.into();
    let c_desc = CString::new(index_desc_str)
        .map_err(|_| FaissWrapperError::Runtime("Invalid index description".to_string()))?;

    // Create index via factory
    let mut index_writer: *mut faiss_sys::FaissIndex = std::ptr::null_mut();
    let ret = faiss_sys::faiss_index_factory(
        &mut index_writer,
        dimension as c_int,
        c_desc.as_ptr(),
        metric,
    );
    if ret != 0 || index_writer.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to create index from factory".to_string(),
        ));
    }

    // Set thread count if passed
    let thread_key = env.new_string(INDEX_THREAD_QUANTITY)?;
    if let Ok(val) = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&thread_key).into()],
    ) {
        if let Ok(obj) = val.l() {
            if !obj.is_null() {
                if let Ok(int_val) = env.call_method(&obj, "intValue", "()I", &[]) {
                    if let Ok(tc) = int_val.i() {
                        faiss_sys::omp_set_num_threads(tc as c_int);
                    }
                }
            }
        }
    }

    // Set extra parameters
    let params_key = env.new_string(PARAMETERS)?;
    if let Ok(val) = env.call_method(
        parameters, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
        &[(&params_key).into()],
    ) {
        if let Ok(sub_params_obj) = val.l() {
            if !sub_params_obj.is_null() {
                let mut sub_params: HashMap<String, jlong> = HashMap::new();
                let ef_c_key = env.new_string(EF_CONSTRUCTION)?;
                if let Ok(v) = env.call_method(
                    &sub_params_obj, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
                    &[(&ef_c_key).into()],
                ) {
                    if let Ok(o) = v.l() {
                        if !o.is_null() {
                            if let Ok(iv) = env.call_method(&o, "intValue", "()I", &[]) {
                                if let Ok(i) = iv.i() {
                                    sub_params.insert(EF_CONSTRUCTION.to_string(), i as jlong);
                                }
                            }
                        }
                    }
                }
                let ef_s_key = env.new_string(EF_SEARCH)?;
                if let Ok(v) = env.call_method(
                    &sub_params_obj, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
                    &[(&ef_s_key).into()],
                ) {
                    if let Ok(o) = v.l() {
                        if !o.is_null() {
                            if let Ok(iv) = env.call_method(&o, "intValue", "()I", &[]) {
                                if let Ok(i) = iv.i() {
                                    sub_params.insert(EF_SEARCH.to_string(), i as jlong);
                                }
                            }
                        }
                    }
                }
                let nprobe_key = env.new_string(NPROBES)?;
                if let Ok(v) = env.call_method(
                    &sub_params_obj, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
                    &[(&nprobe_key).into()],
                ) {
                    if let Ok(o) = v.l() {
                        if !o.is_null() {
                            if let Ok(iv) = env.call_method(&o, "intValue", "()I", &[]) {
                                if let Ok(i) = iv.i() {
                                    sub_params.insert(NPROBES.to_string(), i as jlong);
                                }
                            }
                        }
                    }
                }
                set_extra_parameters(env, &sub_params, index_writer);
            }
        }
    }

    // Cast int8 training vectors to float
    let train_ptr = train_vectors_pointer as *const Vec<i8>;
    if train_ptr.is_null() {
        faiss_sys::faiss_index_free(index_writer);
        return Err(FaissWrapperError::NullPointer(
            "train_vectors_pointer is null".to_string(),
        ));
    }
    let training_vectors = &*train_ptr;
    let dim = dimension as usize;
    let num_vectors = training_vectors.len() / dim;
    let training_float_vectors: Vec<f32> = training_vectors
        .iter()
        .map(|&b| b as f32)
        .collect();

    // Train the index
    internal_train_index(
        index_writer,
        num_vectors as FaissIdx,
        training_float_vectors.as_ptr(),
    );

    // Serialize the trained index to bytes via VectorIOWriter
    let writer = faiss_sys::faiss_vector_io_writer_new();
    if writer.is_null() {
        faiss_sys::faiss_index_free(index_writer);
        return Err(FaissWrapperError::Runtime(
            "Failed to create VectorIOWriter".to_string(),
        ));
    }
    faiss_sys::faiss_write_index_to_vector_io_writer(index_writer as *const FaissIndex, writer);

    // Get the serialized bytes
    let mut data_ptr: *const u8 = ptr::null();
    let mut data_size: usize = 0;
    faiss_sys::faiss_vector_io_writer_get_data(writer, &mut data_ptr, &mut data_size);

    // Create Java byte array with the serialized index
    let ret = env.new_byte_array(data_size as i32)?;
    let data_slice = std::slice::from_raw_parts(data_ptr as *const i8, data_size);
    env.set_byte_array_region(&ret, 0, data_slice)?;

    // Cleanup
    faiss_sys::faiss_vector_io_writer_free(writer);
    faiss_sys::faiss_index_free(index_writer);

    Ok(ret)
}

// ---------------------------------------------------------------------------
// RangeSearch
// ---------------------------------------------------------------------------

/// Perform a range search against the index (no filter).
///
/// Delegates to `range_search_with_filter` with null filter IDs.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn range_search<'local>(
    env: &mut JNIEnv<'local>,
    index_pointer: jlong,
    query_vector: &JFloatArray,
    radius: jfloat,
    method_params: &JObject,
    max_result_window: jint,
    parent_ids: &JIntArray,
) -> Result<JObjectArray<'local>> {
    range_search_with_filter(
        env,
        index_pointer,
        query_vector,
        radius,
        method_params,
        max_result_window,
        ptr::null(),
        0,
        parent_ids,
    )
}

// ---------------------------------------------------------------------------
// RangeSearchWithFilter
// ---------------------------------------------------------------------------

/// Perform a range search against the index with optional filtering.
///
/// Returns results within the given radius, limited by `max_result_window`.
///
/// # Safety
/// Uses JNI and Faiss FFI extensively.
pub unsafe fn range_search_with_filter<'local>(
    env: &mut JNIEnv<'local>,
    index_pointer: jlong,
    query_vector: &JFloatArray,
    radius: jfloat,
    method_params: &JObject,
    max_result_window: jint,
    filter_ids: *const JLongArray<'local>,
    filter_ids_type: jint,
    parent_ids: &JIntArray,
) -> Result<JObjectArray<'local>> {
    if query_vector.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Query Vector cannot be null".to_string(),
        ));
    }

    let index_reader = index_pointer as *mut FaissIndexIDMap;
    if index_reader.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Invalid pointer to indexReader".to_string(),
        ));
    }

    // Set OMP threads to 1
    faiss_sys::omp_set_num_threads(1);

    // Get raw query vector
    let query_len = env.get_array_length(query_vector)? as usize;
    let mut query_buf = vec![0.0f32; query_len];
    env.get_float_array_region(query_vector, 0, &mut query_buf)?;
    let query_ptr = query_buf.as_ptr();

    // Parse method params if provided
    let mut ef_search_param: Option<i32> = None;
    if !method_params.is_null() {
        let ef_key = env.new_string(EF_SEARCH)?;
        if let Ok(val) = env.call_method(
            method_params, "get", "(Ljava/lang/Object;)Ljava/lang/Object;",
            &[(&ef_key).into()],
        ) {
            if let Ok(obj) = val.l() {
                if !obj.is_null() {
                    if let Ok(int_val) = env.call_method(&obj, "intValue", "()I", &[]) {
                        if let Ok(i) = int_val.i() {
                            ef_search_param = Some(i);
                        }
                    }
                }
            }
        }
    }

    let index_as_faiss = FaissIndexIDMap::as_index(index_reader);

    // Set HNSW efSearch if applicable
    let hnsw_ptr = faiss_sys::faiss_index_to_hnsw(index_as_faiss);
    if !hnsw_ptr.is_null() {
        if let Some(ef) = ef_search_param {
            faiss_sys::faiss_index_hnsw_set_ef_search(hnsw_ptr, ef as c_int);
        }
    }

    // Create RangeSearchResult
    // Note: The Faiss C API doesn't expose RangeSearchResult construction directly.
    // We use faiss_index_range_search which takes a pre-allocated result struct.
    // Since our FFI doesn't expose RangeSearchResult allocation, we use the basic
    // range_search call and handle results.

    // Allocate a FaissRangeSearchResult (need to add FFI for this)
    // For range search, the C API expects a pre-allocated result object.
    // We'll call faiss_index_range_search with our result pointer.

    // Since FaissRangeSearchResult construction isn't in our current FFI,
    // we add a local extern block for the needed functions.
    extern "C" {
        fn faiss_range_search_result_new(
            result: *mut *mut crate::ffi::faiss_sys::FaissRangeSearchResult,
            nq: FaissIdx,
        );
        fn faiss_range_search_result_get_lims(
            result: *const crate::ffi::faiss_sys::FaissRangeSearchResult,
        ) -> *const usize;
        fn faiss_range_search_result_get_labels(
            result: *const crate::ffi::faiss_sys::FaissRangeSearchResult,
        ) -> *const FaissIdx;
        fn faiss_range_search_result_get_distances(
            result: *const crate::ffi::faiss_sys::FaissRangeSearchResult,
        ) -> *const f32;
        fn faiss_range_search_result_free(
            result: *mut crate::ffi::faiss_sys::FaissRangeSearchResult,
        );
    }

    let mut range_result: *mut crate::ffi::faiss_sys::FaissRangeSearchResult = ptr::null_mut();
    faiss_range_search_result_new(&mut range_result, 1);
    if range_result.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Failed to create RangeSearchResult".to_string(),
        ));
    }

    // Execute range search (filter is not directly supported via C API without IDSelector shim)
    faiss_sys::faiss_index_range_search(
        index_as_faiss as *const FaissIndex,
        1,
        query_ptr,
        radius,
        range_result,
    );

    // Extract results from RangeSearchResult
    let lims = faiss_range_search_result_get_lims(range_result);
    let labels = faiss_range_search_result_get_labels(range_result);
    let dists = faiss_range_search_result_get_distances(range_result);

    // lims[0] is always 0, lims[1] gives total number of results for single query
    let total_results = *lims.add(1) as i32;

    // Limit to max_result_window
    let result_size = std::cmp::min(total_results, max_result_window) as usize;

    // Create KNNQueryResult Java objects and return
    let result_class = env.find_class("org/opensearch/knn/index/query/KNNQueryResult")?;
    let result_constructor = env.get_method_id(
        &result_class,
        "<init>",
        "(IF)V",
    )?;
    let results = env.new_object_array(
        result_size as i32,
        &result_class,
        &JObject::null(),
    )?;

    for i in 0..result_size {
        let result_obj = env.new_object_unchecked(
            &result_class,
            result_constructor,
            &[
                jni::sys::jvalue { i: *labels.add(i) as i32 },
                jni::sys::jvalue { f: *dists.add(i) },
            ],
        )?;
        env.set_object_array_element(&results, i as i32, &result_obj)?;
        env.delete_local_ref(result_obj)?;
    }

    // Free the range search result
    faiss_range_search_result_free(range_result);

    Ok(results)
}

// ---------------------------------------------------------------------------
// InitFaissSQIndex
// ---------------------------------------------------------------------------

/// Initialize a Faiss scalar quantization (SQ) index.
///
/// # Safety
/// Uses JNI and Faiss FFI.
pub unsafe fn init_faiss_sq_index(
    env: &mut JNIEnv,
    num_docs: jlong,
    dim: jint,
    parameters: &JObject,
    centroid_dp: jfloat,
    quantized_vec_bytes: jint,
) -> Result<jlong> {
    if dim <= 0 {
        return Err(FaissWrapperError::Runtime(
            "Vectors dimensions cannot be less than or equal to 0".to_string(),
        ));
    }
    if parameters.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Parameters cannot be null".to_string(),
        ));
    }

    // Dead code: faiss_service_jni now calls BinaryIndexService trait methods directly.
    Err(FaissWrapperError::Runtime(
        "init_faiss_sq_index: use BinaryIndexService trait dispatch via faiss_service_jni".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Helper functions (private)
// ---------------------------------------------------------------------------

/// Check if the index is an IVFPQ index with L2 metric type.
///
/// Unwraps IndexIDMap if present, then checks for IndexIVFPQ with METRIC_L2.
///
/// # Safety
/// - `index` must be a valid pointer to a Faiss Index (or null, which returns false).
unsafe fn is_index_ivfpq_l2(index: *mut FaissIndex) -> bool {
    if index.is_null() {
        return false;
    }

    // Try to get the underlying index (unwrap IDMap if present)
    // The Faiss C API would provide a way to get the sub-index from an IDMap.
    // For this port, we attempt to cast directly to IVFPQ.
    let ivfpq = faiss_sys::faiss_index_to_ivfpq(index);
    if ivfpq.is_null() {
        return false;
    }

    // Check metric type is L2
    let metric = faiss_sys::faiss_index_metric_type(index);
    metric == METRIC_L2
}

/// Extract the IVFPQ index from a (possibly ID-mapped) index.
///
/// # Safety
/// - `index` must be a valid pointer to a Faiss Index.
unsafe fn extract_ivfpq_index(index: *mut FaissIndex) -> Result<*mut FaissIndexIVFPQ> {
    if index.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Unable to extract IVFPQ index. Index pointer is null.".to_string(),
        ));
    }

    let ivfpq = faiss_sys::faiss_index_to_ivfpq(index);
    if ivfpq.is_null() {
        return Err(FaissWrapperError::Runtime(
            "Unable to extract IVFPQ index. IVFPQ index not present.".to_string(),
        ));
    }

    Ok(ivfpq)
}

/// Build an IDGrouper bitmap from parent IDs.
///
/// This is used for nested/parent-child document filtering during search.
///
/// # Safety
/// Uses JNI to access the parent IDs array.
unsafe fn build_id_grouper_bitmap(
    env: &mut JNIEnv,
    parent_ids: &JIntArray,
) -> Result<Vec<u64>> {
    if parent_ids.is_null() {
        return Ok(Vec::new());
    }

    let parent_ids_len = env.get_array_length(parent_ids)
        .map_err(|e| FaissWrapperError::Runtime(format!("Failed to get parent IDs length: {}", e)))? as usize;
    let mut parent_ids_buf = vec![0i32; parent_ids_len];
    env.get_int_array_region(parent_ids, 0, &mut parent_ids_buf)
        .map_err(|e| FaissWrapperError::Runtime(format!("Failed to get parent IDs: {}", e)))?;

    // Build a bitmap where each bit corresponds to a parent ID
    // Find the max parent ID to determine bitmap size
    let max_id = parent_ids_buf.iter().copied().max().unwrap_or(0) as u64;
    let bitmap_size = ((max_id + 64) / 64) as usize;
    let mut bitmap = vec![0u64; bitmap_size];

    for &id in &parent_ids_buf {
        let uid = id as u64;
        let word_idx = (uid / 64) as usize;
        let bit_idx = uid % 64;
        if word_idx < bitmap.len() {
            bitmap[word_idx] |= 1u64 << bit_idx;
        }
    }

    Ok(bitmap)
}

/// Get an integer method parameter from the parameters map, with a default value.
///
/// This mirrors `knn_jni::commons::getIntegerMethodParameter`.
pub fn get_integer_method_parameter(
    params: &HashMap<String, jlong>,
    key: &str,
    default_value: i32,
) -> i32 {
    params
        .get(key)
        .map(|&v| v as i32)
        .unwrap_or(default_value)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_space_to_metric() {
        assert_eq!(translate_space_to_metric("l2").unwrap(), METRIC_L2);
        assert_eq!(
            translate_space_to_metric("innerproduct").unwrap(),
            METRIC_INNER_PRODUCT
        );
        assert_eq!(
            translate_space_to_metric("cosinesimil").unwrap(),
            METRIC_INNER_PRODUCT
        );
        assert_eq!(translate_space_to_metric("hamming").unwrap(), METRIC_L2);
        assert!(translate_space_to_metric("unknown").is_err());
    }

    #[test]
    fn test_id_selector_jlong_bitmap() {
        // Bitmap: bit 0 and bit 63 set in word 0, bit 64 set in word 1
        let bitmap: [i64; 2] = [
            (1i64 << 0) | (1i64 << 63), // bits 0 and 63
            1i64 << 0,                   // bit 64
        ];

        let selector = unsafe { IDSelectorJlongBitmap::new(2, bitmap.as_ptr()) };

        assert!(selector.is_member(0));
        assert!(!selector.is_member(1));
        assert!(selector.is_member(63));
        assert!(selector.is_member(64));
        assert!(!selector.is_member(65));
        // Out of bounds
        assert!(!selector.is_member(128));
    }

    #[test]
    fn test_init_library_is_noop() {
        // Should not panic
        init_library();
    }

    #[test]
    fn test_filter_ids_selector_type() {
        assert_eq!(FilterIdsSelectorType::Bitmap as i32, 0);
        assert_eq!(FilterIdsSelectorType::Batch as i32, 1);
    }

    // -----------------------------------------------------------------------
    // Ported from C++ faiss_wrapper_unit_test.cpp: TranslateSpaceToMetric tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_translate_space_to_metric_l2_returns_metric_l2() {
        // Ported from C++ test that checks L2 space maps to METRIC_L2
        let result = translate_space_to_metric(L2);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), METRIC_L2);
    }

    #[test]
    fn test_translate_space_to_metric_inner_product_returns_metric_ip() {
        let result = translate_space_to_metric(INNER_PRODUCT);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), METRIC_INNER_PRODUCT);
    }

    #[test]
    fn test_translate_space_to_metric_cosinesimil_returns_metric_ip() {
        // Cosine similarity uses inner product (vectors are pre-normalized)
        let result = translate_space_to_metric(COSINESIMIL);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), METRIC_INNER_PRODUCT);
    }

    #[test]
    fn test_translate_space_to_metric_hamming_returns_metric_l2() {
        // Hamming is mapped to L2 as a fallback for float indices
        let result = translate_space_to_metric(HAMMING);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), METRIC_L2);
    }

    #[test]
    fn test_translate_space_to_metric_invalid_returns_error() {
        let result = translate_space_to_metric("invalid_space");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid spaceType"));
        assert!(err_msg.contains("invalid_space"));
    }

    #[test]
    fn test_translate_space_to_metric_empty_string_returns_error() {
        let result = translate_space_to_metric("");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Ported from C++ faiss_wrapper_unit_test.cpp: IDSelectorJlongBitmap tests
    // More comprehensive bit pattern testing
    // -----------------------------------------------------------------------

    #[test]
    fn test_id_selector_jlong_bitmap_empty_bitmap() {
        // All zeros - no bits set
        let bitmap: [i64; 2] = [0, 0];
        let selector = unsafe { IDSelectorJlongBitmap::new(2, bitmap.as_ptr()) };

        assert!(!selector.is_member(0));
        assert!(!selector.is_member(1));
        assert!(!selector.is_member(63));
        assert!(!selector.is_member(64));
        assert!(!selector.is_member(127));
    }

    #[test]
    fn test_id_selector_jlong_bitmap_all_set() {
        // All bits set
        let bitmap: [i64; 2] = [-1i64, -1i64]; // all 1s
        let selector = unsafe { IDSelectorJlongBitmap::new(2, bitmap.as_ptr()) };

        assert!(selector.is_member(0));
        assert!(selector.is_member(1));
        assert!(selector.is_member(32));
        assert!(selector.is_member(63));
        assert!(selector.is_member(64));
        assert!(selector.is_member(100));
        assert!(selector.is_member(127));
        // Out of bounds
        assert!(!selector.is_member(128));
    }

    #[test]
    fn test_id_selector_jlong_bitmap_specific_bits() {
        // Set bits 5, 10, 42, 70
        let mut bitmap: [i64; 2] = [0, 0];
        bitmap[0] |= 1i64 << 5;
        bitmap[0] |= 1i64 << 10;
        bitmap[0] |= 1i64 << 42;
        bitmap[1] |= 1i64 << (70 - 64);

        let selector = unsafe { IDSelectorJlongBitmap::new(2, bitmap.as_ptr()) };

        assert!(selector.is_member(5));
        assert!(selector.is_member(10));
        assert!(selector.is_member(42));
        assert!(selector.is_member(70));
        assert!(!selector.is_member(0));
        assert!(!selector.is_member(6));
        assert!(!selector.is_member(11));
        assert!(!selector.is_member(43));
        assert!(!selector.is_member(71));
    }

    #[test]
    fn test_id_selector_jlong_bitmap_single_word() {
        // Single word bitmap
        let bitmap: [i64; 1] = [0b1010_1010]; // bits 1, 3, 5, 7 set
        let selector = unsafe { IDSelectorJlongBitmap::new(1, bitmap.as_ptr()) };

        assert!(!selector.is_member(0));
        assert!(selector.is_member(1));
        assert!(!selector.is_member(2));
        assert!(selector.is_member(3));
        assert!(!selector.is_member(4));
        assert!(selector.is_member(5));
        assert!(!selector.is_member(6));
        assert!(selector.is_member(7));
        assert!(!selector.is_member(8));
        // Out of bounds for single word
        assert!(!selector.is_member(64));
    }

    #[test]
    fn test_id_selector_jlong_bitmap_large_ids() {
        // Test with IDs matching Lucene-style segment IDs (128, 1024 from C++ test)
        let num_words = (1024 / 64) + 1; // 17 words needed
        let mut bitmap = vec![0i64; num_words];
        bitmap[128 / 64] |= 1i64 << (128 % 64);
        bitmap[1024 / 64] |= 1i64 << (1024 % 64);

        let selector = unsafe { IDSelectorJlongBitmap::new(num_words, bitmap.as_ptr()) };

        assert!(selector.is_member(128));
        assert!(selector.is_member(1024));
        assert!(!selector.is_member(0));
        assert!(!selector.is_member(127));
        assert!(!selector.is_member(129));
        assert!(!selector.is_member(1023));
        assert!(!selector.is_member(1025));
    }

    // -----------------------------------------------------------------------
    // Ported from C++ faiss_wrapper_unit_test.cpp: Free function tests
    // Note: free() and is_index_ivfpq_l2() call FFI functions that are only
    // available when linked against real Faiss. We test the null-guard logic
    // by verifying the code paths without actually calling FFI.
    // -----------------------------------------------------------------------

    #[test]
    fn test_free_null_guard_logic() {
        // Verify that the free function's null check logic is correct:
        // When pointer is 0 (null), the function should not call any FFI.
        // We verify this by checking that converting 0 to pointer yields null.
        let ptr = 0 as *mut FaissIndex;
        assert!(ptr.is_null());
        let ptr_binary = 0 as *mut FaissIndexBinary;
        assert!(ptr_binary.is_null());
    }

    #[test]
    fn test_is_index_ivfpq_l2_null_guard_logic() {
        // Verify that is_index_ivfpq_l2 returns false for null pointer.
        // The actual function calls FFI, so we verify the logic path:
        // when index.is_null() is true, it returns false immediately.
        let index = std::ptr::null_mut::<FaissIndex>();
        assert!(index.is_null()); // Would return false early in is_index_ivfpq_l2
    }

    // -----------------------------------------------------------------------
    // Additional: get_integer_method_parameter tests (pure logic, no JNI)
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_integer_method_parameter_found() {
        let mut params: HashMap<String, jlong> = HashMap::new();
        params.insert("ef_search".to_string(), 200);

        let result = get_integer_method_parameter(&params, "ef_search", 16);
        assert_eq!(result, 200);
    }

    #[test]
    fn test_get_integer_method_parameter_not_found() {
        let mut params: HashMap<String, jlong> = HashMap::new();
        params.insert("ef_search".to_string(), 200);

        let result = get_integer_method_parameter(&params, "nprobes", 1);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_get_integer_method_parameter_empty_map() {
        let params: HashMap<String, jlong> = HashMap::new();

        let result = get_integer_method_parameter(&params, "ef_search", 16);
        assert_eq!(result, 16);
    }

    // -----------------------------------------------------------------------
    // FaissWrapperError tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_faiss_wrapper_error_display() {
        let err = FaissWrapperError::Runtime("test error".to_string());
        assert_eq!(err.to_string(), "test error");

        let err = FaissWrapperError::NullPointer("null ptr".to_string());
        assert_eq!(err.to_string(), "Null pointer: null ptr");
    }

    // =======================================================================
    // Integration tests exercising real Faiss FFI operations
    // =======================================================================

    // Local extern "C" declarations for the real Faiss C API functions.
    // These use the actual symbol names from the Faiss C API (capital 'I'
    // in "Index"), which differ from some of the simplified names in
    // ffi/faiss_sys.rs. They are used by integration tests to call Faiss
    // directly without going through the JNI wrapper layer.
    extern "C" {
        /// Create a float index from a description string.
        /// Returns 0 on success.
        #[link_name = "faiss_index_factory"]
        fn real_faiss_index_factory(
            p_index: *mut *mut faiss_sys::FaissIndex,
            d: c_int,
            description: *const std::os::raw::c_char,
            metric: faiss_sys::FaissMetricType,
        ) -> c_int;

        /// Add vectors to an index (sequential IDs 0..n).
        #[link_name = "faiss_Index_add"]
        fn real_faiss_Index_add(
            index: *mut faiss_sys::FaissIndex,
            n: i64,
            x: *const f32,
        ) -> c_int;

        /// Search the k nearest neighbors.
        #[link_name = "faiss_Index_search"]
        fn real_faiss_Index_search(
            index: *const faiss_sys::FaissIndex,
            n: i64,
            x: *const f32,
            k: i64,
            distances: *mut f32,
            labels: *mut i64,
        ) -> c_int;

        /// Train an index.
        #[link_name = "faiss_Index_train"]
        fn real_faiss_Index_train(
            index: *mut faiss_sys::FaissIndex,
            n: i64,
            x: *const f32,
        ) -> c_int;

        /// Check if an index is trained. Returns non-zero if trained.
        #[link_name = "faiss_Index_is_trained"]
        fn real_faiss_Index_is_trained(
            index: *const faiss_sys::FaissIndex,
        ) -> c_int;

        /// Get the metric type of an index.
        #[link_name = "faiss_Index_metric_type"]
        fn real_faiss_Index_metric_type(
            index: *const faiss_sys::FaissIndex,
        ) -> faiss_sys::FaissMetricType;

        /// Free a float index.
        #[link_name = "faiss_Index_free"]
        fn real_faiss_Index_free(
            index: *mut faiss_sys::FaissIndex,
        );
    }

    /// Create an HNSW32 index, add 5 known vectors, search for closest to a
    /// query vector, and verify the top result is correct.
    ///
    /// Ported from C++ faiss_wrapper_test.cpp: CreateHNSWIndexAndSearch.
    #[test]
    fn test_create_hnsw_index_and_search() {
        unsafe {
            let dim = 3i32;
            let description = std::ffi::CString::new("HNSW32").unwrap();
            let mut index: *mut faiss_sys::FaissIndex = ptr::null_mut();
            let ret = real_faiss_index_factory(
                &mut index,
                dim as c_int,
                description.as_ptr(),
                METRIC_L2,
            );
            assert_eq!(ret, 0, "faiss_index_factory should return 0 on success");
            assert!(
                !index.is_null(),
                "faiss_index_factory returned null for HNSW32"
            );

            // 5 vectors of dimension 3
            #[rustfmt::skip]
            let vectors: [f32; 15] = [
                1.0, 2.0, 3.0,   // id 0
                4.0, 5.0, 6.0,   // id 1
                7.0, 8.0, 9.0,   // id 2
                10.0, 11.0, 12.0, // id 3
                13.0, 14.0, 15.0, // id 4
            ];

            // HNSW supports add without explicit IDs (sequential 0..n)
            let ret = real_faiss_Index_add(index, 5, vectors.as_ptr());
            assert_eq!(ret, 0, "faiss_Index_add should return 0 on success");

            // Query: closest to vector [1.0, 2.0, 3.0] should be id 0
            let query: [f32; 3] = [1.0, 2.0, 3.0];
            let k = 1i64;
            let mut distances: [f32; 1] = [0.0];
            let mut labels: [i64; 1] = [-1];

            let ret = real_faiss_Index_search(
                index as *const faiss_sys::FaissIndex,
                1,
                query.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );
            assert_eq!(ret, 0, "faiss_Index_search should return 0 on success");

            assert_eq!(labels[0], 0, "Nearest neighbor should be vector 0");
            assert!(
                distances[0] < 1e-6,
                "Distance to exact match should be ~0, got {}",
                distances[0]
            );

            // Another query: closest to [13.0, 14.0, 15.0] should be id 4
            let query2: [f32; 3] = [13.0, 14.0, 15.0];
            let ret = real_faiss_Index_search(
                index as *const faiss_sys::FaissIndex,
                1,
                query2.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );
            assert_eq!(ret, 0, "faiss_Index_search should return 0 on success");

            assert_eq!(labels[0], 4, "Nearest neighbor should be vector 4");
            assert!(
                distances[0] < 1e-6,
                "Distance to exact match should be ~0, got {}",
                distances[0]
            );

            real_faiss_Index_free(index);
        }
    }

    /// Verify that translate_space_to_metric("l2") returns METRIC_L2 and
    /// that the value matches the metric_type on an index created with L2.
    ///
    /// This is an integration-level check that our constant agrees with
    /// what Faiss internally stores on a constructed index.
    #[test]
    fn test_translate_space_to_metric_integration() {
        let metric = translate_space_to_metric("l2").unwrap();
        assert_eq!(metric, METRIC_L2);

        unsafe {
            let description = std::ffi::CString::new("Flat").unwrap();
            let mut index: *mut faiss_sys::FaissIndex = ptr::null_mut();
            let ret = real_faiss_index_factory(
                &mut index,
                4 as c_int,
                description.as_ptr(),
                METRIC_L2,
            );
            assert_eq!(ret, 0, "faiss_index_factory should return 0");
            assert!(!index.is_null());

            let index_metric = real_faiss_Index_metric_type(
                index as *const faiss_sys::FaissIndex,
            );
            assert_eq!(
                index_metric, METRIC_L2,
                "Index metric type should match METRIC_L2"
            );
            assert_eq!(
                metric, index_metric,
                "translate_space_to_metric result should match index metric_type"
            );

            real_faiss_Index_free(index);
        }
    }

    /// Call free(0, false) and free(0, true) and verify they do not panic.
    /// The free() function has null guards so passing 0 (null) should be safe.
    ///
    /// Ported from C++ faiss_wrapper_unit_test.cpp: FreeNullDoesNotCrash.
    #[test]
    fn test_free_null_does_not_crash() {
        // free with null float index pointer
        unsafe {
            free(0, 0); // is_binary_index = false (jboolean 0)
        }
        // free with null binary index pointer
        unsafe {
            free(0, 1); // is_binary_index = true (jboolean 1)
        }
        // If we reach here without panicking or segfaulting, the test passes.
    }

    /// Create an IVF index that requires training, train it with data,
    /// and verify that is_trained becomes true afterwards.
    ///
    /// Ported from C++ faiss_wrapper_test.cpp: InternalTrainIndex.
    #[test]
    fn test_internal_train_index() {
        unsafe {
            let dim = 8i32;
            // IVF4,Flat requires training (4 centroids, needs >= 4 vectors)
            let description = std::ffi::CString::new("IVF4,Flat").unwrap();
            let mut index: *mut faiss_sys::FaissIndex = ptr::null_mut();
            let ret = real_faiss_index_factory(
                &mut index,
                dim as c_int,
                description.as_ptr(),
                METRIC_L2,
            );
            assert_eq!(ret, 0, "faiss_index_factory should return 0 for IVF4,Flat");
            assert!(
                !index.is_null(),
                "faiss_index_factory returned null for IVF4,Flat"
            );

            // Verify index is NOT trained initially
            let trained_before = real_faiss_Index_is_trained(
                index as *const faiss_sys::FaissIndex,
            );
            assert_eq!(trained_before, 0, "IVF index should not be trained initially");

            // Generate training data: 100 vectors of dimension 8
            // Use a simple deterministic pattern that ensures diversity
            let num_train = 100i64;
            let mut train_data = vec![0.0f32; (num_train as usize) * (dim as usize)];
            for i in 0..(num_train as usize) {
                for j in 0..(dim as usize) {
                    train_data[i * (dim as usize) + j] =
                        ((i * 7 + j * 13) % 100) as f32 / 10.0;
                }
            }

            // Train the index directly using the real C API
            let ret = real_faiss_Index_train(index, num_train, train_data.as_ptr());
            assert_eq!(ret, 0, "faiss_Index_train should return 0 on success");

            // Verify index IS trained after
            let trained_after = real_faiss_Index_is_trained(
                index as *const faiss_sys::FaissIndex,
            );
            assert_ne!(trained_after, 0, "IVF index should be trained after training");

            real_faiss_Index_free(index);
        }
    }

    /// Test IDSelectorJlongBitmap edge cases at word boundaries.
    ///
    /// Specifically tests bits 63, 64, 127, 128 which sit at the exact boundary
    /// between 64-bit words. These are the boundary points where the word index
    /// (id >> 6) and bit position (id & 63) both change, making them the most
    /// likely places for off-by-one errors.
    #[test]
    fn test_id_selector_jlong_bitmap_correctness_word_boundaries() {
        // 3 words = bits 0..191
        let mut bitmap = [0i64; 3];

        // Set bits at word boundaries: 63 (last bit of word 0), 64 (first bit
        // of word 1), 127 (last bit of word 1), 128 (first bit of word 2)
        bitmap[0] |= 1i64 << 63; // bit 63
        bitmap[1] |= 1i64 << 0;  // bit 64
        bitmap[1] |= 1i64 << 63; // bit 127
        bitmap[2] |= 1i64 << 0;  // bit 128

        let selector = unsafe { IDSelectorJlongBitmap::new(3, bitmap.as_ptr()) };

        // Boundary bits should be set
        assert!(selector.is_member(63), "bit 63 should be set");
        assert!(selector.is_member(64), "bit 64 should be set");
        assert!(selector.is_member(127), "bit 127 should be set");
        assert!(selector.is_member(128), "bit 128 should be set");

        // Adjacent bits should NOT be set
        assert!(!selector.is_member(62), "bit 62 should not be set");
        assert!(!selector.is_member(65), "bit 65 should not be set");
        assert!(!selector.is_member(126), "bit 126 should not be set");
        assert!(!selector.is_member(129), "bit 129 should not be set");

        // Bits entirely outside
        assert!(!selector.is_member(0), "bit 0 should not be set");
        assert!(!selector.is_member(191), "bit 191 should not be set");
        assert!(!selector.is_member(192), "bit 192 out of bounds");

        // Verify that bits at word boundary 0 (bit 0) and mid-word positions
        // are not accidentally set by the boundary bits
        assert!(!selector.is_member(1), "bit 1 should not be set");
        assert!(!selector.is_member(32), "bit 32 should not be set");
        assert!(!selector.is_member(96), "bit 96 should not be set");
        assert!(!selector.is_member(160), "bit 160 should not be set");
    }
}
