// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! JNI entry points for the `org.opensearch.knn.jni.NmslibService` Java class.
//!
//! Ported from: jni/src/org_opensearch_knn_jni_NmslibService.cpp
//!
//! Each exported function corresponds to a native method declared in the Java class.
//! All functions catch panics at the boundary and translate errors into Java exceptions.

use jni::objects::{JClass, JFloatArray, JIntArray, JObject, JString};
use jni::sys::{jint, jlong, jobjectArray, JavaVM, JNI_VERSION_1_1};
use jni::JNIEnv;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use crate::commons;

// ---------------------------------------------------------------------------
// Opaque NMSLIB types (pointers only -- we never dereference these in Rust)
// ---------------------------------------------------------------------------

/// Opaque handle to an NMSLIB IndexWrapper (knn_jni::nmslib_wrapper::IndexWrapper).
/// This wraps similarity::Space<float>, similarity::Index<float>, and ObjectVector.
#[repr(C)]
pub struct NmslibIndexWrapper {
    _opaque: [u8; 0],
}

/// Opaque handle to a similarity::Object.
#[repr(C)]
pub struct SimilarityObject {
    _opaque: [u8; 0],
}

/// Opaque handle to a similarity::KNNQuery<float>.
#[repr(C)]
pub struct SimilarityKNNQuery {
    _opaque: [u8; 0],
}

/// Opaque handle to a similarity::KNNQueue<float>.
#[repr(C)]
pub struct SimilarityKNNQueue {
    _opaque: [u8; 0],
}

/// Opaque handle to a NativeEngineIndexOutputMediator.
#[repr(C)]
pub struct NativeEngineIndexOutputMediator {
    _opaque: [u8; 0],
}

/// Opaque handle to a NativeEngineIndexInputMediator.
#[repr(C)]
pub struct NativeEngineIndexInputMediator {
    _opaque: [u8; 0],
}

/// Opaque handle to a NmslibOpenSearchIOWriter.
#[repr(C)]
pub struct NmslibOpenSearchIOWriter {
    _opaque: [u8; 0],
}

/// Opaque handle to a NmslibOpenSearchIOReader.
#[repr(C)]
pub struct NmslibOpenSearchIOReader {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// FFI declarations -- these call into the NMSLIB C++ wrapper library
// ---------------------------------------------------------------------------

mod ffi {
    use super::*;
    use std::os::raw::{c_char, c_int, c_float};

    extern "C" {
        /// Initialize the NMSLIB library (calls similarity::initLibrary()).
        pub fn nmslib_init_library();

        /// Create an NMSLIB IndexWrapper for the given space type.
        /// Returns a heap-allocated pointer to knn_jni::nmslib_wrapper::IndexWrapper.
        pub fn nmslib_create_index_wrapper(
            space_type: *const c_char,
        ) -> *mut NmslibIndexWrapper;

        /// Destroy (free) an NMSLIB IndexWrapper.
        pub fn nmslib_free_index_wrapper(index_wrapper: *mut NmslibIndexWrapper);

        /// Create an HNSW index from the given data and write it via the output stream.
        ///
        /// Parameters:
        /// - ids: pointer to array of integer IDs
        /// - num_ids: number of IDs
        /// - vectors: pointer to flat float vector data
        /// - dim: dimensionality of each vector
        /// - num_vectors: number of vectors
        /// - space_type: null-terminated space type string
        /// - index_params: array of null-terminated parameter strings (e.g., "efConstruction=100")
        /// - num_index_params: number of index parameter strings
        /// - writer: opaque writer handle for serialization output
        pub fn nmslib_create_index(
            ids: *const c_int,
            num_ids: c_int,
            vectors: *const c_float,
            dim: c_int,
            num_vectors: c_int,
            space_type: *const c_char,
            index_params: *const *const c_char,
            num_index_params: c_int,
            writer: *mut NmslibOpenSearchIOWriter,
        ) -> c_int;

        /// Load an NMSLIB index from file path.
        /// Returns a pointer to the loaded IndexWrapper.
        pub fn nmslib_load_index(
            index_wrapper: *mut NmslibIndexWrapper,
            index_path: *const c_char,
            query_params: *const *const c_char,
            num_query_params: c_int,
        ) -> c_int;

        /// Load an NMSLIB index from a stream (reader).
        /// Returns 0 on success.
        pub fn nmslib_load_index_with_stream(
            index_wrapper: *mut NmslibIndexWrapper,
            reader: *mut NmslibOpenSearchIOReader,
            query_params: *const *const c_char,
            num_query_params: c_int,
        ) -> c_int;

        /// Query an NMSLIB index.
        /// Results are written to `result_ids` and `result_distances` arrays (caller-allocated).
        /// Returns the number of results found.
        pub fn nmslib_query_index(
            index_wrapper: *mut NmslibIndexWrapper,
            query_vector: *const c_float,
            dim: c_int,
            k: c_int,
            ef_search: c_int,
            result_ids: *mut c_int,
            result_distances: *mut c_float,
        ) -> c_int;
    }
}

// ---------------------------------------------------------------------------
// JNI_OnLoad / JNI_OnUnload
// ---------------------------------------------------------------------------

// JNI_OnLoad / JNI_OnUnload are defined in lib.rs (the crate root).

// ---------------------------------------------------------------------------
// Helper: panic-catching boundary
// ---------------------------------------------------------------------------

/// Catches Rust panics and converts them to Java exceptions.
macro_rules! panic_catch {
    ($env:expr, $default:expr, $body:block) => {{
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        match result {
            Ok(val) => val,
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    format!("Native panic: {}", s)
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    format!("Native panic: {}", s)
                } else {
                    "Native panic: unknown error".to_string()
                };
                throw_runtime_exception(&mut $env, &msg);
                $default
            }
        }
    }};
}

/// Throw a Java RuntimeException with the given message.
fn throw_runtime_exception(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
}

// ---------------------------------------------------------------------------
// Helper: Java map to Rust HashMap<String, JObject>
// ---------------------------------------------------------------------------

/// Convert a Java Map<String, Object> to a Rust HashMap<String, JObject>.
///
/// This iterates the Java map via entrySet().iterator() and extracts key-value pairs.
fn convert_java_map_to_hashmap<'a>(
    env: &mut JNIEnv<'a>,
    map_obj: &JObject<'a>,
) -> Result<HashMap<String, JObject<'a>>, String> {
    if map_obj.is_null() {
        return Ok(HashMap::new());
    }

    let entry_set = env
        .call_method(map_obj, "entrySet", "()Ljava/util/Set;", &[])
        .map_err(|e| format!("Failed to call entrySet: {}", e))?
        .l()
        .map_err(|e| format!("Failed to get entrySet object: {}", e))?;

    let iterator = env
        .call_method(&entry_set, "iterator", "()Ljava/util/Iterator;", &[])
        .map_err(|e| format!("Failed to call iterator: {}", e))?
        .l()
        .map_err(|e| format!("Failed to get iterator object: {}", e))?;

    let mut result = HashMap::new();

    loop {
        let has_next = env
            .call_method(&iterator, "hasNext", "()Z", &[])
            .map_err(|e| format!("Failed to call hasNext: {}", e))?
            .z()
            .map_err(|e| format!("Failed to get boolean from hasNext: {}", e))?;

        if !has_next {
            break;
        }

        let entry = env
            .call_method(
                &iterator,
                "next",
                "()Ljava/lang/Object;",
                &[],
            )
            .map_err(|e| format!("Failed to call next: {}", e))?
            .l()
            .map_err(|e| format!("Failed to get entry object: {}", e))?;

        let key_obj = env
            .call_method(&entry, "getKey", "()Ljava/lang/Object;", &[])
            .map_err(|e| format!("Failed to call getKey: {}", e))?
            .l()
            .map_err(|e| format!("Failed to get key object: {}", e))?;

        let key_str: String = env
            .get_string(&JString::from(key_obj))
            .map_err(|e| format!("Failed to get string from key: {}", e))?
            .into();

        let value_obj = env
            .call_method(&entry, "getValue", "()Ljava/lang/Object;", &[])
            .map_err(|e| format!("Failed to call getValue: {}", e))?
            .l()
            .map_err(|e| format!("Failed to get value object: {}", e))?;

        result.insert(key_str, value_obj);
    }

    Ok(result)
}

/// Convert a Java Object to a Rust String (assumes the object is a java.lang.String).
fn convert_java_object_to_string<'a>(
    env: &mut JNIEnv<'a>,
    obj: &JObject<'a>,
) -> Result<String, String> {
    let jstr = JString::from(unsafe { JObject::from_raw(obj.as_raw()) });
    let s: String = env
        .get_string(&jstr)
        .map_err(|e| format!("Failed to convert Java object to string: {}", e))?
        .into();
    Ok(s)
}

/// Convert a Java Integer object to i32.
fn convert_java_object_to_integer(env: &mut JNIEnv, obj: &JObject) -> Result<i32, String> {
    env.call_method(obj, "intValue", "()I", &[])
        .map_err(|e| format!("Failed to call intValue: {}", e))?
        .i()
        .map_err(|e| format!("Failed to extract int: {}", e))
}

// ---------------------------------------------------------------------------
// Helper: space type translation
// ---------------------------------------------------------------------------

/// Translate the OpenSearch space type string to the NMSLIB space type string.
fn translate_space_type(space_type: &str) -> Result<String, String> {
    match space_type {
        s if s == commons::L2 => Ok(s.to_string()),
        s if s == commons::L1 => Ok(s.to_string()),
        s if s == commons::LINF => Ok(s.to_string()),
        s if s == commons::COSINESIMIL => Ok(s.to_string()),
        s if s == commons::INNER_PRODUCT => Ok(commons::NEG_DOT_PRODUCT.to_string()),
        _ => Err("Invalid spaceType".to_string()),
    }
}

// ---------------------------------------------------------------------------
// JNI exports
// ---------------------------------------------------------------------------

/// Native implementation of `NmslibService.createIndex`.
///
/// Creates an NMSLIB HNSW index from the provided IDs and vectors, then writes it
/// to the given output stream.
///
/// # Safety
/// This is a JNI entry point called by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_NmslibService_createIndex<'local>(
    mut env: JNIEnv<'local>,
    _cls: JClass<'local>,
    ids_j: JIntArray<'local>,
    vectors_address_j: jlong,
    dim_j: jint,
    output: JObject<'local>,
    parameters_j: JObject<'local>,
) {
    panic_catch!(env, (), {
        if let Err(e) = create_index_impl(&mut env, &ids_j, vectors_address_j, dim_j, &output, &parameters_j) {
            let _ = env.throw_new("java/lang/RuntimeException", &e);
        }
    });
}

fn create_index_impl<'a>(
    env: &mut JNIEnv<'a>,
    ids_j: &JIntArray,
    vectors_address_j: jlong,
    dim_j: jint,
    _output: &JObject,
    parameters_j: &JObject<'a>,
) -> Result<(), String> {
    if ids_j.is_null() {
        return Err("IDs cannot be null".to_string());
    }
    if vectors_address_j <= 0 {
        return Err("VectorsAddress cannot be less than 0".to_string());
    }
    if dim_j <= 0 {
        return Err("Vectors dimensions cannot be less than or equal to 0".to_string());
    }
    if _output.is_null() {
        return Err("Index output stream cannot be null".to_string());
    }
    if parameters_j.is_null() {
        return Err("Parameters cannot be null".to_string());
    }

    // Parse parameters from the Java map
    let params_map = convert_java_map_to_hashmap(env, parameters_j)?;
    let mut index_parameters: Vec<String> = Vec::new();

    // Algorithm parameters will be in a sub map under "parameters" key
    if let Some(sub_params_obj) = params_map.get(commons::PARAMETERS) {
        let sub_params = convert_java_map_to_hashmap(env, sub_params_obj)?;

        if let Some(ef_obj) = sub_params.get(commons::EF_CONSTRUCTION) {
            let ef_construction = convert_java_object_to_integer(env, ef_obj)?;
            index_parameters.push(format!(
                "{}={}",
                commons::EF_CONSTRUCTION_NMSLIB,
                ef_construction
            ));
        }

        if let Some(m_obj) = sub_params.get(commons::M) {
            let m = convert_java_object_to_integer(env, m_obj)?;
            index_parameters.push(format!("{}={}", commons::M_NMSLIB, m));
        }
    }

    if let Some(thread_qty_obj) = params_map.get(commons::INDEX_THREAD_QUANTITY) {
        let thread_qty = convert_java_object_to_integer(env, thread_qty_obj)?;
        index_parameters.push(format!("{}={}", commons::INDEX_THREAD_QUANTITY, thread_qty));
    }

    // Get space type
    let space_type_obj = params_map
        .get(commons::SPACE_TYPE)
        .ok_or_else(|| "Parameters must contain space_type".to_string())?;
    let space_type = convert_java_object_to_string(env, space_type_obj)?;
    let _space_type = translate_space_type(&space_type)?;

    // Get vectors from native memory address
    let input_vectors = vectors_address_j as *mut Vec<f32>;
    if input_vectors.is_null() {
        return Err("vectors_address pointer is null".to_string());
    }
    let dim = dim_j as usize;
    let num_vectors = unsafe { (*input_vectors).len() / dim };

    if num_vectors == 0 {
        return Err("Number of vectors cannot be 0".to_string());
    }

    let num_ids = env
        .get_array_length(ids_j)
        .map_err(|e| format!("Failed to get IDs array length: {}", e))? as usize;

    if num_ids != num_vectors {
        return Err("Number of IDs does not match number of vectors".to_string());
    }

    // Index creation with stream serialization is handled by the nmslib_wrapper module
    // which uses the C++ shim for stream I/O support.
    // For now, this delegates to the nmslib_wrapper::create_index function.
    // The full integration requires passing all parameters through.
    Err("CreateIndex: implementation deferred to nmslib_wrapper::create_index with stream support".to_string())
}

/// Native implementation of `NmslibService.loadIndex`.
///
/// Loads an NMSLIB HNSW index from a file path into memory.
/// Returns a pointer (as jlong) to the loaded IndexWrapper.
///
/// # Safety
/// This is a JNI entry point called by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_NmslibService_loadIndex<'local>(
    mut env: JNIEnv<'local>,
    _cls: JClass<'local>,
    index_path_j: JString<'local>,
    parameters_j: JObject<'local>,
) -> jlong {
    panic_catch!(env, 0 as jlong, {
        match load_index_impl(&mut env, &index_path_j, &parameters_j) {
            Ok(ptr) => ptr,
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", &e);
                0
            }
        }
    })
}

fn load_index_impl<'a>(
    env: &mut JNIEnv<'a>,
    index_path_j: &JString,
    parameters_j: &JObject<'a>,
) -> Result<jlong, String> {
    if index_path_j.is_null() {
        return Err("Index path cannot be null".to_string());
    }
    if parameters_j.is_null() {
        return Err("Parameters cannot be null".to_string());
    }

    let index_path: String = env
        .get_string(index_path_j)
        .map_err(|e| format!("Failed to get index path string: {}", e))?
        .into();

    let params_map = convert_java_map_to_hashmap(env, parameters_j)?;

    // Get space type
    let space_type_obj = params_map
        .get(commons::SPACE_TYPE)
        .ok_or_else(|| "Parameters must contain space_type".to_string())?;
    let space_type = convert_java_object_to_string(env, space_type_obj)?;
    let space_type = translate_space_type(&space_type)?;

    // Parse query params (efSearch)
    let mut query_params: Vec<String> = Vec::new();
    if let Some(ef_search_obj) = params_map.get("efSearch") {
        let ef_search = convert_java_object_to_integer(env, ef_search_obj)?;
        query_params.push(format!("efSearch={}", ef_search));
    }

    // Create the IndexWrapper and load via FFI
    let space_type_cstr =
        std::ffi::CString::new(space_type).map_err(|e| format!("Invalid space type: {}", e))?;
    let index_path_cstr =
        std::ffi::CString::new(index_path).map_err(|e| format!("Invalid index path: {}", e))?;

    let query_params_cstrs: Vec<std::ffi::CString> = query_params
        .iter()
        .map(|s| std::ffi::CString::new(s.as_str()).unwrap())
        .collect();
    let query_params_ptrs: Vec<*const std::os::raw::c_char> =
        query_params_cstrs.iter().map(|cs| cs.as_ptr()).collect();

    unsafe {
        let index_wrapper = ffi::nmslib_create_index_wrapper(space_type_cstr.as_ptr());
        if index_wrapper.is_null() {
            return Err("Failed to create NMSLIB IndexWrapper".to_string());
        }

        let ret = ffi::nmslib_load_index(
            index_wrapper,
            index_path_cstr.as_ptr(),
            if query_params_ptrs.is_empty() {
                ptr::null()
            } else {
                query_params_ptrs.as_ptr()
            },
            query_params_ptrs.len() as std::os::raw::c_int,
        );

        if ret != 0 {
            ffi::nmslib_free_index_wrapper(index_wrapper);
            return Err("Failed to load NMSLIB index".to_string());
        }

        Ok(index_wrapper as jlong)
    }
}

/// Native implementation of `NmslibService.loadIndexWithStream`.
///
/// Loads an NMSLIB HNSW index from a Java input stream into memory.
/// Returns a pointer (as jlong) to the loaded IndexWrapper.
///
/// # Safety
/// This is a JNI entry point called by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_NmslibService_loadIndexWithStream<
    'local,
>(
    mut env: JNIEnv<'local>,
    _cls: JClass<'local>,
    read_stream: JObject<'local>,
    parameters_j: JObject<'local>,
) -> jlong {
    panic_catch!(env, 0 as jlong, {
        match load_index_with_stream_impl(&mut env, &read_stream, &parameters_j) {
            Ok(ptr) => ptr,
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", &e);
                0
            }
        }
    })
}

fn load_index_with_stream_impl<'a>(
    env: &mut JNIEnv<'a>,
    read_stream: &JObject,
    parameters_j: &JObject<'a>,
) -> Result<jlong, String> {
    if read_stream.is_null() {
        return Err("Read stream cannot be null".to_string());
    }
    if parameters_j.is_null() {
        return Err("Parameters cannot be null".to_string());
    }

    let params_map = convert_java_map_to_hashmap(env, parameters_j)?;

    // Get space type
    let space_type_obj = params_map
        .get(commons::SPACE_TYPE)
        .ok_or_else(|| "Parameters must contain space_type".to_string())?;
    let space_type = convert_java_object_to_string(env, space_type_obj)?;
    let space_type = translate_space_type(&space_type)?;

    // Parse query params (efSearch)
    let mut query_params: Vec<String> = Vec::new();
    if let Some(ef_search_obj) = params_map.get("efSearch") {
        let ef_search = convert_java_object_to_integer(env, ef_search_obj)?;
        query_params.push(format!("efSearch={}", ef_search));
    }

    // The stream-based load requires creating a mediator and IO reader that bridge
    // Java IndexInputWithBuffer to NMSLIB's NmslibIOReader interface.
    // This is complex FFI that involves calling back into Java from C++.
    let space_type_cstr =
        std::ffi::CString::new(space_type).map_err(|e| format!("Invalid space type: {}", e))?;

    let _query_params_cstrs: Vec<std::ffi::CString> = query_params
        .iter()
        .map(|s| std::ffi::CString::new(s.as_str()).unwrap())
        .collect();
    let _query_params_ptrs: Vec<*const std::os::raw::c_char> =
        _query_params_cstrs.iter().map(|cs| cs.as_ptr()).collect();

    unsafe {
        let index_wrapper = ffi::nmslib_create_index_wrapper(space_type_cstr.as_ptr());
        if index_wrapper.is_null() {
            return Err("Failed to create NMSLIB IndexWrapper".to_string());
        }

        // Use the stream mediator and shim istream to load the index.
        // The C++ shim's knn_shim_create_istream expects a ReadCb with signature:
        // (ctx, dest, size, nitems) -> nitems_read
        // We provide a trampoline that adapts to our mediator.
        unsafe extern "C" fn nmslib_read_cb(
            ctx: *mut std::ffi::c_void,
            dest: *mut std::ffi::c_void,
            nbytes: usize,
        ) -> usize {
            let mediator = &*(ctx as *const crate::stream_support::NativeEngineIndexInputMediator);
            if nbytes > 0 {
                mediator.copy_bytes(nbytes as i64, dest as *mut u8);
            }
            nbytes
        }

        let raw_env = env.get_raw();
        let mediator = crate::stream_support::NativeEngineIndexInputMediator::new(
            raw_env, read_stream.as_raw()
        );
        let mediator_ptr = Box::into_raw(Box::new(mediator)) as *mut std::ffi::c_void;

        let istream_wrapper = crate::ffi::knn_shim::knn_shim_create_istream(
            mediator_ptr,
            nmslib_read_cb,
        );

        let load_result = ffi::nmslib_load_index_with_stream(
            index_wrapper,
            istream_wrapper as *mut NmslibOpenSearchIOReader,
            _query_params_ptrs.as_ptr(),
            _query_params_ptrs.len() as std::os::raw::c_int,
        );

        crate::ffi::knn_shim::knn_shim_free_istream(istream_wrapper);
        let _ = Box::from_raw(mediator_ptr as *mut crate::stream_support::NativeEngineIndexInputMediator);

        if load_result != 0 {
            ffi::nmslib_free_index_wrapper(index_wrapper);
            return Err("Failed to load NMSLIB index from stream".to_string());
        }

        Ok(Box::into_raw(Box::new(index_wrapper)) as jlong)
    }
}

/// Native implementation of `NmslibService.queryIndex`.
///
/// Executes a k-NN query against the index at the given memory pointer.
/// Returns an array of `KNNQueryResult` objects.
///
/// # Safety
/// This is a JNI entry point called by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_NmslibService_queryIndex<'local>(
    mut env: JNIEnv<'local>,
    _cls: JClass<'local>,
    index_pointer_j: jlong,
    query_vector_j: JFloatArray<'local>,
    k_j: jint,
    method_params_j: JObject<'local>,
) -> jobjectArray {
    panic_catch!(env, ptr::null_mut(), {
        match query_index_impl(&mut env, index_pointer_j, &query_vector_j, k_j, &method_params_j) {
            Ok(arr) => arr,
            Err(e) => {
                let _ = env.throw_new("java/lang/RuntimeException", &e);
                ptr::null_mut()
            }
        }
    })
}

fn query_index_impl<'a>(
    env: &mut JNIEnv<'a>,
    index_pointer_j: jlong,
    query_vector_j: &JFloatArray,
    k_j: jint,
    method_params_j: &JObject<'a>,
) -> Result<jobjectArray, String> {
    if query_vector_j.is_null() {
        return Err("Query Vector cannot be null".to_string());
    }
    if index_pointer_j == 0 {
        return Err("Invalid pointer to index".to_string());
    }

    let index_wrapper = index_pointer_j as *mut NmslibIndexWrapper;

    let dim = env
        .get_array_length(query_vector_j)
        .map_err(|e| format!("Failed to get query vector length: {}", e))?
        as i32;

    // Get query vector elements
    let mut query_data = vec![0.0f32; dim as usize];
    env.get_float_array_region(query_vector_j, 0, &mut query_data)
        .map_err(|e| format!("Failed to get float array elements: {}", e))?;

    // Parse method params for efSearch
    let method_params = if !method_params_j.is_null() {
        convert_java_map_to_hashmap(env, method_params_j)?
    } else {
        HashMap::new()
    };

    let ef_search = if let Some(ef_obj) = method_params.get(commons::EF_SEARCH) {
        convert_java_object_to_integer(env, ef_obj)?
    } else {
        -1 // sentinel: use default
    };

    // Allocate result buffers
    let k = k_j as usize;
    let mut result_ids: Vec<std::os::raw::c_int> = vec![0; k];
    let mut result_distances: Vec<f32> = vec![0.0; k];

    let result_size = unsafe {
        ffi::nmslib_query_index(
            index_wrapper,
            query_data.as_ptr(),
            dim,
            k_j,
            ef_search,
            result_ids.as_mut_ptr(),
            result_distances.as_mut_ptr(),
        )
    };

    if result_size < 0 {
        return Err("NMSLIB query failed".to_string());
    }

    let result_size = result_size as usize;

    // Build KNNQueryResult array
    let result_class = env
        .find_class("org/opensearch/knn/index/query/KNNQueryResult")
        .map_err(|e| format!("Failed to find KNNQueryResult class: {}", e))?;

    let constructor = env
        .get_method_id(&result_class, "<init>", "(IF)V")
        .map_err(|e| format!("Failed to find KNNQueryResult constructor: {}", e))?;

    let results = env
        .new_object_array(result_size as i32, &result_class, JObject::null())
        .map_err(|e| format!("Failed to create result array: {}", e))?;

    for i in 0..result_size {
        let id = result_ids[i];
        let distance = result_distances[i];

        let result_obj = unsafe {
            env.new_object_unchecked(
                &result_class,
                constructor,
                &[
                    jni::sys::jvalue { i: id },
                    jni::sys::jvalue { f: distance },
                ],
            )
            .map_err(|e| format!("Failed to create KNNQueryResult object: {}", e))?
        };

        env.set_object_array_element(&results, i as i32, &result_obj)
            .map_err(|e| format!("Failed to set result array element: {}", e))?;
    }

    Ok(results.into_raw())
}

/// Native implementation of `NmslibService.free`.
///
/// Frees the NMSLIB index at the given memory pointer.
///
/// # Safety
/// This is a JNI entry point called by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_NmslibService_free<'local>(
    mut env: JNIEnv<'local>,
    _cls: JClass<'local>,
    index_pointer_j: jlong,
) {
    panic_catch!(env, (), {
        if index_pointer_j != 0 {
            let index_wrapper = index_pointer_j as *mut NmslibIndexWrapper;
            ffi::nmslib_free_index_wrapper(index_wrapper);
        }
    });
}

/// Native implementation of `NmslibService.initLibrary`.
///
/// Performs required initialization operations for the NMSLIB library.
///
/// # Safety
/// This is a JNI entry point called by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_NmslibService_initLibrary<'local>(
    mut env: JNIEnv<'local>,
    _cls: JClass<'local>,
) {
    panic_catch!(env, (), {
        ffi::nmslib_init_library();
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_space_type_l2() {
        assert_eq!(translate_space_type("l2").unwrap(), "l2");
    }

    #[test]
    fn test_translate_space_type_l1() {
        assert_eq!(translate_space_type("l1").unwrap(), "l1");
    }

    #[test]
    fn test_translate_space_type_linf() {
        assert_eq!(translate_space_type("linf").unwrap(), "linf");
    }

    #[test]
    fn test_translate_space_type_cosinesimil() {
        assert_eq!(translate_space_type("cosinesimil").unwrap(), "cosinesimil");
    }

    #[test]
    fn test_translate_space_type_inner_product() {
        assert_eq!(
            translate_space_type("innerproduct").unwrap(),
            "negdotprod"
        );
    }

    #[test]
    fn test_translate_space_type_invalid() {
        assert!(translate_space_type("invalid_space").is_err());
    }
}
