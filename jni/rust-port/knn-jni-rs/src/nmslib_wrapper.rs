// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Rust port of jni/src/nmslib_wrapper.cpp
//!
//! NMSLIB operations: CreateIndex, LoadIndex, LoadIndexWithStream, QueryIndex, Free, InitLibrary.

use jni::objects::{JClass, JFloatArray, JIntArray, JObject, JString};
use jni::sys::{jint, jlong, jobjectArray};
use jni::JNIEnv;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;

// ---------------------------------------------------------------------------
// Constants (mirrors knn_jni:: constants from jni_util.h / jni_util.cpp)
// ---------------------------------------------------------------------------

pub const SPACE_TYPE: &str = "spaceType";
pub const PARAMETERS: &str = "parameters";
pub const INDEX_THREAD_QUANTITY: &str = "indexThreadQty";
pub const EF_CONSTRUCTION: &str = "ef_construction";
pub const EF_CONSTRUCTION_NMSLIB: &str = "efConstruction";
pub const M: &str = "m";
pub const M_NMSLIB: &str = "M";
pub const EF_SEARCH: &str = "ef_search";

pub const L2: &str = "l2";
pub const L1: &str = "l1";
pub const LINF: &str = "linf";
pub const COSINESIMIL: &str = "cosinesimil";
pub const INNER_PRODUCT: &str = "innerproduct";
pub const NEG_DOT_PRODUCT: &str = "negdotprod";

/// Default label value -- we do not use label functionality of nmslib.
const DEFAULT_LABEL: i32 = -1;

// ---------------------------------------------------------------------------
// NMSLIB FFI declarations
// These will be moved to a dedicated ffi module later.
// ---------------------------------------------------------------------------
mod ffi {
    use std::os::raw::{c_char, c_float, c_int, c_void};

    /// Opaque handle to a similarity::Space<float>
    pub enum NmslibSpace {}

    /// Opaque handle to a similarity::Index<float>
    pub enum NmslibIndex {}

    /// Opaque handle to a similarity::Object
    pub enum NmslibObject {}

    /// Opaque handle to a similarity::KNNQuery<float>
    pub enum NmslibKNNQuery {}

    /// Opaque handle to a similarity::KNNQueue<float>
    pub enum NmslibKNNQueue {}

    // nmslib sizes for object layout
    pub const ID_SIZE: usize = 4; // sizeof(IdType) = sizeof(int)
    pub const LABEL_SIZE: usize = 4; // sizeof(LabelType) = sizeof(int)
    pub const DATALENGTH_SIZE: usize = 4; // sizeof(size_t stored as 4 bytes in object)

    extern "C" {
        // --- Library init ---
        pub fn nmslib_init_library() -> c_int;

        // --- Space creation ---
        /// Create a space by name. Returns a pointer to the created Space<float>.
        /// The caller is responsible for freeing via nmslib_free_space.
        pub fn nmslib_create_space(space_type: *const c_char) -> *mut NmslibSpace;
        pub fn nmslib_free_space(space: *mut NmslibSpace);

        // --- Index creation and lifecycle ---
        /// Create an HNSW index for the given space.
        pub fn nmslib_create_hnsw_index(
            space_type: *const c_char,
            space: *const NmslibSpace,
            data_objects: *const *const NmslibObject,
            data_count: c_int,
        ) -> *mut NmslibIndex;

        /// Build the index with parameters (null-terminated array of "key=value" C strings).
        pub fn nmslib_index_create(
            index: *mut NmslibIndex,
            params: *const *const c_char,
            num_params: c_int,
        ) -> c_int;

        /// Save the index using a writer callback.
        pub fn nmslib_index_save_with_stream(
            index: *mut NmslibIndex,
            writer_ctx: *mut c_void,
            write_fn: Option<unsafe extern "C" fn(*mut c_void, *const c_char, usize)>,
            flush_fn: Option<unsafe extern "C" fn(*mut c_void)>,
        ) -> c_int;

        /// Load the index from a file path.
        pub fn nmslib_index_load(index: *mut NmslibIndex, path: *const c_char) -> c_int;

        /// Load the index using a reader callback.
        pub fn nmslib_index_load_with_stream(
            index: *mut NmslibIndex,
            reader_ctx: *mut c_void,
            read_fn: Option<unsafe extern "C" fn(*mut c_void, *mut c_char, usize)>,
            remaining_fn: Option<unsafe extern "C" fn(*mut c_void) -> usize>,
        ) -> c_int;

        /// Set query-time parameters on the index.
        pub fn nmslib_index_set_query_time_params(
            index: *mut NmslibIndex,
            params: *const *const c_char,
            num_params: c_int,
        ) -> c_int;

        /// Free an index.
        pub fn nmslib_free_index(index: *mut NmslibIndex);

        // --- Object management ---
        /// Create an nmslib Object from raw data.
        /// id, label, data_len, data_ptr
        pub fn nmslib_create_object(
            id: c_int,
            label: c_int,
            data_len: usize,
            data: *const c_float,
        ) -> *mut NmslibObject;

        pub fn nmslib_free_object(obj: *mut NmslibObject);

        // --- Query ---
        /// Create a KNN query.
        pub fn nmslib_create_knn_query(
            space: *const NmslibSpace,
            query_obj: *const NmslibObject,
            k: c_int,
        ) -> *mut NmslibKNNQuery;

        /// Create an HNSW query with ef_search.
        pub fn nmslib_create_hnsw_query(
            space: *const NmslibSpace,
            query_obj: *const NmslibObject,
            k: c_int,
            ef_search: c_int,
        ) -> *mut NmslibKNNQuery;

        /// Execute a search.
        pub fn nmslib_index_search(
            index: *const NmslibIndex,
            query: *mut NmslibKNNQuery,
        ) -> c_int;

        /// Get the result queue from a query (cloned).
        pub fn nmslib_query_result_clone(
            query: *const NmslibKNNQuery,
        ) -> *mut NmslibKNNQueue;

        /// Get the size of the result queue.
        pub fn nmslib_queue_size(queue: *const NmslibKNNQueue) -> c_int;

        /// Get the top distance from the queue.
        pub fn nmslib_queue_top_distance(queue: *const NmslibKNNQueue) -> c_float;

        /// Pop the top element and return its id.
        pub fn nmslib_queue_pop_id(queue: *mut NmslibKNNQueue) -> c_int;

        /// Free the query.
        pub fn nmslib_free_knn_query(query: *mut NmslibKNNQuery);

        /// Free the queue.
        pub fn nmslib_free_knn_queue(queue: *mut NmslibKNNQueue);
    }
}

// ---------------------------------------------------------------------------
// NMSLIB stream trampoline callbacks for knn_shim
// ---------------------------------------------------------------------------
// These trampolines bridge between the knn_shim C++ callback signatures
// (which use size_t size, size_t nitems) and the Rust mediator objects.

/// Write trampoline for NMSLIB ostream (called by the C++ CallbackOutputStreambuf).
///
/// Signature matches `WriteCb`: `fn(ctx, src, size, nitems) -> usize`
/// The ctx is a pointer to a `NativeEngineIndexOutputMediator`.
///
/// # Safety
/// - `ctx` must point to a valid `NativeEngineIndexOutputMediator`.
/// - `src` must point to at least `size * nitems` readable bytes.
unsafe extern "C" fn nmslib_write_trampoline(
    ctx: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    nbytes: usize,
) -> usize {
    let mediator =
        &mut *(ctx as *mut crate::stream_support::NativeEngineIndexOutputMediator);
    if nbytes > 0 {
        mediator.write_bytes(src as *const u8, nbytes);
    }
    nbytes
}

/// Read trampoline for NMSLIB istream (called by the C++ CallbackInputStreambuf).
///
/// # Safety
/// - `ctx` must point to a valid `NativeEngineIndexInputMediator`.
/// - `dest` must point to at least `nbytes` writable bytes.
unsafe extern "C" fn nmslib_read_trampoline(
    ctx: *mut std::ffi::c_void,
    dest: *mut std::ffi::c_void,
    nbytes: usize,
) -> usize {
    let mediator =
        &*(ctx as *const crate::stream_support::NativeEngineIndexInputMediator);
    if nbytes > 0 {
        mediator.copy_bytes(nbytes as i64, dest as *mut u8);
    }
    nbytes
}

// ---------------------------------------------------------------------------
// IndexWrapper -- mirrors knn_jni::nmslib_wrapper::IndexWrapper
// ---------------------------------------------------------------------------

/// Wraps an NMSLIB index together with its space for lifetime management.
pub struct IndexWrapper {
    pub space: *mut ffi::NmslibSpace,
    pub index: *mut ffi::NmslibIndex,
}

impl IndexWrapper {
    /// Create a new IndexWrapper for the given (already translated) space type.
    pub fn new(space_type: &str) -> Result<Self, String> {
        let c_space = CString::new(space_type).map_err(|e| e.to_string())?;
        unsafe {
            let space = ffi::nmslib_create_space(c_space.as_ptr());
            if space.is_null() {
                return Err(format!("Failed to create space: {}", space_type));
            }
            let index = ffi::nmslib_create_hnsw_index(
                c_space.as_ptr(),
                space,
                ptr::null(),
                0,
            );
            if index.is_null() {
                ffi::nmslib_free_space(space);
                return Err("Failed to create HNSW index".to_string());
            }
            Ok(IndexWrapper { space, index })
        }
    }
}

impl Drop for IndexWrapper {
    fn drop(&mut self) {
        unsafe {
            if !self.index.is_null() {
                ffi::nmslib_free_index(self.index);
            }
            if !self.space.is_null() {
                ffi::nmslib_free_space(self.space);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: translate space type string
// ---------------------------------------------------------------------------

fn translate_space_type(space_type: &str) -> Result<String, String> {
    match space_type {
        L2 | L1 | LINF | COSINESIMIL => Ok(space_type.to_string()),
        INNER_PRODUCT => Ok(NEG_DOT_PRODUCT.to_string()),
        _ => Err(format!("Invalid spaceType: {}", space_type)),
    }
}

// ---------------------------------------------------------------------------
// Helper: convert Java Map to Rust HashMap<String, JObject>
// ---------------------------------------------------------------------------

fn java_map_to_hashmap<'local>(
    env: &mut JNIEnv<'local>,
    map_obj: &JObject<'local>,
) -> Result<HashMap<String, JObject<'local>>, String> {
    // Get the entrySet
    let entry_set = env
        .call_method(map_obj, "entrySet", "()Ljava/util/Set;", &[])
        .map_err(|e| format!("Failed to call entrySet: {}", e))?
        .l()
        .map_err(|e| format!("entrySet not an object: {}", e))?;

    let iterator = env
        .call_method(&entry_set, "iterator", "()Ljava/util/Iterator;", &[])
        .map_err(|e| format!("Failed to get iterator: {}", e))?
        .l()
        .map_err(|e| format!("iterator not an object: {}", e))?;

    let mut result = HashMap::new();

    loop {
        let has_next = env
            .call_method(&iterator, "hasNext", "()Z", &[])
            .map_err(|e| format!("hasNext failed: {}", e))?
            .z()
            .map_err(|e| format!("hasNext not bool: {}", e))?;

        if !has_next {
            break;
        }

        let entry = env
            .call_method(&iterator, "next", "()Ljava/lang/Object;", &[])
            .map_err(|e| format!("next failed: {}", e))?
            .l()
            .map_err(|e| format!("next not object: {}", e))?;

        let key_obj = env
            .call_method(&entry, "getKey", "()Ljava/lang/Object;", &[])
            .map_err(|e| format!("getKey failed: {}", e))?
            .l()
            .map_err(|e| format!("getKey not object: {}", e))?;

        let value_obj = env
            .call_method(&entry, "getValue", "()Ljava/lang/Object;", &[])
            .map_err(|e| format!("getValue failed: {}", e))?
            .l()
            .map_err(|e| format!("getValue not object: {}", e))?;

        let key_jstring = JString::from(key_obj);
        let key_str: String = env
            .get_string(&key_jstring)
            .map_err(|e| format!("get_string failed: {}", e))?
            .into();

        result.insert(key_str, value_obj);
    }

    Ok(result)
}

/// Extract an integer from a Java object (java.lang.Integer).
fn java_object_to_int(env: &mut JNIEnv, obj: &JObject) -> Result<i32, String> {
    let val = env
        .call_method(obj, "intValue", "()I", &[])
        .map_err(|e| format!("intValue failed: {}", e))?
        .i()
        .map_err(|e| format!("intValue not int: {}", e))?;
    Ok(val)
}

/// Extract a string from a Java Object (java.lang.String or .toString()).
fn java_object_to_string(env: &mut JNIEnv, obj: &JObject) -> Result<String, String> {
    let jstr = JString::from(unsafe { JObject::from_raw(obj.as_raw()) });
    let s: String = env
        .get_string(&jstr)
        .map_err(|e| format!("get_string failed: {}", e))?
        .into();
    Ok(s)
}

/// Get integer method parameter from the map, returns default if not found.
fn get_integer_method_parameter(
    env: &mut JNIEnv,
    params: &HashMap<String, JObject>,
    key: &str,
    default: i32,
) -> i32 {
    match params.get(key) {
        Some(obj) => java_object_to_int(env, obj).unwrap_or(default),
        None => default,
    }
}

// ---------------------------------------------------------------------------
// Panic-catching boundary macro
// ---------------------------------------------------------------------------

/// Catches Rust panics and converts them to Java exceptions.
macro_rules! panic_catch {
    ($env:expr, $default:expr, $body:block) => {{
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        match result {
            Ok(val) => val,
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic in native code".to_string()
                };
                let _ = $env.throw_new("java/lang/RuntimeException", &msg);
                $default
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// JNI exported functions
// ---------------------------------------------------------------------------

/// Create an NMSLIB index from ids and vectors, writing it to the output stream.
///
/// Corresponds to: knn_jni::nmslib_wrapper::CreateIndex
// #[no_mangle] -- removed: entry point is in nmslib_service_jni.rs
pub extern "system" fn Java_org_opensearch_knn_jni_NmslibService_createIndex<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    ids: JIntArray,
    vectors_address: jlong,
    dim: jint,
    output: JObject,
    parameters: JObject<'local>,
) {
    panic_catch!(env, (), {
        if let Err(e) = create_index_impl(&mut env, &ids, vectors_address, dim, &output, &parameters) {
            let _ = env.throw_new("java/lang/RuntimeException", &e);
        }
    });
}

fn create_index_impl<'a>(
    env: &mut JNIEnv<'a>,
    ids: &JIntArray,
    vectors_address: jlong,
    dim: jint,
    output: &JObject,
    parameters: &JObject<'a>,
) -> Result<(), String> {
    if ids.is_null() {
        return Err("IDs cannot be null".to_string());
    }
    if vectors_address <= 0 {
        return Err("VectorsAddress cannot be less than 0".to_string());
    }
    if dim <= 0 {
        return Err("Vectors dimensions cannot be less than or equal to 0".to_string());
    }
    if output.is_null() {
        return Err("Index output stream cannot be null".to_string());
    }
    if parameters.is_null() {
        return Err("Parameters cannot be null".to_string());
    }

    // Convert parameters map
    let params_map = java_map_to_hashmap(env, parameters)?;
    let mut index_parameters: Vec<String> = Vec::new();

    // Algorithm parameters will be in a sub map under "parameters"
    if let Some(sub_params_obj) = params_map.get(PARAMETERS) {
        let sub_params = java_map_to_hashmap(env, sub_params_obj)?;

        if let Some(ef_obj) = sub_params.get(EF_CONSTRUCTION) {
            let ef = java_object_to_int(env, ef_obj)?;
            index_parameters.push(format!("{}={}", EF_CONSTRUCTION_NMSLIB, ef));
        }

        if let Some(m_obj) = sub_params.get(M) {
            let m_val = java_object_to_int(env, m_obj)?;
            index_parameters.push(format!("{}={}", M_NMSLIB, m_val));
        }
    }

    if let Some(thread_qty_obj) = params_map.get(INDEX_THREAD_QUANTITY) {
        let thread_qty = java_object_to_int(env, thread_qty_obj)?;
        index_parameters.push(format!("{}={}", INDEX_THREAD_QUANTITY, thread_qty));
    }

    // Get space type
    let space_type_obj = params_map
        .get(SPACE_TYPE)
        .ok_or_else(|| "spaceType not found in parameters".to_string())?;
    let space_type_str = java_object_to_string(env, space_type_obj)?;
    let space_type = translate_space_type(&space_type_str)?;

    // Get vectors from the address (Vec<f32> was allocated in C++ side)
    let vectors_ptr = vectors_address as *const Vec<f32>;
    if vectors_ptr.is_null() {
        return Err("vectors_address pointer is null".to_string());
    }
    let input_vectors: &Vec<f32> = unsafe { &*vectors_ptr };
    let dim_usize = dim as usize;
    let num_vectors = input_vectors.len() / dim_usize;

    if num_vectors == 0 {
        return Err("Number of vectors cannot be 0".to_string());
    }

    // Get IDs
    let ids_array = unsafe {
        env.get_array_elements(ids, jni::objects::ReleaseMode::NoCopyBack)
            .map_err(|e| format!("Failed to get int array elements: {}", e))?
    };
    let num_ids = ids_array.len();
    if num_ids != num_vectors {
        return Err("Number of IDs does not match number of vectors".to_string());
    }

    // Create NMSLIB objects and build index
    // This section involves complex NMSLIB object layout and buffer management.
    // The full implementation requires calling into NMSLIB's C++ API through FFI.
    // Stubbed for now -- the actual implementation will:
    // 1. Allocate an object buffer with the NMSLIB object layout
    // 2. Create similarity::Object instances pointing into the buffer
    // 3. Create an HNSW index with the given space
    // 4. Call CreateIndex with the parameters
    // 5. SaveIndexWithStream via the output mediator

    let c_space_type = CString::new(space_type.as_str()).map_err(|e| e.to_string())?;

    // Convert index_parameters to C strings
    let c_params: Vec<CString> = index_parameters
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap())
        .collect();
    let c_param_ptrs: Vec<*const c_char> = c_params.iter().map(|cs| cs.as_ptr()).collect();

    unsafe {
        // Create space
        let space = ffi::nmslib_create_space(c_space_type.as_ptr());
        if space.is_null() {
            return Err(format!("Failed to create space: {}", space_type));
        }

        // Create objects
        let vector_size_bytes = dim_usize * std::mem::size_of::<f32>();
        let mut objects: Vec<*mut ffi::NmslibObject> = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let vector_offset = i * dim_usize;
            let data_ptr = input_vectors.as_ptr().add(vector_offset);
            let obj = ffi::nmslib_create_object(
                ids_array[i],
                DEFAULT_LABEL,
                vector_size_bytes,
                data_ptr,
            );
            if obj.is_null() {
                // Cleanup previously created objects
                for prev in &objects {
                    ffi::nmslib_free_object(*prev);
                }
                ffi::nmslib_free_space(space);
                return Err("Failed to create NMSLIB object".to_string());
            }
            objects.push(obj);
        }

        // Create index
        let obj_ptrs: Vec<*const ffi::NmslibObject> =
            objects.iter().map(|o| *o as *const ffi::NmslibObject).collect();
        let index = ffi::nmslib_create_hnsw_index(
            c_space_type.as_ptr(),
            space,
            obj_ptrs.as_ptr(),
            num_vectors as c_int,
        );

        if index.is_null() {
            for obj in &objects {
                ffi::nmslib_free_object(*obj);
            }
            ffi::nmslib_free_space(space);
            return Err("Failed to create HNSW index".to_string());
        }

        // Build the index
        let ret = ffi::nmslib_index_create(
            index,
            c_param_ptrs.as_ptr(),
            c_params.len() as c_int,
        );
        if ret != 0 {
            ffi::nmslib_free_index(index);
            for obj in &objects {
                ffi::nmslib_free_object(*obj);
            }
            ffi::nmslib_free_space(space);
            return Err("Failed to build index".to_string());
        }

        // Save index with stream via knn_shim ostream backed by mediator callbacks.
        // The mediator writes bytes to the Java IndexOutputWithBuffer via JNI.
        use crate::ffi::knn_shim;
        use std::ffi::c_void;

        let raw_env = env.get_raw();
        let output_obj = output.as_raw();

        let mediator = Box::new(
            crate::stream_support::NativeEngineIndexOutputMediator::new(raw_env, output_obj),
        );
        let mediator_ptr = &*mediator as *const crate::stream_support::NativeEngineIndexOutputMediator
            as *mut c_void;

        let ostream_wrapper = knn_shim::knn_shim_create_ostream(
            mediator_ptr,
            nmslib_write_trampoline,
        );
        if ostream_wrapper.is_null() {
            ffi::nmslib_free_index(index);
            for obj in &objects {
                ffi::nmslib_free_object(*obj);
            }
            ffi::nmslib_free_space(space);
            return Err("Failed to create ostream for NMSLIB save".to_string());
        }

        let ostream_ptr = knn_shim::knn_shim_get_ostream_ptr(ostream_wrapper);

        // Pass the raw ostream* to NMSLIB's save function
        let save_ret = ffi::nmslib_index_save_with_stream(
            index,
            ostream_ptr,
            None,
            None,
        );

        // Flush and free the ostream wrapper
        knn_shim::knn_shim_free_ostream(ostream_wrapper);
        // Drop the mediator (flush happens in free_ostream above via streambuf sync)
        drop(mediator);

        if save_ret != 0 {
            ffi::nmslib_free_index(index);
            for obj in &objects {
                ffi::nmslib_free_object(*obj);
            }
            ffi::nmslib_free_space(space);
            return Err("Failed to save index with stream".to_string());
        }

        // Cleanup
        ffi::nmslib_free_index(index);
        for obj in &objects {
            ffi::nmslib_free_object(*obj);
        }
        ffi::nmslib_free_space(space);
    }

    // Delete the input vectors (mirrors C++ `delete inputVectors`)
    unsafe {
        let _ = Box::from_raw(vectors_address as *mut Vec<f32>);
    }

    Ok(())
}

/// Load an NMSLIB index from a file path.
///
/// Corresponds to: knn_jni::nmslib_wrapper::LoadIndex
// #[no_mangle] -- removed: entry point is in nmslib_service_jni.rs
pub extern "system" fn Java_org_opensearch_knn_jni_NmslibService_loadIndex<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    index_path: JString,
    parameters: JObject<'local>,
) -> jlong {
    panic_catch!(env, 0, {
        match load_index_impl(&mut env, &index_path, &parameters) {
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
    index_path: &JString,
    parameters: &JObject<'a>,
) -> Result<jlong, String> {
    if index_path.is_null() {
        return Err("Index path cannot be null".to_string());
    }
    if parameters.is_null() {
        return Err("Parameters cannot be null".to_string());
    }

    let path_str: String = env
        .get_string(index_path)
        .map_err(|e| format!("Failed to get index path string: {}", e))?
        .into();

    let params_map = java_map_to_hashmap(env, parameters)?;

    // Get space type
    let space_type_obj = params_map
        .get(SPACE_TYPE)
        .ok_or_else(|| "spaceType not found in parameters".to_string())?;
    let space_type_str = java_object_to_string(env, space_type_obj)?;
    let space_type = translate_space_type(&space_type_str)?;

    // Parse query params
    let mut query_params: Vec<String> = Vec::new();
    if let Some(ef_search_obj) = params_map.get("efSearch") {
        let ef_search = java_object_to_int(env, ef_search_obj)?;
        query_params.push(format!("efSearch={}", ef_search));
    }

    // Create IndexWrapper and load
    let index_wrapper = Box::new(IndexWrapper::new(&space_type)?);

    let c_path = CString::new(path_str.as_str()).map_err(|e| e.to_string())?;
    let c_query_params: Vec<CString> = query_params
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap())
        .collect();
    let c_query_param_ptrs: Vec<*const c_char> =
        c_query_params.iter().map(|cs| cs.as_ptr()).collect();

    unsafe {
        let ret = ffi::nmslib_index_load(index_wrapper.index, c_path.as_ptr());
        if ret != 0 {
            return Err(format!("Failed to load index from: {}", path_str));
        }

        let ret = ffi::nmslib_index_set_query_time_params(
            index_wrapper.index,
            c_query_param_ptrs.as_ptr(),
            c_query_params.len() as c_int,
        );
        if ret != 0 {
            return Err("Failed to set query time params".to_string());
        }
    }

    // Leak the Box so it lives until Free is called
    let ptr = Box::into_raw(index_wrapper);
    Ok(ptr as jlong)
}

/// Load an NMSLIB index from an input stream.
///
/// Corresponds to: knn_jni::nmslib_wrapper::LoadIndexWithStream
// #[no_mangle] -- removed: entry point is in nmslib_service_jni.rs
pub extern "system" fn Java_org_opensearch_knn_jni_NmslibService_loadIndexWithStream<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    read_stream: JObject,
    parameters: JObject<'local>,
) -> jlong {
    panic_catch!(env, 0, {
        match load_index_with_stream_impl(&mut env, &read_stream, &parameters) {
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
    parameters: &JObject<'a>,
) -> Result<jlong, String> {
    if read_stream.is_null() {
        return Err("Read stream cannot be null".to_string());
    }
    if parameters.is_null() {
        return Err("Parameters cannot be null".to_string());
    }

    let params_map = java_map_to_hashmap(env, parameters)?;

    // Get space type
    let space_type_obj = params_map
        .get(SPACE_TYPE)
        .ok_or_else(|| "spaceType not found in parameters".to_string())?;
    let space_type_str = java_object_to_string(env, space_type_obj)?;
    let space_type = translate_space_type(&space_type_str)?;

    // Parse query params
    let mut query_params: Vec<String> = Vec::new();
    if let Some(ef_search_obj) = params_map.get("efSearch") {
        let ef_search = java_object_to_int(env, ef_search_obj)?;
        query_params.push(format!("efSearch={}", ef_search));
    }

    // Create IndexWrapper
    let index_wrapper = Box::new(IndexWrapper::new(&space_type)?);

    let c_query_params: Vec<CString> = query_params
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap())
        .collect();
    let c_query_param_ptrs: Vec<*const c_char> =
        c_query_params.iter().map(|cs| cs.as_ptr()).collect();

    unsafe {
        // Set query time params first (matches C++ LoadIndexWithStream order)
        let ret = ffi::nmslib_index_set_query_time_params(
            index_wrapper.index,
            c_query_param_ptrs.as_ptr(),
            c_query_params.len() as c_int,
        );
        if ret != 0 {
            return Err("Failed to set query time params".to_string());
        }

        // Load index via knn_shim istream backed by mediator callbacks.
        // The mediator reads bytes from the Java IndexInputWithBuffer via JNI.
        use crate::ffi::knn_shim;
        use std::ffi::c_void;

        let raw_env = env.get_raw();
        let read_stream_obj = read_stream.as_raw();

        let mediator = Box::new(
            crate::stream_support::NativeEngineIndexInputMediator::new(raw_env, read_stream_obj),
        );
        let mediator_ptr = &*mediator as *const crate::stream_support::NativeEngineIndexInputMediator
            as *mut c_void;

        let istream_wrapper = knn_shim::knn_shim_create_istream(
            mediator_ptr,
            nmslib_read_trampoline,
        );
        if istream_wrapper.is_null() {
            return Err("Failed to create istream for NMSLIB load".to_string());
        }

        let istream_ptr = knn_shim::knn_shim_get_istream_ptr(istream_wrapper);

        // Pass the raw istream* to NMSLIB's load function
        let load_ret = ffi::nmslib_index_load_with_stream(
            index_wrapper.index,
            istream_ptr,
            None,
            None,
        );

        // Free the istream wrapper
        knn_shim::knn_shim_free_istream(istream_wrapper);
        drop(mediator);

        if load_ret != 0 {
            return Err("Failed to load index with stream".to_string());
        }
    }

    // Leak the Box so it lives until Free is called
    let ptr = Box::into_raw(index_wrapper);
    Ok(ptr as jlong)
}

/// Query a loaded NMSLIB index.
///
/// Corresponds to: knn_jni::nmslib_wrapper::QueryIndex
// #[no_mangle] -- removed: entry point is in nmslib_service_jni.rs
pub extern "system" fn Java_org_opensearch_knn_jni_NmslibService_queryIndex<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    index_pointer: jlong,
    query_vector: JFloatArray,
    k: jint,
    method_params: JObject<'local>,
) -> jobjectArray {
    panic_catch!(env, ptr::null_mut(), {
        match query_index_impl(&mut env, index_pointer, &query_vector, k, &method_params) {
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
    index_pointer: jlong,
    query_vector: &JFloatArray,
    k: jint,
    method_params: &JObject<'a>,
) -> Result<jobjectArray, String> {
    if query_vector.is_null() {
        return Err("Query Vector cannot be null".to_string());
    }
    if index_pointer == 0 {
        return Err("Invalid pointer to index".to_string());
    }

    let index_wrapper = unsafe { &*(index_pointer as *const IndexWrapper) };

    // Get the query vector data
    let dim = env
        .get_array_length(query_vector)
        .map_err(|e| format!("Failed to get query vector length: {}", e))? as usize;
    let mut query_data = vec![0.0f32; dim];
    env.get_float_array_region(query_vector, 0, &mut query_data)
        .map_err(|e| format!("Failed to get float array elements: {}", e))?;
    let vector_size_bytes = dim * std::mem::size_of::<f32>();

    // Create query object
    let query_obj = unsafe {
        ffi::nmslib_create_object(
            -1,
            -1,
            vector_size_bytes,
            query_data.as_ptr() as *const c_float,
        )
    };
    if query_obj.is_null() {
        return Err("Failed to create query object".to_string());
    }

    // Parse method params for ef_search
    let params: HashMap<String, JObject> = if !method_params.is_null() {
        java_map_to_hashmap(env, method_params)?
    } else {
        HashMap::new()
    };
    let query_ef_search = get_integer_method_parameter(env, &params, EF_SEARCH, -1);

    // Create query
    let query = unsafe {
        if query_ef_search == -1 {
            ffi::nmslib_create_knn_query(index_wrapper.space, query_obj, k)
        } else {
            ffi::nmslib_create_hnsw_query(index_wrapper.space, query_obj, k, query_ef_search)
        }
    };
    if query.is_null() {
        unsafe { ffi::nmslib_free_object(query_obj) };
        return Err("Failed to create KNN query".to_string());
    }

    // Execute search
    unsafe {
        let ret = ffi::nmslib_index_search(index_wrapper.index, query);
        if ret != 0 {
            ffi::nmslib_free_knn_query(query);
            ffi::nmslib_free_object(query_obj);
            return Err("Index search failed".to_string());
        }
    }

    // Get results
    let neighbors = unsafe { ffi::nmslib_query_result_clone(query) };
    if neighbors.is_null() {
        unsafe {
            ffi::nmslib_free_knn_query(query);
            ffi::nmslib_free_object(query_obj);
        }
        return Err("Failed to clone query results".to_string());
    }

    let result_size = unsafe { ffi::nmslib_queue_size(neighbors) } as i32;

    // Create Java result array
    let result_class = env
        .find_class("org/opensearch/knn/index/query/KNNQueryResult")
        .map_err(|e| format!("Failed to find KNNQueryResult class: {}", e))?;

    let constructor = env
        .get_method_id(&result_class, "<init>", "(IF)V")
        .map_err(|e| format!("Failed to find KNNQueryResult constructor: {}", e))?;

    let results = env
        .new_object_array(result_size, &result_class, JObject::null())
        .map_err(|e| format!("Failed to create result array: {}", e))?;

    for i in 0..result_size {
        let distance = unsafe { ffi::nmslib_queue_top_distance(neighbors) };
        let id = unsafe { ffi::nmslib_queue_pop_id(neighbors) };

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

        env.set_object_array_element(&results, i, &result_obj)
            .map_err(|e| format!("Failed to set array element: {}", e))?;
    }

    // Cleanup
    unsafe {
        ffi::nmslib_free_knn_queue(neighbors);
        ffi::nmslib_free_knn_query(query);
        ffi::nmslib_free_object(query_obj);
    }

    Ok(results.into_raw())
}

/// Free a loaded NMSLIB index.
///
/// Corresponds to: knn_jni::nmslib_wrapper::Free
// #[no_mangle] -- removed: entry point is in nmslib_service_jni.rs
pub extern "system" fn Java_org_opensearch_knn_jni_NmslibService_free(
    mut env: JNIEnv,
    _class: JClass,
    index_pointer: jlong,
) {
    panic_catch!(env, (), {
        if index_pointer == 0 {
            return;
        }
        unsafe {
            let _ = Box::from_raw(index_pointer as *mut IndexWrapper);
            // Box Drop will call IndexWrapper::drop which frees the index and space
        }
    });
}

/// Initialize the NMSLIB library.
///
/// Corresponds to: knn_jni::nmslib_wrapper::InitLibrary
// #[no_mangle] -- removed: entry point is in nmslib_service_jni.rs
pub extern "system" fn Java_org_opensearch_knn_jni_NmslibService_initLibrary(
    mut env: JNIEnv,
    _class: JClass,
) {
    panic_catch!(env, (), {
        unsafe {
            ffi::nmslib_init_library();
        }
    });
}
