// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! JNI entry points for the `org.opensearch.knn.jni.FaissService` Java class.
//!
//! Each exported function corresponds to a native method declared in the Java class.
//! Panics are caught at the boundary and converted into Java exceptions via
//! `crate::jni_catch_panic` / `crate::jni_catch_panic_void`.
//!
//! This layer calls directly into Rust implementations in:
//! - `crate::faiss_wrapper` (load, query, train, free, etc.)
//! - `crate::faiss_index_service` (init_index, insert_to_index, write_index)
//! - `crate::commons` (store_vector_data)
//! - `crate::jni_util` (JNI type conversions)

#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

use jni::objects::{
    JByteArray, JClass, JFloatArray, JIntArray, JLongArray, JObject, JObjectArray, JString,
};
use jni::sys::{jboolean, jfloat, jint, jlong};
use jni::JNIEnv;
use std::ptr;

use crate::faiss_index_service::{
    BinaryIndexService, ByteIndexService, FloatIndexService, IndexService,
};
use crate::faiss_wrapper;
use crate::{jni_catch_panic, jni_catch_panic_void, KnnError, KnnResult};

// ---------------------------------------------------------------------------
// Helper: convert FaissWrapperError to KnnError
// ---------------------------------------------------------------------------

impl From<faiss_wrapper::FaissWrapperError> for KnnError {
    fn from(e: faiss_wrapper::FaissWrapperError) -> Self {
        match e {
            faiss_wrapper::FaissWrapperError::Runtime(msg) => KnnError::NativeLibrary(msg),
            faiss_wrapper::FaissWrapperError::Jni(je) => KnnError::Jni(je),
            faiss_wrapper::FaissWrapperError::NullPointer(msg) => KnnError::NullPointer(msg),
        }
    }
}

impl From<crate::faiss_index_service::IndexServiceError> for KnnError {
    fn from(e: crate::faiss_index_service::IndexServiceError) -> Self {
        match e {
            crate::faiss_index_service::IndexServiceError::IndexNotTrained => {
                KnnError::NativeLibrary("Index is not trained".to_string())
            }
            crate::faiss_index_service::IndexServiceError::FaissError(msg) => {
                KnnError::NativeLibrary(msg)
            }
            crate::faiss_index_service::IndexServiceError::InvalidArgument(msg) => {
                KnnError::NativeLibrary(msg)
            }
            crate::faiss_index_service::IndexServiceError::WriteError(msg) => {
                KnnError::NativeLibrary(msg)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: extract index service parameters from a Java Map<String, Object>
// ---------------------------------------------------------------------------

/// Parse index-related parameters from a Java Map into the structures needed
/// by the IndexService trait methods.
///
/// Returns (metric, index_description, thread_count, parameters_map).
unsafe fn parse_index_params(
    env: &mut JNIEnv,
    parameters: &JObject,
) -> KnnResult<(
    crate::faiss_index_service::FaissMetricType,
    String,
    i32,
    std::collections::HashMap<String, crate::faiss_index_service::ParamValue>,
)> {
    use crate::jni_util::{
        EF_CONSTRUCTION, EF_SEARCH, INDEX_DESCRIPTION, INDEX_THREAD_QUANTITY, NPROBES, PARAMETERS,
        SPACE_TYPE,
    };

    let jni_util = crate::global_jni_util();

    // Extract space type
    let space_type_obj = get_map_value(env, parameters, SPACE_TYPE)?;
    let space_type_str = jni_util
        .convert_java_object_to_rust_string(env, &space_type_obj)
        .map_err(|e| KnnError::InvalidArgument(format!("spaceType: {}", e)))?;
    let metric = match space_type_str.as_str() {
        "l2" | "hamming" => crate::faiss_index_service::FaissMetricType::MetricL2,
        "innerproduct" | "cosinesimil" => {
            crate::faiss_index_service::FaissMetricType::MetricInnerProduct
        }
        other => {
            return Err(KnnError::NativeLibrary(
                format!("Invalid spaceType: {}", other),
            ));
        }
    };

    // Extract index description
    let index_desc_obj = get_map_value(env, parameters, INDEX_DESCRIPTION)?;
    let index_desc_str = jni_util
        .convert_java_object_to_rust_string(env, &index_desc_obj)
        .map_err(|e| KnnError::InvalidArgument(format!("index_description: {}", e)))?;

    // Extract thread count (optional)
    let mut thread_count = 0i32;
    if let Ok(tc_obj) = get_map_value(env, parameters, INDEX_THREAD_QUANTITY) {
        if !tc_obj.is_null() {
            if let Ok(val) = env.call_method(&tc_obj, "intValue", "()I", &[]) {
                if let Ok(tc) = val.i() {
                    thread_count = tc;
                }
            }
        }
    }

    // Extract sub-parameters (ef_construction, ef_search, nprobes)
    let mut params_map =
        std::collections::HashMap::<String, crate::faiss_index_service::ParamValue>::new();
    if let Ok(sub_params_obj) = get_map_value(env, parameters, PARAMETERS) {
        if !sub_params_obj.is_null() {
            if let Some(ef_c) = get_int_from_map(env, &sub_params_obj, EF_CONSTRUCTION) {
                params_map.insert(
                    EF_CONSTRUCTION.to_string(),
                    crate::faiss_index_service::ParamValue::Int(ef_c),
                );
            }
            if let Some(ef_s) = get_int_from_map(env, &sub_params_obj, EF_SEARCH) {
                params_map.insert(
                    EF_SEARCH.to_string(),
                    crate::faiss_index_service::ParamValue::Int(ef_s),
                );
            }
            if let Some(np) = get_int_from_map(env, &sub_params_obj, NPROBES) {
                params_map.insert(
                    NPROBES.to_string(),
                    crate::faiss_index_service::ParamValue::Int(np),
                );
            }
        }
    }

    Ok((metric, index_desc_str, thread_count, params_map))
}

/// Get a value from a Java Map<String, Object> by key.
unsafe fn get_map_value<'local>(
    env: &mut JNIEnv<'local>,
    map: &JObject,
    key: &str,
) -> KnnResult<JObject<'local>> {
    let jkey = env.new_string(key)?;
    let result = env
        .call_method(
            map,
            "get",
            "(Ljava/lang/Object;)Ljava/lang/Object;",
            &[(&jkey).into()],
        )?
        .l()?;
    Ok(result)
}

/// Get an integer from a Java Map by key, returning None if not present.
unsafe fn get_int_from_map(env: &mut JNIEnv, map: &JObject, key: &str) -> Option<i32> {
    let jkey = env.new_string(key).ok()?;
    let val = env
        .call_method(
            map,
            "get",
            "(Ljava/lang/Object;)Ljava/lang/Object;",
            &[(&jkey).into()],
        )
        .ok()?
        .l()
        .ok()?;
    if val.is_null() {
        return None;
    }
    let int_val = env.call_method(&val, "intValue", "()I", &[]).ok()?.i().ok()?;
    Some(int_val)
}

// ---------------------------------------------------------------------------
// initIndex / initBinaryIndex / initByteIndex
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_initIndex(
    mut env: JNIEnv,
    _cls: JClass,
    num_docs: jlong,
    dim: jint,
    parameters: JObject,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        if dim <= 0 {
            return Err(KnnError::InvalidArgument(
                "Vectors dimensions cannot be less than or equal to 0".to_string(),
            ));
        }
        if parameters.is_null() {
            return Err(KnnError::InvalidArgument(
                "Parameters cannot be null".to_string(),
            ));
        }

        let (metric, index_desc, thread_count, params_map) =
            parse_index_params(env, &parameters)?;
        let service = FloatIndexService::new();
        let result = service.init_index(
            metric,
            &index_desc,
            dim,
            num_docs as i32,
            thread_count,
            &params_map,
        )?;
        Ok(result)
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_initBinaryIndex(
    mut env: JNIEnv,
    _cls: JClass,
    num_docs: jlong,
    dim: jint,
    parameters: JObject,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        if dim <= 0 {
            return Err(KnnError::InvalidArgument(
                "Vectors dimensions cannot be less than or equal to 0".to_string(),
            ));
        }
        if parameters.is_null() {
            return Err(KnnError::InvalidArgument(
                "Parameters cannot be null".to_string(),
            ));
        }

        let (metric, index_desc, thread_count, params_map) =
            parse_index_params(env, &parameters)?;
        let service = BinaryIndexService::new();
        let result = service.init_index(
            metric,
            &index_desc,
            dim,
            num_docs as i32,
            thread_count,
            &params_map,
        )?;
        Ok(result)
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_initByteIndex(
    mut env: JNIEnv,
    _cls: JClass,
    num_docs: jlong,
    dim: jint,
    parameters: JObject,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        if dim <= 0 {
            return Err(KnnError::InvalidArgument(
                "Vectors dimensions cannot be less than or equal to 0".to_string(),
            ));
        }
        if parameters.is_null() {
            return Err(KnnError::InvalidArgument(
                "Parameters cannot be null".to_string(),
            ));
        }

        let (metric, index_desc, thread_count, params_map) =
            parse_index_params(env, &parameters)?;
        let service = ByteIndexService::new();
        let result = service.init_index(
            metric,
            &index_desc,
            dim,
            num_docs as i32,
            thread_count,
            &params_map,
        )?;
        Ok(result)
    })
}

// ---------------------------------------------------------------------------
// insertToIndex / insertToBinaryIndex / insertToByteIndex
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_insertToIndex(
    mut env: JNIEnv,
    _cls: JClass,
    ids: JIntArray,
    vectors_address: jlong,
    dim: jint,
    index_address: jlong,
    thread_count: jint,
) {
    jni_catch_panic_void(&mut env, |env| {
        let jni_util = crate::global_jni_util();
        let ids_vec = jni_util
            .convert_java_int_array_to_i64_vector(env, &ids)
            .map_err(|e| KnnError::InvalidArgument(format!("{}", e)))?;
        let service = FloatIndexService::new();
        service.insert_to_index(
            dim,
            ids_vec.len() as i32,
            thread_count,
            vectors_address,
            &ids_vec,
            index_address,
        )?;
        Ok(())
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_insertToBinaryIndex(
    mut env: JNIEnv,
    _cls: JClass,
    ids: JIntArray,
    vectors_address: jlong,
    dim: jint,
    index_address: jlong,
    thread_count: jint,
) {
    jni_catch_panic_void(&mut env, |env| {
        let jni_util = crate::global_jni_util();
        let ids_vec = jni_util
            .convert_java_int_array_to_i64_vector(env, &ids)
            .map_err(|e| KnnError::InvalidArgument(format!("{}", e)))?;
        let service = BinaryIndexService::new();
        service.insert_to_index(
            dim,
            ids_vec.len() as i32,
            thread_count,
            vectors_address,
            &ids_vec,
            index_address,
        )?;
        Ok(())
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_insertToByteIndex(
    mut env: JNIEnv,
    _cls: JClass,
    ids: JIntArray,
    vectors_address: jlong,
    dim: jint,
    index_address: jlong,
    thread_count: jint,
) {
    jni_catch_panic_void(&mut env, |env| {
        let jni_util = crate::global_jni_util();
        let ids_vec = jni_util
            .convert_java_int_array_to_i64_vector(env, &ids)
            .map_err(|e| KnnError::InvalidArgument(format!("{}", e)))?;
        let service = ByteIndexService::new();
        service.insert_to_index(
            dim,
            ids_vec.len() as i32,
            thread_count,
            vectors_address,
            &ids_vec,
            index_address,
        )?;
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// writeIndex / writeBinaryIndex / writeByteIndex
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_writeIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_address: jlong,
    output: JObject,
) {
    jni_catch_panic_void(&mut env, |env| {
        use crate::stream_support;
        use crate::faiss_index_service::FaissIOWriter as IndexServiceIOWriter;

        let raw_env = env.get_raw();
        let raw_obj = output.as_raw();

        let writer = stream_support::create_faiss_stream_writer(raw_env, raw_obj);

        let service = FloatIndexService::new();
        // Cast between the two opaque FaissIOWriter types (both are zero-sized repr(C) opaques)
        let io_writer_ptr = writer.io_writer as *mut IndexServiceIOWriter;
        let result = service.write_index(io_writer_ptr, index_address, false);

        // Flush and clean up the writer
        stream_support::destroy_faiss_stream_writer(writer);

        result.map_err(KnnError::from)
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_writeBinaryIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_address: jlong,
    output: JObject,
    skip_flat: jboolean,
) {
    jni_catch_panic_void(&mut env, |env| {
        use crate::stream_support;
        use crate::faiss_index_service::FaissIOWriter as IndexServiceIOWriter;

        let raw_env = env.get_raw();
        let raw_obj = output.as_raw();

        let writer = stream_support::create_faiss_stream_writer(raw_env, raw_obj);

        let service = BinaryIndexService::new();
        let io_writer_ptr = writer.io_writer as *mut IndexServiceIOWriter;
        let result = service.write_index(io_writer_ptr, index_address, skip_flat != 0);

        stream_support::destroy_faiss_stream_writer(writer);

        result.map_err(KnnError::from)
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_writeByteIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_address: jlong,
    output: JObject,
) {
    jni_catch_panic_void(&mut env, |env| {
        use crate::stream_support;
        use crate::faiss_index_service::FaissIOWriter as IndexServiceIOWriter;

        let raw_env = env.get_raw();
        let raw_obj = output.as_raw();

        let writer = stream_support::create_faiss_stream_writer(raw_env, raw_obj);

        let service = ByteIndexService::new();
        let io_writer_ptr = writer.io_writer as *mut IndexServiceIOWriter;
        let result = service.write_index(io_writer_ptr, index_address, false);

        stream_support::destroy_faiss_stream_writer(writer);

        result.map_err(KnnError::from)
    });
}

// ---------------------------------------------------------------------------
// createIndexFromTemplate / createBinaryIndexFromTemplate / createByteIndexFromTemplate
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_createIndexFromTemplate(
    mut env: JNIEnv,
    _cls: JClass,
    ids: JIntArray,
    vectors_address: jlong,
    dim: jint,
    output: JObject,
    template_index: JByteArray,
    parameters: JObject,
) {
    jni_catch_panic_void(&mut env, |env| {
        faiss_wrapper::create_index_from_template(
            env,
            &ids,
            vectors_address,
            dim,
            &output,
            &template_index,
            &parameters,
        )?;
        Ok(())
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_createBinaryIndexFromTemplate(
    mut env: JNIEnv,
    _cls: JClass,
    ids: JIntArray,
    vectors_address: jlong,
    dim: jint,
    output: JObject,
    template_index: JByteArray,
    parameters: JObject,
) {
    jni_catch_panic_void(&mut env, |env| {
        faiss_wrapper::create_binary_index_from_template(
            env,
            &ids,
            vectors_address,
            dim,
            &output,
            &template_index,
            &parameters,
        )?;
        Ok(())
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_createByteIndexFromTemplate(
    mut env: JNIEnv,
    _cls: JClass,
    ids: JIntArray,
    vectors_address: jlong,
    dim: jint,
    output: JObject,
    template_index: JByteArray,
    parameters: JObject,
) {
    jni_catch_panic_void(&mut env, |env| {
        faiss_wrapper::create_byte_index_from_template(
            env,
            &ids,
            vectors_address,
            dim,
            &output,
            &template_index,
            &parameters,
        )?;
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// loadIndex / loadIndexWithStream / loadBinaryIndex / loadBinaryIndexWithStream
// loadIndexWithStreamADCParams
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_loadIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_path: JString,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        let result = faiss_wrapper::load_index(env, &index_path)?;
        Ok(result)
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_loadIndexWithStream(
    mut env: JNIEnv,
    _cls: JClass,
    read_stream: JObject,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        use crate::stream_support;

        let raw_env = env.get_raw();
        let raw_obj = read_stream.as_raw();

        let reader = stream_support::create_faiss_stream_reader(raw_env, raw_obj);
        let result = faiss_wrapper::load_index_with_stream(reader.io_reader);

        // Clean up the reader regardless of success/failure
        stream_support::destroy_faiss_stream_reader(reader);

        result.map_err(KnnError::from)
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_loadBinaryIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_path: JString,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        let result = faiss_wrapper::load_binary_index(env, &index_path)?;
        Ok(result)
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_loadBinaryIndexWithStream(
    mut env: JNIEnv,
    _cls: JClass,
    read_stream: JObject,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        use crate::stream_support;

        let raw_env = env.get_raw();
        let raw_obj = read_stream.as_raw();

        let reader = stream_support::create_faiss_stream_reader(raw_env, raw_obj);
        let result = faiss_wrapper::load_binary_index_with_stream(reader.io_reader);

        stream_support::destroy_faiss_stream_reader(reader);

        result.map_err(KnnError::from)
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_loadIndexWithStreamADCParams(
    mut env: JNIEnv,
    _cls: JClass,
    read_stream: JObject,
    parameters: JObject,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        use crate::stream_support;

        let raw_env = env.get_raw();
        let raw_obj = read_stream.as_raw();

        let reader = stream_support::create_faiss_stream_reader(raw_env, raw_obj);
        let result = faiss_wrapper::load_index_with_stream_adc_params(
            reader.io_reader,
            env,
            &parameters,
        );

        stream_support::destroy_faiss_stream_reader(reader);

        result.map_err(KnnError::from)
    })
}

// ---------------------------------------------------------------------------
// isSharedIndexStateRequired / initSharedIndexState / setSharedIndexState
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_isSharedIndexStateRequired(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
) -> jboolean {
    jni_catch_panic(&mut env, 0, |_env| {
        let required = faiss_wrapper::is_shared_index_state_required(index_pointer);
        Ok(if required { 1u8 } else { 0u8 })
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_initSharedIndexState(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
) -> jlong {
    jni_catch_panic(&mut env, 0, |_env| {
        let result = faiss_wrapper::init_shared_index_state(index_pointer)?;
        Ok(result)
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_setSharedIndexState(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
    shared_index_state_pointer: jlong,
) {
    jni_catch_panic_void(&mut env, |_env| {
        faiss_wrapper::set_shared_index_state(index_pointer, shared_index_state_pointer)?;
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// queryIndex / queryIndexWithFilter / queryBinaryIndexWithFilter
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_queryIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
    query_vector: JFloatArray,
    k: jint,
    method_params: JObject,
    parent_ids: JIntArray,
) -> jni::sys::jobjectArray {
    jni_catch_panic(&mut env, ptr::null_mut(), |env| {
        let results = faiss_wrapper::query_index(
            env,
            index_pointer,
            &query_vector,
            k,
            &method_params,
            &parent_ids,
        )?;
        Ok(results.into_raw())
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_queryIndexWithFilter(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
    query_vector: JFloatArray,
    k: jint,
    method_params: JObject,
    filter_ids: JLongArray,
    filter_ids_type: jint,
    parent_ids: JIntArray,
) -> jni::sys::jobjectArray {
    // Extract raw handle before entering the panic-catching closure to avoid
    // lifetime issues with JNI wrapper types.
    let filter_ids_raw = filter_ids.as_raw();
    jni_catch_panic(&mut env, ptr::null_mut(), |env| {
        let filter_local: JLongArray = JLongArray::from_raw(filter_ids_raw);
        let filter_ptr = if filter_local.is_null() {
            ptr::null()
        } else {
            &filter_local as *const JLongArray
        };
        let results = faiss_wrapper::query_index_with_filter(
            env,
            index_pointer,
            &query_vector,
            k,
            &method_params,
            filter_ptr,
            filter_ids_type,
            &parent_ids,
        )?;
        Ok(results.into_raw())
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_queryBinaryIndexWithFilter(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
    query_vector: JByteArray,
    k: jint,
    method_params: JObject,
    filter_ids: JLongArray,
    filter_ids_type: jint,
    parent_ids: JIntArray,
) -> jni::sys::jobjectArray {
    let filter_ids_raw = filter_ids.as_raw();
    jni_catch_panic(&mut env, ptr::null_mut(), |env| {
        let filter_local: JLongArray = JLongArray::from_raw(filter_ids_raw);
        let filter_ptr = if filter_local.is_null() {
            ptr::null()
        } else {
            &filter_local as *const JLongArray
        };
        let results = faiss_wrapper::query_binary_index_with_filter(
            env,
            index_pointer,
            &query_vector,
            k,
            &method_params,
            filter_ptr,
            filter_ids_type,
            &parent_ids,
        )?;
        Ok(results.into_raw())
    })
}

// ---------------------------------------------------------------------------
// free / freeSharedIndexState
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_free(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
    is_binary_index: jboolean,
) {
    jni_catch_panic_void(&mut env, |_env| {
        faiss_wrapper::free(index_pointer, is_binary_index);
        Ok(())
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_freeSharedIndexState(
    mut env: JNIEnv,
    _cls: JClass,
    shared_index_state_pointer: jlong,
) {
    jni_catch_panic_void(&mut env, |_env| {
        faiss_wrapper::free_shared_index_state(shared_index_state_pointer);
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// initLibrary
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_initLibrary(
    mut env: JNIEnv,
    _cls: JClass,
) {
    jni_catch_panic_void(&mut env, |_env| {
        faiss_wrapper::init_library();
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// trainIndex / trainBinaryIndex / trainByteIndex
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_trainIndex(
    mut env: JNIEnv,
    _cls: JClass,
    parameters: JObject,
    dimension: jint,
    train_vectors_pointer: jlong,
) -> jni::sys::jbyteArray {
    jni_catch_panic(&mut env, ptr::null_mut(), |env| {
        let result =
            faiss_wrapper::train_index(env, &parameters, dimension, train_vectors_pointer)?;
        Ok(result.into_raw())
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_trainBinaryIndex(
    mut env: JNIEnv,
    _cls: JClass,
    parameters: JObject,
    dimension: jint,
    train_vectors_pointer: jlong,
) -> jni::sys::jbyteArray {
    jni_catch_panic(&mut env, ptr::null_mut(), |env| {
        let result = faiss_wrapper::train_binary_index(
            env,
            &parameters,
            dimension,
            train_vectors_pointer,
        )?;
        Ok(result.into_raw())
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_trainByteIndex(
    mut env: JNIEnv,
    _cls: JClass,
    parameters: JObject,
    dimension: jint,
    train_vectors_pointer: jlong,
) -> jni::sys::jbyteArray {
    jni_catch_panic(&mut env, ptr::null_mut(), |env| {
        let result = faiss_wrapper::train_byte_index(
            env,
            &parameters,
            dimension,
            train_vectors_pointer,
        )?;
        Ok(result.into_raw())
    })
}

// ---------------------------------------------------------------------------
// transferVectors
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_transferVectors(
    mut env: JNIEnv,
    _cls: JClass,
    vectors_pointer: jlong,
    vectors: JObjectArray,
) -> jlong {
    jni_catch_panic(&mut env, 0, |env| {
        let jni_util = crate::global_jni_util();

        let vect: *mut Vec<f32> = if vectors_pointer == 0 {
            Box::into_raw(Box::new(Vec::<f32>::new()))
        } else {
            vectors_pointer as *mut Vec<f32>
        };

        // Get the dimension from the 2D Java array
        let dim = jni_util
            .get_inner_dimension_of_2d_java_float_array(env, &vectors)
            .map_err(|e| KnnError::InvalidArgument(format!("{}", e)))?;

        // Convert 2D Java float array to a flat float vector and append
        let mut new_data = Vec::<f32>::new();
        jni_util
            .convert_2d_java_object_array_and_store_to_float_vector(
                env, &vectors, dim, &mut new_data,
            )
            .map_err(|e| KnnError::InvalidArgument(format!("{}", e)))?;

        if !new_data.is_empty() {
            // Insert at the beginning (matching C++ vect->insert(vect->begin(), ...))
            let existing = &*vect;
            let mut combined = Vec::with_capacity(new_data.len() + existing.len());
            combined.extend_from_slice(&new_data);
            combined.extend_from_slice(existing);
            *vect = combined;
        }

        Ok(vect as jlong)
    })
}

// ---------------------------------------------------------------------------
// rangeSearchIndex / rangeSearchIndexWithFilter
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_rangeSearchIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
    query_vector: JFloatArray,
    radius: jfloat,
    method_params: JObject,
    max_result_window: jint,
    parent_ids: JIntArray,
) -> jni::sys::jobjectArray {
    jni_catch_panic(&mut env, ptr::null_mut(), |env| {
        let results = faiss_wrapper::range_search(
            env,
            index_pointer,
            &query_vector,
            radius,
            &method_params,
            max_result_window,
            &parent_ids,
        )?;
        Ok(results.into_raw())
    })
}

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_rangeSearchIndexWithFilter(
    mut env: JNIEnv,
    _cls: JClass,
    index_pointer: jlong,
    query_vector: JFloatArray,
    radius: jfloat,
    method_params: JObject,
    max_result_window: jint,
    filter_ids: JLongArray,
    filter_ids_type: jint,
    parent_ids: JIntArray,
) -> jni::sys::jobjectArray {
    let filter_ids_raw = filter_ids.as_raw();
    jni_catch_panic(&mut env, ptr::null_mut(), |env| {
        let filter_local: JLongArray = JLongArray::from_raw(filter_ids_raw);
        let filter_ptr = if filter_local.is_null() {
            ptr::null()
        } else {
            &filter_local as *const JLongArray
        };
        let results = faiss_wrapper::range_search_with_filter(
            env,
            index_pointer,
            &query_vector,
            radius,
            &method_params,
            max_result_window,
            filter_ptr,
            filter_ids_type,
            &parent_ids,
        )?;
        Ok(results.into_raw())
    })
}

// ---------------------------------------------------------------------------
// setMergeInterruptCallback
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_setMergeInterruptCallback(
    mut env: JNIEnv,
    _cls: JClass,
) {
    jni_catch_panic_void(&mut env, |env| {
        // Get the raw JNIEnv pointer to pass to the C++ shim
        let raw_env = env.get_raw() as *mut std::ffi::c_void;
        unsafe {
            crate::ffi::knn_shim::knn_shim_set_merge_interrupt_callback(raw_env);
        }
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// initFaissSQIndex
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_initFaissSQIndex(
    mut env: JNIEnv,
    _cls: JClass,
    total_live_docs: jint,
    dim: jint,
    parameters: JObject,
    centroid_dp: jfloat,
    quantized_vec_bytes: jint,
) -> jlong {
    jni_catch_panic(&mut env, 0, |_env| {
        // SQ index initialization requires the FaissSQDistanceComputer C++ template,
        // which inherits from faiss::DistanceComputer. Delegated to C++ shim.
        let result = unsafe {
            crate::ffi::knn_shim::knn_shim_init_sq_index(
                total_live_docs,
                dim,
                centroid_dp,
                quantized_vec_bytes,
            )
        };
        if result == 0 {
            Err(KnnError::Internal("Failed to initialize SQ index".to_string()))
        } else {
            Ok(result)
        }
    })
}

// ---------------------------------------------------------------------------
// addDocsToSQIndex
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_addDocsToSQIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_memory_address: jlong,
    doc_ids: JIntArray,
    num_docs: jint,
    num_added: jint,
) {
    jni_catch_panic_void(&mut env, |_env| {
        unsafe {
            crate::ffi::knn_shim::knn_shim_sq_add_docs(
                index_memory_address,
                num_docs,
                num_added,
            );
        }
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// passSQVectorsWithCorrectionFactors
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_passSQVectorsWithCorrectionFactors(
    mut env: JNIEnv,
    _cls: JClass,
    index_memory_address: jlong,
    buffer: JByteArray,
    num_elements: jint,
) {
    jni_catch_panic_void(&mut env, |env| {
        let buf_len = env.get_array_length(&buffer)? as usize;
        let mut buf_data = vec![0i8; buf_len];
        env.get_byte_array_region(&buffer, 0, &mut buf_data)?;
        unsafe {
            crate::ffi::knn_shim::knn_shim_sq_pass_vectors(
                index_memory_address,
                buf_data.as_ptr() as *const u8,
                num_elements,
            );
        }
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// releaseFaissSQIndex
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_releaseFaissSQIndex(
    mut env: JNIEnv,
    _cls: JClass,
    index_memory_address: jlong,
) {
    jni_catch_panic_void(&mut env, |_env| {
        unsafe {
            crate::ffi::knn_shim::knn_shim_sq_release_index(index_memory_address);
        }
        Ok(())
    });
}
