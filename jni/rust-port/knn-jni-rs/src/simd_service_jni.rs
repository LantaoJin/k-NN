//! JNI entry points for SIMD vector compute service.
//!
//! Rust port of `org_opensearch_knn_jni_SimdVectorComputeService.cpp`.
//! Uses the `jni` crate (0.21+) for JNI interop.

use jni::objects::{JByteArray, JFloatArray, JIntArray, JLongArray};
use jni::sys::{jbyteArray, jclass, jfloat, jfloatArray, jint, jintArray, jlongArray, JavaVM, JNI_ERR, JNI_VERSION_1_1};
use jni::JNIEnv;
use std::panic;
use std::ptr;

// ---------------------------------------------------------------------------
// FFI declarations for the C++ SIMD similarity function layer.
// These will be moved to a dedicated ffi module in a later phase.
// ---------------------------------------------------------------------------
mod ffi {
    #[allow(unused_imports)]
    use std::os::raw::{c_int, c_void};

    /// Opaque handle representing `SimdVectorSearchContext` on the C++ side.
    #[repr(C)]
    pub struct SimdVectorSearchContext {
        _opaque: [u8; 0],
    }

    extern "C" {
        /// Calls `SimilarityFunction::getSearchContext()` — returns thread-local context.
        pub fn knn_simd_get_search_context() -> *mut SimdVectorSearchContext;

        /// Calls `SimilarityFunction::saveSearchContext(...)`.
        pub fn knn_simd_save_search_context(
            query_ptr: *const u8,
            query_byte_size: i32,
            dimension: i32,
            mmap_address_and_size: *const i64,
            num_address_and_size: i32,
            native_function_type_ord: i32,
        ) -> *mut SimdVectorSearchContext;

        /// Calls `srchContext->similarityFunction->calculateSimilarityInBulk(...)`.
        pub fn knn_simd_calculate_similarity_in_bulk(
            ctx: *mut SimdVectorSearchContext,
            internal_vector_ids: *const i32,
            scores: *mut f32,
            num_vectors: i32,
        );

        /// Calls `srchContext->similarityFunction->calculateSimilarity(...)`.
        pub fn knn_simd_calculate_similarity(
            ctx: *mut SimdVectorSearchContext,
            internal_vector_id: i32,
        ) -> f32;

        /// Resizes and writes correction factors into `ctx->tmpBuffer`.
        /// correctionData layout: [lowerInterval, upperInterval, additionalCorrection, quantizedComponentSum(as f32 bits), centroidDp]
        pub fn knn_simd_set_sq_correction_factors(
            ctx: *mut SimdVectorSearchContext,
            lower_interval: f32,
            upper_interval: f32,
            additional_correction: f32,
            quantized_component_sum: i32,
            centroid_dp: f32,
        );
    }
}

// ---------------------------------------------------------------------------
// Helper: throw a Java RuntimeException from a Rust error message.
// ---------------------------------------------------------------------------
fn throw_java_runtime_exception(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
}

/// Macro that wraps a JNI function body in a panic-catching boundary.
/// On panic it throws a Java RuntimeException and returns the specified default value.
macro_rules! jni_entry {
    ($env:expr, $default:expr, $body:block) => {{
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| $body));
        match result {
            Ok(val) => val,
            Err(_) => {
                throw_java_runtime_exception($env, "Rust panic in JNI call");
                $default
            }
        }
    }};
}

// JNI_OnLoad / JNI_OnUnload are defined in lib.rs (the crate root).

// ---------------------------------------------------------------------------
// scoreSimilarityInBulk
// ---------------------------------------------------------------------------

/// JNI entry point: `SimdVectorComputeService.scoreSimilarityInBulk`
///
/// Calculates similarity scores in bulk for the given vector IDs and returns the max score.
#[no_mangle]
pub extern "system" fn Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarityInBulk(
    mut env: JNIEnv,
    _clazz: jclass,
    internal_vector_ids: jintArray,
    jscores: jfloatArray,
    num_vectors: jint,
) -> jfloat {
    jni_entry!(&mut env, 0.0f32, {
        if num_vectors <= 0 {
            return f32::MIN;
        }

        unsafe {
            // Get search context from thread-local storage
            let ctx = ffi::knn_simd_get_search_context();
            if ctx.is_null() {
                throw_java_runtime_exception(
                    &mut env,
                    "No search context has been initialized, SimdVectorSearchContext* was empty.",
                );
                return 0.0f32;
            }

            // Copy vector IDs into a local buffer (read-only)
            let int_array = JIntArray::from_raw(internal_vector_ids);
            let ids_len = env.get_array_length(&int_array).expect("Failed to get vector IDs array length") as usize;
            let mut ids_buf = vec![0i32; ids_len];
            env.get_int_array_region(&int_array, 0, &mut ids_buf).expect("Failed to get vector IDs region");

            // Copy scores into a local buffer (read-write)
            let float_array = JFloatArray::from_raw(jscores);
            let scores_len = env.get_array_length(&float_array).expect("Failed to get scores array length") as usize;
            let mut scores_buf = vec![0.0f32; scores_len];
            env.get_float_array_region(&float_array, 0, &mut scores_buf).expect("Failed to get scores region");

            // Bulk similarity calculation
            ffi::knn_simd_calculate_similarity_in_bulk(
                ctx,
                ids_buf.as_ptr(),
                scores_buf.as_mut_ptr(),
                num_vectors,
            );

            // Find max score
            let max_score = scores_buf[..num_vectors as usize]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            // Write scores back to the Java array
            env.set_float_array_region(&float_array, 0, &scores_buf).expect("Failed to set scores region");

            max_score
        }
    })
}

// ---------------------------------------------------------------------------
// saveSearchContext
// ---------------------------------------------------------------------------

/// JNI entry point: `SimdVectorComputeService.saveSearchContext`
///
/// Saves the query vector and mmap metadata into thread-local search context.
#[no_mangle]
pub extern "system" fn Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSearchContext(
    mut env: JNIEnv,
    _clazz: jclass,
    query: jfloatArray,
    address_and_size: jlongArray,
    native_function_type_ord: jint,
) {
    jni_entry!(&mut env, (), {
        unsafe {
            // Copy query float array into a local buffer
            let query_arr = JFloatArray::from_raw(query);
            let query_vec_size = env
                .get_array_length(&query_arr)
                .expect("Failed to get query array length");
            let mut query_buf = vec![0.0f32; query_vec_size as usize];
            env.get_float_array_region(&query_arr, 0, &mut query_buf)
                .expect("Failed to get query array region");

            // Copy mmap address_and_size long array into a local buffer
            let address_arr = JLongArray::from_raw(address_and_size);
            let mmap_length = env
                .get_array_length(&address_arr)
                .expect("Failed to get addressAndSize array length");
            let mut mmap_buf = vec![0i64; mmap_length as usize];
            env.get_long_array_region(&address_arr, 0, &mut mmap_buf)
                .expect("Failed to get addressAndSize array region");

            let query_ptr = query_buf.as_ptr() as *const u8;
            let query_byte_size = (std::mem::size_of::<f32>() as i32) * query_vec_size;
            let mmap_ptr = mmap_buf.as_ptr() as *const i64;

            ffi::knn_simd_save_search_context(
                query_ptr,
                query_byte_size,
                query_vec_size,
                mmap_ptr,
                mmap_length,
                native_function_type_ord,
            );
        }
    })
}

// ---------------------------------------------------------------------------
// scoreSimilarity
// ---------------------------------------------------------------------------

/// JNI entry point: `SimdVectorComputeService.scoreSimilarity`
///
/// Calculates similarity for a single vector ID using the current thread-local context.
#[no_mangle]
pub extern "system" fn Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarity(
    mut env: JNIEnv,
    _clazz: jclass,
    internal_vector_id: jint,
) -> jfloat {
    jni_entry!(&mut env, 0.0f32, {
        unsafe {
            let ctx = ffi::knn_simd_get_search_context();
            if ctx.is_null() {
                throw_java_runtime_exception(
                    &mut env,
                    "No search context has been initialized.",
                );
                return 0.0f32;
            }
            ffi::knn_simd_calculate_similarity(ctx, internal_vector_id)
        }
    })
}

// ---------------------------------------------------------------------------
// saveSQSearchContext
// ---------------------------------------------------------------------------

/// JNI entry point: `SimdVectorComputeService.saveSQSearchContext`
///
/// Saves quantized query and scalar quantization correction factors into the
/// thread-local search context.
#[no_mangle]
pub extern "system" fn Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSQSearchContext(
    mut env: JNIEnv,
    _clazz: jclass,
    quantized_query: jbyteArray,
    lower_interval: jfloat,
    upper_interval: jfloat,
    additional_correction: jfloat,
    quantized_component_sum: jint,
    address_and_size: jlongArray,
    function_type_ord: jint,
    dimension: jint,
    centroid_dp: jfloat,
) {
    jni_entry!(&mut env, (), {
        unsafe {
            // Copy quantized query byte array into a local buffer
            let query_arr = JByteArray::from_raw(quantized_query);
            let query_byte_size = env
                .get_array_length(&query_arr)
                .expect("Failed to get quantized query array length");
            let mut query_buf = vec![0i8; query_byte_size as usize];
            env.get_byte_array_region(&query_arr, 0, &mut query_buf)
                .expect("Failed to get quantized query array region");

            // Copy mmap address_and_size long array into a local buffer
            let address_arr = JLongArray::from_raw(address_and_size);
            let mmap_length = env
                .get_array_length(&address_arr)
                .expect("Failed to get addressAndSize array length");
            let mut mmap_buf = vec![0i64; mmap_length as usize];
            env.get_long_array_region(&address_arr, 0, &mut mmap_buf)
                .expect("Failed to get addressAndSize array region");

            let query_ptr = query_buf.as_ptr() as *const u8;
            let mmap_ptr = mmap_buf.as_ptr() as *const i64;

            // Save the search context (this resets tmpBuffer on the C++ side)
            let ctx = ffi::knn_simd_save_search_context(
                query_ptr,
                query_byte_size,
                dimension,
                mmap_ptr,
                mmap_length,
                function_type_ord,
            );

            // Now store SQ correction factors into ctx->tmpBuffer via FFI
            if !ctx.is_null() {
                ffi::knn_simd_set_sq_correction_factors(
                    ctx,
                    lower_interval,
                    upper_interval,
                    additional_correction,
                    quantized_component_sum,
                    centroid_dp,
                );
            }
        }
    })
}
