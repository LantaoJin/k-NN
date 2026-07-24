// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! JNI entry points for JNICommons: storeVectorData, storeBinaryVectorData,
//! storeByteVectorData, freeVectorData, freeBinaryVectorData, freeByteVectorData.
//!
//! Ported from: jni/src/org_opensearch_knn_jni_JNICommons.cpp

use jni::objects::{JClass, JObjectArray};
use jni::sys::{jboolean, jlong, jint, JavaVM, JNI_VERSION_1_1};
use jni::JNIEnv;
use std::os::raw::c_void;
use std::panic;

use crate::commons;

// ---------------------------------------------------------------------------
// Panic-catching boundary helper
// ---------------------------------------------------------------------------

/// Wraps a closure in a panic-catching boundary.
/// If the closure panics, throws a Java RuntimeException and returns the fallback value.
fn catch_panic_and_throw<F, R>(env: &mut JNIEnv, fallback: R, f: F) -> R
where
    F: FnOnce(&mut JNIEnv) -> R + panic::UnwindSafe,
{
    // We need a raw env pointer to reconstruct JNIEnv inside the panic boundary
    let raw_env = env.get_raw();

    match panic::catch_unwind(move || {
        // Safety: raw_env is valid for the duration of the JNI call.
        let mut env = unsafe { JNIEnv::from_raw(raw_env).expect("Failed to obtain JNIEnv from raw") };
        f(&mut env)
    }) {
        Ok(result) => result,
        Err(panic_info) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic in Rust JNI code".to_string()
            };

            // Attempt to throw a Java exception. If this fails, there is nothing more we can do.
            let _ = env.throw_new("java/lang/RuntimeException", &msg);
            fallback
        }
    }
}

// JNI_OnLoad / JNI_OnUnload are defined in lib.rs (the crate root)

// ---------------------------------------------------------------------------
// storeVectorData (float[][])
// ---------------------------------------------------------------------------

/// JNI entry point: `Java_org_opensearch_knn_jni_JNICommons_storeVectorData`
///
/// Stores float vector data into native memory. If `memory_address` is 0, allocates a new
/// vector with the given initial capacity. Otherwise appends to (or overwrites) the existing
/// vector at that address.
///
/// # Safety
/// Called from JVM via JNI. All pointer arguments are managed by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_JNICommons_storeVectorData(
    mut env: JNIEnv,
    _cls: JClass,
    memory_address: jlong,
    data: JObjectArray,
    initial_capacity: jlong,
    append: jboolean,
) -> jlong {
    let mem_addr = memory_address;
    catch_panic_and_throw(&mut env, mem_addr, move |env| unsafe {
        commons::store_vector_data(env, memory_address, &data, initial_capacity, append)
    })
}

// ---------------------------------------------------------------------------
// storeBinaryVectorData (byte[][] -> u8)
// ---------------------------------------------------------------------------

/// JNI entry point: `Java_org_opensearch_knn_jni_JNICommons_storeBinaryVectorData`
///
/// Stores binary (uint8) vector data into native memory.
///
/// # Safety
/// Called from JVM via JNI. All pointer arguments are managed by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_JNICommons_storeBinaryVectorData(
    mut env: JNIEnv,
    _cls: JClass,
    memory_address: jlong,
    data: JObjectArray,
    initial_capacity: jlong,
    append: jboolean,
) -> jlong {
    let mem_addr = memory_address;
    catch_panic_and_throw(&mut env, mem_addr, move |env| unsafe {
        commons::store_binary_vector_data(env, memory_address, &data, initial_capacity, append)
    })
}

// ---------------------------------------------------------------------------
// storeByteVectorData (byte[][] -> i8)
// ---------------------------------------------------------------------------

/// JNI entry point: `Java_org_opensearch_knn_jni_JNICommons_storeByteVectorData`
///
/// Stores signed byte (int8) vector data into native memory.
///
/// # Safety
/// Called from JVM via JNI. All pointer arguments are managed by the JVM.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_JNICommons_storeByteVectorData(
    mut env: JNIEnv,
    _cls: JClass,
    memory_address: jlong,
    data: JObjectArray,
    initial_capacity: jlong,
    append: jboolean,
) -> jlong {
    let mem_addr = memory_address;
    catch_panic_and_throw(&mut env, mem_addr, move |env| unsafe {
        commons::store_byte_vector_data(env, memory_address, &data, initial_capacity, append)
    })
}

// ---------------------------------------------------------------------------
// freeVectorData (float)
// ---------------------------------------------------------------------------

/// JNI entry point: `Java_org_opensearch_knn_jni_JNICommons_freeVectorData`
///
/// Frees the native memory allocated for float vector data at the given address.
///
/// # Safety
/// Called from JVM via JNI. `memory_address` must be 0 or a valid pointer previously
/// returned by `storeVectorData`.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_JNICommons_freeVectorData(
    mut env: JNIEnv,
    _cls: JClass,
    memory_address: jlong,
) {
    catch_panic_and_throw(&mut env, (), move |_env| unsafe {
        commons::free_vector_data(memory_address);
    })
}

// ---------------------------------------------------------------------------
// freeBinaryVectorData (u8)
// ---------------------------------------------------------------------------

/// JNI entry point: `Java_org_opensearch_knn_jni_JNICommons_freeBinaryVectorData`
///
/// Frees the native memory allocated for binary (uint8) vector data at the given address.
///
/// # Safety
/// Called from JVM via JNI. `memory_address` must be 0 or a valid pointer previously
/// returned by `storeBinaryVectorData`.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_JNICommons_freeBinaryVectorData(
    mut env: JNIEnv,
    _cls: JClass,
    memory_address: jlong,
) {
    catch_panic_and_throw(&mut env, (), move |_env| unsafe {
        commons::free_binary_vector_data(memory_address);
    })
}

// ---------------------------------------------------------------------------
// freeByteVectorData (i8)
// ---------------------------------------------------------------------------

/// JNI entry point: `Java_org_opensearch_knn_jni_JNICommons_freeByteVectorData`
///
/// Frees the native memory allocated for signed byte (int8) vector data at the given address.
///
/// # Safety
/// Called from JVM via JNI. `memory_address` must be 0 or a valid pointer previously
/// returned by `storeByteVectorData`.
#[no_mangle]
pub unsafe extern "system" fn Java_org_opensearch_knn_jni_JNICommons_freeByteVectorData(
    mut env: JNIEnv,
    _cls: JClass,
    memory_address: jlong,
) {
    catch_panic_and_throw(&mut env, (), move |_env| unsafe {
        commons::free_byte_vector_data(memory_address);
    })
}
