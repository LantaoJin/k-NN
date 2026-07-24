// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Root module for the k-NN JNI Rust crate (`knn-jni-rs`).
//!
//! This crate provides Rust implementations of the JNI native methods for the
//! OpenSearch k-NN plugin, replacing the original C++ JNI layer while retaining
//! FFI calls into Faiss and NMSLIB for the actual vector search operations.
//!
//! # Architecture
//!
//! - `jni_util`: JNI utility helpers (type conversions, cached classes, exception handling)
//! - `commons`: Common vector storage and parameter extraction utilities
//! - `ffi`: Raw `extern "C"` bindings to Faiss and NMSLIB C APIs
//! - `faiss_index_service`: IndexService trait and implementations for Faiss indices
//! - `faiss_methods`: Thin wrappers around Faiss C API functions (index_factory, I/O)
//! - `faiss_util`: Faiss utility functions (IDGrouper, etc.)
//! - `faiss_service_jni`: JNI entry points for `FaissService` Java class
//! - `nmslib_wrapper`: NMSLIB operations (create/load/query/free)
//! - `nmslib_service_jni`: JNI entry points for `NmslibService` Java class
//! - `simd`: SIMD dispatch for similarity functions
//! - `simd_service_jni`: JNI entry points for `SimdVectorComputeService`

#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

// ========================== Sub-module declarations ==========================

// --- GATED: these modules have compilation errors and are gated until Phase E ---
// Gate = #[cfg(any())] — compiles to nothing, un-gate one at a time.
pub mod jni_util;
pub mod faiss_wrapper;
pub mod faiss_service_jni;
pub mod nmslib_wrapper;
pub mod nmslib_service_jni;
pub mod simd_service_jni;

// --- UNGATED: these modules compile or are close to compiling ---
pub mod commons;
pub mod ffi;
pub mod faiss_index_service;
pub mod faiss_methods;
pub mod faiss_util;
pub mod simd;
pub mod jni_commons;
pub mod stream_support;

// ========================== Re-exports for convenience ==========================

// GATED: re-exports depend on jni_util which is gated
// pub use jni_util::{JNIUtil, JniUtilError, JniReleaseGuard};

// ========================== Crate-wide error type ==========================

use std::fmt;

/// Unified error type used across the entire knn-jni-rs crate.
///
/// This captures all possible failure modes when interacting with JNI,
/// native libraries (Faiss/NMSLIB), and internal logic.
#[derive(Debug)]
pub enum KnnError {
    /// An error originating from the JNI layer (class not found, method call failed, etc.)
    Jni(jni::errors::Error),

    /// An error from a native library (Faiss or NMSLIB) communicated via FFI.
    NativeLibrary(String),

    /// A null pointer was encountered where a valid pointer was expected.
    NullPointer(String),

    /// Invalid argument provided by the caller (maps to IllegalArgumentException in Java).
    InvalidArgument(String),

    /// An I/O error (file not found, read failure, etc.).
    Io(std::io::Error),

    /// The operation was aborted (e.g., index build abort due to merge cancellation).
    Aborted(String),

    /// A catch-all for other unexpected errors.
    Internal(String),
}

impl fmt::Display for KnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KnnError::Jni(e) => write!(f, "JNI error: {}", e),
            KnnError::NativeLibrary(msg) => write!(f, "Native library error: {}", msg),
            KnnError::NullPointer(msg) => write!(f, "Null pointer: {}", msg),
            KnnError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            KnnError::Io(e) => write!(f, "I/O error: {}", e),
            KnnError::Aborted(msg) => write!(f, "Operation aborted: {}", msg),
            KnnError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for KnnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KnnError::Jni(e) => Some(e),
            KnnError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<jni::errors::Error> for KnnError {
    fn from(e: jni::errors::Error) -> Self {
        KnnError::Jni(e)
    }
}

impl From<std::io::Error> for KnnError {
    fn from(e: std::io::Error) -> Self {
        KnnError::Io(e)
    }
}

/// Crate-wide Result type alias using [`KnnError`].
pub type KnnResult<T> = std::result::Result<T, KnnError>;

// ========================== Global state ==========================

use once_cell::sync::OnceCell;
use jni::JavaVM;
use jni::sys::{jint, JNI_ERR, JNI_VERSION_1_1};
use std::ffi::c_void;

const KNN_JNI_VERSION: jint = JNI_VERSION_1_1;

static GLOBAL_JNI_UTIL: OnceCell<jni_util::JNIUtil> = OnceCell::new();

/// Returns a reference to the global [`jni_util::JNIUtil`] instance.
///
/// # Panics
/// Panics if called before `JNI_OnLoad` has completed initialization.
pub fn global_jni_util() -> &'static jni_util::JNIUtil {
    GLOBAL_JNI_UTIL
        .get()
        .expect("JNI_OnLoad has not been called; GLOBAL_JNI_UTIL is uninitialized")
}

// ========================== JNI_OnLoad / JNI_OnUnload ==========================

#[no_mangle]
pub unsafe extern "system" fn JNI_OnLoad(
    vm: *mut jni::sys::JavaVM,
    _reserved: *mut c_void,
) -> jint {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Create two JavaVM handles from the raw pointer:
        // one to get an env, one to pass ownership to JNIUtil.
        let java_vm_for_env = match unsafe { JavaVM::from_raw(vm) } {
            Ok(jvm) => jvm,
            Err(_) => return JNI_ERR,
        };

        let mut env = match java_vm_for_env.get_env() {
            Ok(env) => env,
            Err(_) => return JNI_ERR,
        };

        // Create a second handle to pass ownership to JNIUtil.
        // This is safe: JavaVM::from_raw doesn't take ownership of the raw pointer.
        let java_vm_for_util = match unsafe { JavaVM::from_raw(vm) } {
            Ok(jvm) => jvm,
            Err(_) => return JNI_ERR,
        };

        let mut jni_util_instance = jni_util::JNIUtil::new();
        if let Err(_e) = jni_util_instance.initialize(&mut env, java_vm_for_util) {
            return JNI_ERR;
        }

        let _ = GLOBAL_JNI_UTIL.set(jni_util_instance);
        KNN_JNI_VERSION
    }));

    match result {
        Ok(version) => version,
        Err(_) => JNI_ERR,
    }
}

#[no_mangle]
pub unsafe extern "system" fn JNI_OnUnload(
    _vm: *mut jni::sys::JavaVM,
    _reserved: *mut c_void,
) {
    // Global refs released by the JVM on library unload.
}

// ========================== Panic-catching boundary ==========================

/// Catches panics at the JNI boundary and translates them into Java exceptions.
///
/// Every `#[no_mangle] pub extern "system" fn` JNI entry point MUST call this
/// macro/function to prevent Rust panics from unwinding across the FFI boundary
/// (which is undefined behavior).
///
/// # Usage
///
/// ```rust,ignore
/// #[no_mangle]
/// pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_someMethod(
///     mut env: JNIEnv,
///     _class: JClass,
///     arg: jlong,
/// ) -> jlong {
///     jni_catch_panic(&mut env, 0, || {
///         // ... actual implementation ...
///         Ok(42)
///     })
/// }
/// ```
///
/// If the closure panics, the panic is caught, a Java `RuntimeException` is thrown,
/// and `default_value` is returned to the JVM.
///
/// If the closure returns `Err(KnnError)`, the error is translated into the
/// appropriate Java exception and `default_value` is returned.
pub fn jni_catch_panic<F, T>(
    env: &mut jni::JNIEnv,
    default_value: T,
    f: F,
) -> T
where
    F: FnOnce(&mut jni::JNIEnv) -> KnnResult<T> + std::panic::UnwindSafe,
    T: Copy,
{
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        f(env)
    }));

    match result {
        Ok(Ok(value)) => value,
        Ok(Err(knn_err)) => {
            throw_knn_error(env, &knn_err);
            default_value
        }
        Err(panic_payload) => {
            // Extract panic message if possible
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                format!("Native code panicked: {}", s)
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                format!("Native code panicked: {}", s)
            } else {
                "Native code panicked (unknown payload)".to_string()
            };
            let _ = env.throw_new("java/lang/RuntimeException", &msg);
            default_value
        }
    }
}

/// Variant of [`jni_catch_panic`] for JNI methods that return void.
///
/// # Usage
///
/// ```rust,ignore
/// #[no_mangle]
/// pub unsafe extern "system" fn Java_org_opensearch_knn_jni_FaissService_freeIndex(
///     mut env: JNIEnv,
///     _class: JClass,
///     ptr: jlong,
/// ) {
///     jni_catch_panic_void(&mut env, || {
///         // ... actual implementation ...
///         Ok(())
///     })
/// }
/// ```
pub fn jni_catch_panic_void<F>(
    env: &mut jni::JNIEnv,
    f: F,
)
where
    F: FnOnce(&mut jni::JNIEnv) -> KnnResult<()> + std::panic::UnwindSafe,
{
    jni_catch_panic(env, (), |env| f(env));
}

/// Translates a [`KnnError`] into the appropriate Java exception.
fn throw_knn_error(env: &mut jni::JNIEnv, error: &KnnError) {
    let (exception_class, message) = match error {
        KnnError::InvalidArgument(msg) => (
            "java/lang/IllegalArgumentException",
            msg.clone(),
        ),
        KnnError::NullPointer(msg) => (
            "java/lang/NullPointerException",
            msg.clone(),
        ),
        KnnError::Io(e) => (
            "java/io/IOException",
            format!("{}", e),
        ),
        KnnError::Aborted(msg) => (
            "org/opensearch/knn/index/codec/nativeindex/IndexBuildAbortedException",
            msg.clone(),
        ),
        KnnError::Jni(e) => (
            "java/lang/RuntimeException",
            format!("JNI error: {}", e),
        ),
        KnnError::NativeLibrary(msg) => (
            "java/lang/Exception",
            msg.clone(),
        ),
        KnnError::Internal(msg) => (
            "java/lang/Exception",
            msg.clone(),
        ),
    };

    // Only throw if there is no pending exception already
    if !env.exception_check().unwrap_or(true) {
        let _ = env.throw_new(exception_class, &message);
    }
}

// ========================== Convenience macro ==========================

/// Macro that wraps a JNI entry point body in the panic-catching boundary.
///
/// This is syntactic sugar over [`jni_catch_panic`] for the common case.
///
/// # Examples
///
/// ```rust,ignore
/// // For a function returning jlong:
/// knn_jni_entry!(env, 0_i64, |env| {
///     let result = do_something(env)?;
///     Ok(result as jlong)
/// })
///
/// // For a void function:
/// knn_jni_entry_void!(env, |env| {
///     do_something(env)?;
///     Ok(())
/// })
/// ```
#[macro_export]
macro_rules! knn_jni_entry {
    ($env:expr, $default:expr, $body:expr) => {
        $crate::jni_catch_panic($env, $default, $body)
    };
}

/// Macro variant for void JNI entry points.
#[macro_export]
macro_rules! knn_jni_entry_void {
    ($env:expr, $body:expr) => {
        $crate::jni_catch_panic_void($env, $body)
    };
}
