// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Rust port of knn_jni::faiss_wrapper::FaissMethods
//!
//! This module provides thin wrappers around Faiss C API functions
//! (index_factory, index_binary_factory, IndexIDMap construction, and index I/O).
//! The original C++ class exists primarily to allow mocking in tests;
//! in Rust we model this as a trait so the same testability is preserved.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

// ---------------------------------------------------------------------------
// Opaque Faiss types (pointers only -- we never dereference these in Rust)
// ---------------------------------------------------------------------------

/// Opaque handle to a faiss::Index.
#[repr(C)]
pub struct FaissIndex {
    _opaque: [u8; 0],
}

/// Opaque handle to a faiss::IndexBinary.
#[repr(C)]
pub struct FaissIndexBinary {
    _opaque: [u8; 0],
}

/// Opaque handle to a faiss::IndexIDMap (wraps faiss::Index).
#[repr(C)]
pub struct FaissIndexIDMap {
    _opaque: [u8; 0],
}

/// Opaque handle to a faiss::IndexBinaryIDMap (wraps faiss::IndexBinary).
#[repr(C)]
pub struct FaissIndexBinaryIDMap {
    _opaque: [u8; 0],
}

/// Opaque handle to a faiss::IOWriter.
#[repr(C)]
pub struct FaissIOWriter {
    _opaque: [u8; 0],
}

/// Faiss MetricType enum representation.
/// See faiss/MetricType.h
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
    MetricJaccard = 23,
}

/// Flag used with write_index_binary to skip storage.
pub const IO_FLAG_SKIP_STORAGE: c_int = 1;

// ---------------------------------------------------------------------------
// FFI declarations -- these call into the Faiss C API (libfaiss_c)
// ---------------------------------------------------------------------------

mod ffi {
    use super::*;

    extern "C" {
        /// Creates an index from a factory description string.
        /// Returns 0 on success, nonzero on error.
        /// The created index is written to *p_index.
        pub fn faiss_index_factory(
            p_index: *mut *mut FaissIndex,
            d: c_int,
            description: *const c_char,
            metric: FaissMetricType,
        ) -> c_int;

        /// Creates a binary index from a factory description string.
        /// Returns 0 on success, nonzero on error.
        pub fn faiss_index_binary_factory(
            p_index: *mut *mut FaissIndexBinary,
            d: c_int,
            description: *const c_char,
        ) -> c_int;

        /// Wraps an existing index with an IndexIDMap.
        /// Returns 0 on success, nonzero on error.
        /// Note: The C API function name may differ; adjust as needed.
        pub fn faiss_IndexIDMap_new(
            p_out: *mut *mut FaissIndexIDMap,
            index: *mut FaissIndex,
        ) -> c_int;

        /// Wraps an existing binary index with an IndexBinaryIDMap.
        /// Returns 0 on success, nonzero on error.
        pub fn faiss_IndexBinaryIDMap_new(
            p_out: *mut *mut FaissIndexBinaryIDMap,
            index: *mut FaissIndexBinary,
        ) -> c_int;

        /// Writes an index to an IOWriter.
        /// Returns 0 on success, nonzero on error.
        pub fn faiss_write_index_fname(
            idx: *const FaissIndex,
            writer: *mut FaissIOWriter,
        ) -> c_int;

        /// Writes a binary index to an IOWriter.
        /// flags can include IO_FLAG_SKIP_STORAGE.
        /// Returns 0 on success, nonzero on error.
        pub fn faiss_write_index_binary_fname(
            idx: *const FaissIndexBinary,
            writer: *mut FaissIOWriter,
            flags: c_int,
        ) -> c_int;
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error returned by FaissMethods operations.
#[derive(Debug, thiserror::Error)]
pub enum FaissError {
    #[error("Faiss index_factory failed (code={0})")]
    IndexFactoryFailed(c_int),

    #[error("Faiss index_binary_factory failed (code={0})")]
    IndexBinaryFactoryFailed(c_int),

    #[error("Faiss IndexIDMap creation failed (code={0})")]
    IndexIdMapFailed(c_int),

    #[error("Faiss IndexBinaryIDMap creation failed (code={0})")]
    IndexBinaryIdMapFailed(c_int),

    #[error("Faiss write_index failed (code={0})")]
    WriteIndexFailed(c_int),

    #[error("Faiss write_index_binary failed (code={0})")]
    WriteIndexBinaryFailed(c_int),
}

pub type Result<T> = std::result::Result<T, FaissError>;

// ---------------------------------------------------------------------------
// Trait (mirrors the virtual interface of the C++ class for mockability)
// ---------------------------------------------------------------------------

/// Trait mirroring knn_jni::faiss_wrapper::FaissMethods.
///
/// All pointer arguments are raw pointers because they refer to opaque Faiss
/// objects managed by C/C++ code. Callers must ensure pointer validity.
pub trait FaissMethods {
    /// Calls faiss::index_factory and returns a raw pointer to the new index.
    ///
    /// # Safety
    /// `description` must be a valid null-terminated C string pointer.
    unsafe fn index_factory(
        &self,
        d: c_int,
        description: *const c_char,
        metric: FaissMetricType,
    ) -> Result<*mut FaissIndex>;

    /// Calls faiss::index_binary_factory and returns a raw pointer to the new binary index.
    ///
    /// # Safety
    /// `description` must be a valid null-terminated C string pointer.
    unsafe fn index_binary_factory(
        &self,
        d: c_int,
        description: *const c_char,
    ) -> Result<*mut FaissIndexBinary>;

    /// Wraps an existing index with an IndexIDMap.
    ///
    /// # Safety
    /// `index` must be a valid pointer to a Faiss index.
    unsafe fn index_id_map(&self, index: *mut FaissIndex) -> Result<*mut FaissIndexIDMap>;

    /// Wraps an existing binary index with an IndexBinaryIDMap.
    ///
    /// # Safety
    /// `index` must be a valid pointer to a Faiss binary index.
    unsafe fn index_binary_id_map(
        &self,
        index: *mut FaissIndexBinary,
    ) -> Result<*mut FaissIndexBinaryIDMap>;

    /// Writes an index via the given IOWriter.
    ///
    /// # Safety
    /// Both `idx` and `writer` must be valid pointers.
    unsafe fn write_index(
        &self,
        idx: *const FaissIndex,
        writer: *mut FaissIOWriter,
    ) -> Result<()>;

    /// Writes a binary index via the given IOWriter.
    /// If `skip_flat` is true, the IO_FLAG_SKIP_STORAGE flag is set.
    ///
    /// # Safety
    /// Both `idx` and `writer` must be valid pointers.
    unsafe fn write_index_binary(
        &self,
        idx: *const FaissIndexBinary,
        writer: *mut FaissIOWriter,
        skip_flat: bool,
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Default (production) implementation
// ---------------------------------------------------------------------------

/// Production implementation that delegates directly to the Faiss C API.
pub struct FaissMethodsImpl;

impl FaissMethodsImpl {
    pub fn new() -> Self {
        FaissMethodsImpl
    }
}

impl Default for FaissMethodsImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl FaissMethods for FaissMethodsImpl {
    unsafe fn index_factory(
        &self,
        d: c_int,
        description: *const c_char,
        metric: FaissMetricType,
    ) -> Result<*mut FaissIndex> {
        let mut index: *mut FaissIndex = std::ptr::null_mut();
        let rc = ffi::faiss_index_factory(&mut index, d, description, metric);
        if rc != 0 {
            return Err(FaissError::IndexFactoryFailed(rc));
        }
        Ok(index)
    }

    unsafe fn index_binary_factory(
        &self,
        d: c_int,
        description: *const c_char,
    ) -> Result<*mut FaissIndexBinary> {
        let mut index: *mut FaissIndexBinary = std::ptr::null_mut();
        let rc = ffi::faiss_index_binary_factory(&mut index, d, description);
        if rc != 0 {
            return Err(FaissError::IndexBinaryFactoryFailed(rc));
        }
        Ok(index)
    }

    unsafe fn index_id_map(&self, index: *mut FaissIndex) -> Result<*mut FaissIndexIDMap> {
        let mut id_map: *mut FaissIndexIDMap = std::ptr::null_mut();
        let rc = ffi::faiss_IndexIDMap_new(&mut id_map, index);
        if rc != 0 {
            return Err(FaissError::IndexIdMapFailed(rc));
        }
        Ok(id_map)
    }

    unsafe fn index_binary_id_map(
        &self,
        index: *mut FaissIndexBinary,
    ) -> Result<*mut FaissIndexBinaryIDMap> {
        let mut id_map: *mut FaissIndexBinaryIDMap = std::ptr::null_mut();
        let rc = ffi::faiss_IndexBinaryIDMap_new(&mut id_map, index);
        if rc != 0 {
            return Err(FaissError::IndexBinaryIdMapFailed(rc));
        }
        Ok(id_map)
    }

    unsafe fn write_index(
        &self,
        idx: *const FaissIndex,
        writer: *mut FaissIOWriter,
    ) -> Result<()> {
        let rc = ffi::faiss_write_index_fname(idx, writer);
        if rc != 0 {
            return Err(FaissError::WriteIndexFailed(rc));
        }
        Ok(())
    }

    unsafe fn write_index_binary(
        &self,
        idx: *const FaissIndexBinary,
        writer: *mut FaissIOWriter,
        skip_flat: bool,
    ) -> Result<()> {
        let flags = if skip_flat { IO_FLAG_SKIP_STORAGE } else { 0 };
        let rc = ffi::faiss_write_index_binary_fname(idx, writer, flags);
        if rc != 0 {
            return Err(FaissError::WriteIndexBinaryFailed(rc));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers (match the C++ free-standing style)
// ---------------------------------------------------------------------------

/// Helper: Create a Faiss index using the factory with a Rust string description.
///
/// # Safety
/// Caller must ensure the returned pointer is eventually freed via Faiss.
pub unsafe fn index_factory(
    d: i32,
    description: &CStr,
    metric: FaissMetricType,
) -> Result<*mut FaissIndex> {
    let methods = FaissMethodsImpl::new();
    methods.index_factory(d as c_int, description.as_ptr(), metric)
}

/// Helper: Create a Faiss binary index using the factory with a Rust string description.
///
/// # Safety
/// Caller must ensure the returned pointer is eventually freed via Faiss.
pub unsafe fn index_binary_factory(
    d: i32,
    description: &CStr,
) -> Result<*mut FaissIndexBinary> {
    let methods = FaissMethodsImpl::new();
    methods.index_binary_factory(d as c_int, description.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// A mock implementation for testing.
    struct MockFaissMethods {
        pub factory_result: *mut FaissIndex,
        pub binary_factory_result: *mut FaissIndexBinary,
    }

    impl FaissMethods for MockFaissMethods {
        unsafe fn index_factory(
            &self,
            _d: c_int,
            _description: *const c_char,
            _metric: FaissMetricType,
        ) -> Result<*mut FaissIndex> {
            Ok(self.factory_result)
        }

        unsafe fn index_binary_factory(
            &self,
            _d: c_int,
            _description: *const c_char,
        ) -> Result<*mut FaissIndexBinary> {
            Ok(self.binary_factory_result)
        }

        unsafe fn index_id_map(&self, _index: *mut FaissIndex) -> Result<*mut FaissIndexIDMap> {
            Ok(std::ptr::null_mut())
        }

        unsafe fn index_binary_id_map(
            &self,
            _index: *mut FaissIndexBinary,
        ) -> Result<*mut FaissIndexBinaryIDMap> {
            Ok(std::ptr::null_mut())
        }

        unsafe fn write_index(
            &self,
            _idx: *const FaissIndex,
            _writer: *mut FaissIOWriter,
        ) -> Result<()> {
            Ok(())
        }

        unsafe fn write_index_binary(
            &self,
            _idx: *const FaissIndexBinary,
            _writer: *mut FaissIOWriter,
            _skip_flat: bool,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_mock_index_factory_returns_pointer() {
        // Use a sentinel address to verify the mock returns what we set.
        let sentinel = 0xDEAD_BEEF as *mut FaissIndex;
        let mock = MockFaissMethods {
            factory_result: sentinel,
            binary_factory_result: std::ptr::null_mut(),
        };
        let desc = CString::new("Flat").unwrap();
        unsafe {
            let result = mock
                .index_factory(128, desc.as_ptr(), FaissMetricType::MetricL2)
                .unwrap();
            assert_eq!(result, sentinel);
        }
    }
}
