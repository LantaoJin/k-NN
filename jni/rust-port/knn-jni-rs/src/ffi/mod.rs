// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! FFI bindings for native vector search libraries (Faiss and NMSLIB).
//!
//! This module provides the minimum set of `extern "C"` declarations needed to
//! call into libfaiss and libnmslib from Rust. Complex C++ types are represented
//! as opaque pointers (`*mut c_void` or empty `#[repr(C)]` enums) since we never
//! access their fields directly from Rust -- all interaction goes through the C API.

pub mod faiss_sys;
pub mod knn_shim;
pub mod nmslib_sys;
