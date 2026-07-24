// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Stream I/O mediator pattern for Faiss index reading/writing via Java streams.
//!
//! Ported from:
//! - `jni/include/native_engines_stream_support.h` (NativeEngineIndexInputMediator, NativeEngineIndexOutputMediator)
//! - `jni/include/faiss_stream_support.h` (FaissOpenSearchIOReader, FaissOpenSearchIOWriter)
//!
//! # Architecture
//!
//! The C++ pattern uses two layers:
//! 1. **Mediators** — JNI-level objects that call Java `IndexInputWithBuffer` / `IndexOutputWithBuffer`
//!    methods and use `GetPrimitiveArrayCritical` for zero-copy buffer access.
//! 2. **Faiss IO adapters** — C++ classes inheriting `faiss::IOReader` / `faiss::IOWriter` that
//!    delegate to the mediators.
//!
//! In the Rust port:
//! - The mediators are implemented in pure Rust using the `jni` crate's raw JNI sys bindings.
//! - The Faiss IO adapters cannot be pure Rust (they must implement C++ virtual interfaces).
//!   Instead, we use a **C shim callback** approach: Rust provides callback functions that
//!   a C++ shim (`stream_shim.cpp`, not included here) calls from within its `IOReader::operator()`
//!   and `IOWriter::operator()` implementations.
//!
//! # Companion C++ shim (to be created)
//!
//! A file `stream_shim.cpp` must implement:
//! ```cpp
//! // Creates a faiss::IOReader subclass that delegates to the Rust read callback.
//! extern "C" void* faiss_io_reader_from_callback(
//!     void* ctx,
//!     size_t (*read_fn)(void* ctx, uint8_t* dest, size_t nbytes)
//! );
//!
//! // Creates a faiss::IOWriter subclass that delegates to the Rust write callback.
//! extern "C" void* faiss_io_writer_from_callback(
//!     void* ctx,
//!     size_t (*write_fn)(void* ctx, const uint8_t* src, size_t nbytes)
//! );
//!
//! // Frees a reader created by faiss_io_reader_from_callback.
//! extern "C" void faiss_io_reader_free(void* reader);
//!
//! // Frees a writer created by faiss_io_writer_from_callback.
//! extern "C" void faiss_io_writer_free(void* writer);
//! ```

use std::ffi::c_void;
use std::ptr;

use jni::sys::{jbyteArray, jclass, jfieldID, jmethodID, jobject, JNIEnv as RawJNIEnv};

use crate::ffi::faiss_sys::{FaissIOReader, FaissIOWriter};

// ===========================================================================
// Callback types for the C++ shim
// ===========================================================================

/// Callback type for reading bytes. The C++ IOReader shim calls this.
/// Returns the number of items (not bytes) read, matching faiss::IOReader semantics.
pub type ReadCallback = unsafe extern "C" fn(ctx: *mut c_void, dest: *mut u8, nbytes: usize) -> usize;

/// Callback type for writing bytes. The C++ IOWriter shim calls this.
/// Returns the number of items (not bytes) written, matching faiss::IOWriter semantics.
pub type WriteCallback = unsafe extern "C" fn(ctx: *mut c_void, src: *const u8, nbytes: usize) -> usize;

// ===========================================================================
// Extern "C" declarations for the companion C++ shim (stream_shim.cpp)
// ===========================================================================

extern "C" {
    /// Creates a `faiss::IOReader` subclass that delegates `operator()` to the provided
    /// Rust `read_fn` callback with `ctx` as context pointer.
    ///
    /// Returns an opaque pointer that can be cast to `*mut FaissIOReader`.
    ///
    /// # Safety
    /// - `ctx` must remain valid for the lifetime of the returned reader.
    /// - The caller must eventually call `faiss_io_reader_free` to release memory.
    pub fn faiss_io_reader_from_callback(
        ctx: *mut c_void,
        read_fn: ReadCallback,
    ) -> *mut FaissIOReader;

    /// Creates a `faiss::IOWriter` subclass that delegates `operator()` to the provided
    /// Rust `write_fn` callback with `ctx` as context pointer.
    ///
    /// Returns an opaque pointer that can be cast to `*mut FaissIOWriter`.
    ///
    /// # Safety
    /// - `ctx` must remain valid for the lifetime of the returned writer.
    /// - The caller must eventually call `faiss_io_writer_free` to release memory.
    pub fn faiss_io_writer_from_callback(
        ctx: *mut c_void,
        write_fn: WriteCallback,
    ) -> *mut FaissIOWriter;

    /// Frees a reader previously created by `faiss_io_reader_from_callback`.
    pub fn faiss_io_reader_free(reader: *mut FaissIOReader);

    /// Frees a writer previously created by `faiss_io_writer_from_callback`.
    pub fn faiss_io_writer_free(writer: *mut FaissIOWriter);
}

// ===========================================================================
// Java class/method constants
// ===========================================================================

const INDEX_INPUT_WITH_BUFFER_CLASS: &str = "org/opensearch/knn/index/store/IndexInputWithBuffer";
const INDEX_OUTPUT_WITH_BUFFER_CLASS: &str = "org/opensearch/knn/index/store/IndexOutputWithBuffer";

const COPY_BYTES_METHOD_NAME: &str = "copyBytes";
const COPY_BYTES_METHOD_SIG: &str = "(J)I";

const REMAINING_BYTES_METHOD_NAME: &str = "remainingBytes";
const REMAINING_BYTES_METHOD_SIG: &str = "()J";

const WRITE_BYTES_METHOD_NAME: &str = "writeBytes";
const WRITE_BYTES_METHOD_SIG: &str = "(I)V";

const BUFFER_FIELD_NAME: &str = "buffer";
const BUFFER_FIELD_SIG: &str = "[B";

// ===========================================================================
// NativeEngineIndexInputMediator
// ===========================================================================

/// JNI mediator for reading bytes from a Java `IndexInputWithBuffer` object.
///
/// This struct holds raw JNI pointers and method IDs. It calls the Java object's
/// `copyBytes(long)` method to read data into the Java byte[] buffer, then uses
/// `GetPrimitiveArrayCritical` to efficiently copy bytes to a Rust/C++ destination.
///
/// # Safety
///
/// - Must only be used on the thread that created it (JNI env is thread-local).
/// - The `index_input` jobject must remain valid for the lifetime of this struct.
/// - Must not be used after the JNI env is invalidated.
pub struct NativeEngineIndexInputMediator {
    env: *mut RawJNIEnv,
    index_input: jobject,
    buffer_array: jbyteArray,
    clazz: jclass,
    copy_bytes_method: jmethodID,
    remaining_bytes_method: jmethodID,
}

impl NativeEngineIndexInputMediator {
    /// Creates a new input mediator.
    ///
    /// # Safety
    /// - `env` must be a valid JNI environment pointer for the current thread.
    /// - `index_input` must be a valid jobject reference to an `IndexInputWithBuffer` instance.
    pub unsafe fn new(env: *mut RawJNIEnv, index_input: jobject) -> Self {
        assert!(!env.is_null(), "JNIEnv must not be null");
        assert!(!index_input.is_null(), "index_input must not be null");

        let clazz = Self::find_class(env);
        let copy_bytes_method = Self::get_copy_bytes_method(env, clazz);
        let remaining_bytes_method = Self::get_remaining_bytes_method(env, clazz);
        let buffer_field_id = Self::get_buffer_field_id(env, clazz);
        let buffer_array = Self::get_buffer_array(env, index_input, buffer_field_id);

        NativeEngineIndexInputMediator {
            env,
            index_input,
            buffer_array,
            clazz,
            copy_bytes_method,
            remaining_bytes_method,
        }
    }

    /// Reads `nbytes` bytes from the Java IndexInput and copies them to `destination`.
    ///
    /// This calls Java's `copyBytes(long)` in a loop until all requested bytes are read,
    /// using `GetPrimitiveArrayCritical` for efficient zero-copy access to the Java buffer.
    ///
    /// # Safety
    /// - `destination` must point to at least `nbytes` bytes of writable memory.
    pub unsafe fn copy_bytes(&self, mut nbytes: i64, mut destination: *mut u8) {
        while nbytes > 0 {
            // Call IndexInputWithBuffer.copyBytes(nbytes) -> int (bytes actually read)
            let mut args = jni::sys::jvalue { j: nbytes };
            let read_bytes = ((**self.env).CallNonvirtualIntMethodA.unwrap())(
                self.env,
                self.index_input,
                self.clazz,
                self.copy_bytes_method,
                &mut args,
            );

            // Check for Java exceptions
            self.check_exception("Reading bytes via IndexInput has failed.");

            // === Critical Section Start ===
            // Get direct pointer to the Java byte[] — no copy in most JVMs.
            let primitive_array = ((**self.env).GetPrimitiveArrayCritical.unwrap())(
                self.env,
                self.buffer_array,
                ptr::null_mut(),
            );

            // Copy from Java buffer to destination
            ptr::copy_nonoverlapping(
                primitive_array as *const u8,
                destination,
                read_bytes as usize,
            );

            // Release the primitive array (JNI_ABORT = don't copy back, just free)
            ((**self.env).ReleasePrimitiveArrayCritical.unwrap())(
                self.env,
                self.buffer_array,
                primitive_array,
                jni::sys::JNI_ABORT,
            );
            // === Critical Section End ===

            destination = destination.add(read_bytes as usize);
            nbytes -= read_bytes as i64;
        }
    }

    /// Returns the number of bytes remaining to be read from the underlying IndexInput.
    pub unsafe fn remaining_bytes(&self) -> i64 {
        let bytes = ((**self.env).CallNonvirtualLongMethodA.unwrap())(
            self.env,
            self.index_input,
            self.clazz,
            self.remaining_bytes_method,
            ptr::null_mut(),
        );
        self.check_exception("Checking remaining bytes has failed.");
        bytes
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    unsafe fn find_class(env: *mut RawJNIEnv) -> jclass {
        let class_name = std::ffi::CString::new(INDEX_INPUT_WITH_BUFFER_CLASS).unwrap();
        let clazz = ((**env).FindClass.unwrap())(env, class_name.as_ptr());
        assert!(!clazz.is_null(), "Failed to find class {}", INDEX_INPUT_WITH_BUFFER_CLASS);
        clazz
    }

    unsafe fn get_copy_bytes_method(env: *mut RawJNIEnv, clazz: jclass) -> jmethodID {
        let name = std::ffi::CString::new(COPY_BYTES_METHOD_NAME).unwrap();
        let sig = std::ffi::CString::new(COPY_BYTES_METHOD_SIG).unwrap();
        let method = ((**env).GetMethodID.unwrap())(env, clazz, name.as_ptr(), sig.as_ptr());
        assert!(!method.is_null(), "Failed to find method {}", COPY_BYTES_METHOD_NAME);
        method
    }

    unsafe fn get_remaining_bytes_method(env: *mut RawJNIEnv, clazz: jclass) -> jmethodID {
        let name = std::ffi::CString::new(REMAINING_BYTES_METHOD_NAME).unwrap();
        let sig = std::ffi::CString::new(REMAINING_BYTES_METHOD_SIG).unwrap();
        let method = ((**env).GetMethodID.unwrap())(env, clazz, name.as_ptr(), sig.as_ptr());
        assert!(!method.is_null(), "Failed to find method {}", REMAINING_BYTES_METHOD_NAME);
        method
    }

    unsafe fn get_buffer_field_id(env: *mut RawJNIEnv, clazz: jclass) -> jfieldID {
        let name = std::ffi::CString::new(BUFFER_FIELD_NAME).unwrap();
        let sig = std::ffi::CString::new(BUFFER_FIELD_SIG).unwrap();
        let field = ((**env).GetFieldID.unwrap())(env, clazz, name.as_ptr(), sig.as_ptr());
        assert!(!field.is_null(), "Failed to find field {}", BUFFER_FIELD_NAME);
        field
    }

    unsafe fn get_buffer_array(env: *mut RawJNIEnv, obj: jobject, field_id: jfieldID) -> jbyteArray {
        let array = ((**env).GetObjectField.unwrap())(env, obj, field_id);
        assert!(!array.is_null(), "buffer field is null on IndexInputWithBuffer");
        array as jbyteArray
    }

    unsafe fn check_exception(&self, context_msg: &str) {
        let has_exception = ((**self.env).ExceptionCheck.unwrap())(self.env);
        if has_exception != 0 {
            ((**self.env).ExceptionDescribe.unwrap())(self.env);
            ((**self.env).ExceptionClear.unwrap())(self.env);
            panic!("{}", context_msg);
        }
    }
}

// ===========================================================================
// NativeEngineIndexOutputMediator
// ===========================================================================

/// JNI mediator for writing bytes to a Java `IndexOutputWithBuffer` object.
///
/// Copies bytes from a native source into the Java byte[] buffer in chunks,
/// calling `writeBytes(int)` on the Java object whenever the buffer fills up.
///
/// # Safety
///
/// - Must only be used on the thread that created it.
/// - The `index_output` jobject must remain valid for the lifetime of this struct.
pub struct NativeEngineIndexOutputMediator {
    env: *mut RawJNIEnv,
    index_output: jobject,
    buffer_array: jbyteArray,
    clazz: jclass,
    write_bytes_method: jmethodID,
    buffer_length: usize,
    next_write_index: i32,
}

impl NativeEngineIndexOutputMediator {
    /// Creates a new output mediator.
    ///
    /// # Safety
    /// - `env` must be a valid JNI environment pointer for the current thread.
    /// - `index_output` must be a valid jobject reference to an `IndexOutputWithBuffer` instance.
    pub unsafe fn new(env: *mut RawJNIEnv, index_output: jobject) -> Self {
        assert!(!env.is_null(), "JNIEnv must not be null");
        assert!(!index_output.is_null(), "index_output must not be null");

        let clazz = Self::find_class(env);
        let write_bytes_method = Self::get_write_bytes_method(env, clazz);
        let buffer_field_id = Self::get_buffer_field_id(env, clazz);
        let buffer_array = Self::get_buffer_array(env, index_output, buffer_field_id);
        let buffer_length = ((**env).GetArrayLength.unwrap())(env, buffer_array) as usize;

        NativeEngineIndexOutputMediator {
            env,
            index_output,
            buffer_array,
            clazz,
            write_bytes_method,
            buffer_length,
            next_write_index: 0,
        }
    }

    /// Writes `nbytes` bytes from `source` to the Java IndexOutput via buffered copies.
    ///
    /// Data is copied into the Java byte[] buffer using `GetPrimitiveArrayCritical`.
    /// When the buffer is full, `writeBytes(int)` is called on the Java object to flush.
    ///
    /// # Safety
    /// - `source` must point to at least `nbytes` readable bytes.
    pub unsafe fn write_bytes(&mut self, mut source: *const u8, mut nbytes: usize) {
        while nbytes > 0 {
            let available = self.buffer_length - self.next_write_index as usize;
            let write_count = nbytes.min(available);

            // === Critical Section Start ===
            let primitive_array = ((**self.env).GetPrimitiveArrayCritical.unwrap())(
                self.env,
                self.buffer_array,
                ptr::null_mut(),
            );

            // Copy from source to Java buffer at the current write offset
            ptr::copy_nonoverlapping(
                source,
                (primitive_array as *mut u8).add(self.next_write_index as usize),
                write_count,
            );

            // Release with mode 0: copy back content and free (needed because we wrote data)
            ((**self.env).ReleasePrimitiveArrayCritical.unwrap())(
                self.env,
                self.buffer_array,
                primitive_array,
                0,
            );
            // === Critical Section End ===

            self.next_write_index += write_count as i32;
            if self.next_write_index as usize >= self.buffer_length {
                self.call_write_bytes_in_index_output();
            }

            source = source.add(write_count);
            nbytes -= write_count;
        }
    }

    /// Flushes any remaining buffered bytes to the Java IndexOutput.
    ///
    /// Must be called after all writes are complete to ensure no data is left in the buffer.
    pub unsafe fn flush(&mut self) {
        if self.next_write_index > 0 {
            self.call_write_bytes_in_index_output();
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    unsafe fn call_write_bytes_in_index_output(&mut self) {
        let mut args = jni::sys::jvalue { i: self.next_write_index };
        ((**self.env).CallNonvirtualVoidMethodA.unwrap())(
            self.env,
            self.index_output,
            self.clazz,
            self.write_bytes_method,
            &mut args,
        );
        self.check_exception("Writing bytes via IndexOutput has failed.");
        self.next_write_index = 0;
    }

    unsafe fn find_class(env: *mut RawJNIEnv) -> jclass {
        let class_name = std::ffi::CString::new(INDEX_OUTPUT_WITH_BUFFER_CLASS).unwrap();
        let clazz = ((**env).FindClass.unwrap())(env, class_name.as_ptr());
        assert!(!clazz.is_null(), "Failed to find class {}", INDEX_OUTPUT_WITH_BUFFER_CLASS);
        clazz
    }

    unsafe fn get_write_bytes_method(env: *mut RawJNIEnv, clazz: jclass) -> jmethodID {
        let name = std::ffi::CString::new(WRITE_BYTES_METHOD_NAME).unwrap();
        let sig = std::ffi::CString::new(WRITE_BYTES_METHOD_SIG).unwrap();
        let method = ((**env).GetMethodID.unwrap())(env, clazz, name.as_ptr(), sig.as_ptr());
        assert!(!method.is_null(), "Failed to find method {}", WRITE_BYTES_METHOD_NAME);
        method
    }

    unsafe fn get_buffer_field_id(env: *mut RawJNIEnv, clazz: jclass) -> jfieldID {
        let name = std::ffi::CString::new(BUFFER_FIELD_NAME).unwrap();
        let sig = std::ffi::CString::new(BUFFER_FIELD_SIG).unwrap();
        let field = ((**env).GetFieldID.unwrap())(env, clazz, name.as_ptr(), sig.as_ptr());
        assert!(!field.is_null(), "Failed to find field {}", BUFFER_FIELD_NAME);
        field
    }

    unsafe fn get_buffer_array(env: *mut RawJNIEnv, obj: jobject, field_id: jfieldID) -> jbyteArray {
        let array = ((**env).GetObjectField.unwrap())(env, obj, field_id);
        assert!(!array.is_null(), "buffer field is null on IndexOutputWithBuffer");
        array as jbyteArray
    }

    unsafe fn check_exception(&self, context_msg: &str) {
        let has_exception = ((**self.env).ExceptionCheck.unwrap())(self.env);
        if has_exception != 0 {
            ((**self.env).ExceptionDescribe.unwrap())(self.env);
            ((**self.env).ExceptionClear.unwrap())(self.env);
            panic!("{}", context_msg);
        }
    }
}

// ===========================================================================
// Callback trampolines for the C++ shim
// ===========================================================================

/// Read trampoline called by the C++ `faiss::IOReader` shim.
///
/// `ctx` is a pointer to a `NativeEngineIndexInputMediator`.
/// Reads `nbytes` bytes into `dest` and returns `nbytes` (item count = byte count for size=1).
///
/// # Safety
/// - `ctx` must point to a valid `NativeEngineIndexInputMediator`.
/// - `dest` must point to at least `nbytes` writable bytes.
pub unsafe extern "C" fn read_trampoline(
    ctx: *mut c_void,
    dest: *mut u8,
    nbytes: usize,
) -> usize {
    let mediator = &*(ctx as *const NativeEngineIndexInputMediator);
    if nbytes > 0 {
        mediator.copy_bytes(nbytes as i64, dest);
    }
    // Return nbytes to match faiss::IOReader semantics (returns nitems, with size=1 means bytes)
    nbytes
}

/// Write trampoline called by the C++ `faiss::IOWriter` shim.
///
/// `ctx` is a pointer to a `NativeEngineIndexOutputMediator`.
/// Writes `nbytes` bytes from `src` and returns `nbytes`.
///
/// # Safety
/// - `ctx` must point to a valid `NativeEngineIndexOutputMediator`.
/// - `src` must point to at least `nbytes` readable bytes.
pub unsafe extern "C" fn write_trampoline(
    ctx: *mut c_void,
    src: *const u8,
    nbytes: usize,
) -> usize {
    let mediator = &mut *(ctx as *mut NativeEngineIndexOutputMediator);
    if nbytes > 0 {
        mediator.write_bytes(src, nbytes);
    }
    // Return nbytes to match faiss::IOWriter semantics
    nbytes
}

// ===========================================================================
// High-level API: create and destroy Faiss IO reader/writer with mediator
// ===========================================================================

/// Bundles a mediator with its associated Faiss IOReader for lifetime management.
pub struct FaissStreamReader {
    /// The mediator that performs JNI calls. Boxed to provide a stable address for the callback.
    pub mediator: Box<NativeEngineIndexInputMediator>,
    /// The C++ IOReader pointer returned by the shim.
    pub io_reader: *mut FaissIOReader,
}

/// Bundles a mediator with its associated Faiss IOWriter for lifetime management.
pub struct FaissStreamWriter {
    /// The mediator that performs JNI calls. Boxed to provide a stable address for the callback.
    pub mediator: Box<NativeEngineIndexOutputMediator>,
    /// The C++ IOWriter pointer returned by the shim.
    pub io_writer: *mut FaissIOWriter,
}

/// Creates a `FaissStreamReader` combining a JNI input mediator with a Faiss IOReader.
///
/// The returned struct owns both the mediator and the C++ IOReader. The caller must
/// eventually call `destroy_faiss_stream_reader` to free resources.
///
/// # Safety
/// - `env` must be a valid JNI environment pointer for the current thread.
/// - `index_input` must be a valid reference to an `IndexInputWithBuffer` Java object.
pub unsafe fn create_faiss_stream_reader(
    env: *mut RawJNIEnv,
    index_input: jobject,
) -> FaissStreamReader {
    let mediator = Box::new(NativeEngineIndexInputMediator::new(env, index_input));
    let ctx = &*mediator as *const NativeEngineIndexInputMediator as *mut c_void;
    let io_reader = faiss_io_reader_from_callback(ctx, read_trampoline);
    assert!(!io_reader.is_null(), "faiss_io_reader_from_callback returned null");
    FaissStreamReader { mediator, io_reader }
}

/// Creates a `FaissStreamWriter` combining a JNI output mediator with a Faiss IOWriter.
///
/// The returned struct owns both the mediator and the C++ IOWriter. The caller must
/// eventually call `destroy_faiss_stream_writer` to free resources (which also flushes).
///
/// # Safety
/// - `env` must be a valid JNI environment pointer for the current thread.
/// - `index_output` must be a valid reference to an `IndexOutputWithBuffer` Java object.
pub unsafe fn create_faiss_stream_writer(
    env: *mut RawJNIEnv,
    index_output: jobject,
) -> FaissStreamWriter {
    let mediator = Box::new(NativeEngineIndexOutputMediator::new(env, index_output));
    let ctx = &*mediator as *const NativeEngineIndexOutputMediator as *mut c_void;
    let io_writer = faiss_io_writer_from_callback(ctx, write_trampoline);
    assert!(!io_writer.is_null(), "faiss_io_writer_from_callback returned null");
    FaissStreamWriter { mediator, io_writer }
}

/// Destroys a `FaissStreamReader`, freeing the C++ IOReader and dropping the mediator.
///
/// # Safety
/// - `reader` must have been created by `create_faiss_stream_reader`.
/// - Must not be called more than once for the same reader.
pub unsafe fn destroy_faiss_stream_reader(reader: FaissStreamReader) {
    if !reader.io_reader.is_null() {
        faiss_io_reader_free(reader.io_reader);
    }
    // mediator (Box) is dropped automatically
}

/// Destroys a `FaissStreamWriter`, flushing remaining data and freeing the C++ IOWriter.
///
/// # Safety
/// - `writer` must have been created by `create_faiss_stream_writer`.
/// - Must not be called more than once for the same writer.
pub unsafe fn destroy_faiss_stream_writer(mut writer: FaissStreamWriter) {
    // Flush any remaining buffered bytes before destroying
    writer.mediator.flush();
    if !writer.io_writer.is_null() {
        faiss_io_writer_free(writer.io_writer);
    }
    // mediator (Box) is dropped automatically
}
