# C++ to Rust Porting Rulebook — OpenSearch k-NN JNI Layer

## Overview

This document defines the idiom/type translation rules for porting the k-NN JNI layer
from C++ to Rust. Every agent involved in draft/proper-port phases MUST follow these rules.

## Crate Structure

```
knn-jni-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs               # JNI_OnLoad/JNI_OnUnload, re-exports
│   ├── jni_util.rs          # JNIUtilInterface equivalent (JniEnv helpers)
│   ├── faiss_wrapper.rs     # faiss_wrapper namespace → module
│   ├── faiss_index_service.rs
│   ├── faiss_methods.rs
│   ├── faiss_util.rs
│   ├── nmslib_wrapper.rs    # nmslib_wrapper namespace → module
│   ├── commons.rs           # commons namespace → module
│   ├── simd/
│   │   ├── mod.rs
│   │   └── similarity_function.rs
│   └── ffi/
│       ├── mod.rs
│       ├── faiss_sys.rs     # Raw FFI bindings to libfaiss (bindgen or manual)
│       └── nmslib_sys.rs    # Raw FFI bindings to libnmslib
└── build.rs                 # Link to libfaiss, libnmslib, generate JNI headers
```

## Key Dependencies

- `jni` crate (v0.21+) for JNI interop
- `faiss` (C library) via FFI — we keep the C/C++ faiss library and bind to it
- `nmslib` (C++ library) via FFI — same, keep native lib and bind
- No pure-Rust reimplementation of Faiss/NMSLIB — the Rust layer is glue code

## Type Translation Rules

### Primitives

| C++ Type | Rust Type | Notes |
|----------|-----------|-------|
| `jlong` | `jni::sys::jlong` (i64) | |
| `jint` | `jni::sys::jint` (i32) | |
| `jfloat` | `jni::sys::jfloat` (f32) | |
| `jbyte` | `jni::sys::jbyte` (i8) | |
| `jboolean` | `jni::sys::jboolean` (u8) | |
| `jsize` | `jni::sys::jsize` (i32) | |
| `int` | `i32` | |
| `int64_t` | `i64` | |
| `uint8_t` | `u8` | |
| `int8_t` | `i8` | |
| `float` | `f32` | |
| `size_t` | `usize` | |

### JNI Types

| C++ Type | Rust Type | Notes |
|----------|-----------|-------|
| `JNIEnv*` | `JNIEnv<'local>` (jni crate) | Lifetime-bounded |
| `jobject` | `JObject<'local>` | |
| `jclass` | `JClass<'local>` | |
| `jstring` | `JString<'local>` | |
| `jmethodID` | `JMethodID` | |
| `jfieldID` | `JFieldID` | |
| `jobjectArray` | `JObjectArray<'local>` | |
| `jfloatArray` | `JFloatArray<'local>` | Use `AutoElements` |
| `jintArray` | `JIntArray<'local>` | Use `AutoElements` |
| `jbyteArray` | `JByteArray<'local>` | Use `AutoElements` |
| `jlongArray` | `JLongArray<'local>` | Use `AutoElements` |
| `JavaVM*` | `JavaVM` (jni crate) | Owned |

### STL to Rust

| C++ Type | Rust Type | Notes |
|----------|-----------|-------|
| `std::string` | `String` | |
| `std::vector<T>` | `Vec<T>` | |
| `std::unordered_map<K,V>` | `HashMap<K,V>` | |
| `std::unique_ptr<T>` | `Box<T>` or owned T | |
| `std::runtime_error` | Custom error type or `anyhow::Error` | Throw via JNI |
| `std::function<void()>` | `Box<dyn FnOnce()>` or closure | |

### Pointer/Ownership Patterns

| C++ Pattern | Rust Pattern |
|-------------|--------------|
| `reinterpret_cast<T*>(jlong)` | `jlong as *mut T` then unsafe deref |
| `(jlong) ptr` (ptr to jlong) | `ptr as jlong` |
| `delete ptr` (freeing via jlong) | `unsafe { Box::from_raw(ptr) }; // drops` |
| Raw pointer stored in Java (index pointer) | Keep as raw `*mut T`, free in explicit `Free()` |
| `new T(...)` returned as jlong | `Box::into_raw(Box::new(T {...})) as jlong` |

### Error Handling

| C++ Pattern | Rust Pattern |
|-------------|--------------|
| `throw std::runtime_error(msg)` | Return `Err(...)` from Result, catch at JNI boundary |
| `try { ... } catch (...) { ... }` | `std::panic::catch_unwind` + Result propagation |
| `jniUtil->ThrowJavaException(env, type, msg)` | `env.throw_new(type, msg)` |
| `jniUtil->HasExceptionInStack(env)` | `env.exception_check()? → return early` |
| `jniUtil->CatchCppExceptionAndThrowJava(env)` | Macro at `#[no_mangle]` boundary that catches panics |

### JNI Function Signatures

C++ pattern:
```cpp
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndex(
    JNIEnv *env, jclass cls, jintArray idsJ, ...);
```

Rust pattern:
```rust
#[no_mangle]
pub extern "system" fn Java_org_opensearch_knn_jni_FaissService_createIndex(
    mut env: JNIEnv,
    _class: JClass,
    ids: JIntArray,
    ...
) {
    jni_catch_panic(&mut env, || {
        // implementation
        Ok(())
    });
}
```

## Critical Semantic Rules

1. **Never use `Box::from_raw` on a pointer that was not created by `Box::into_raw`**.
   Faiss/NMSLIB pointers come from their own allocators — use their `free` functions via FFI.

2. **All `#[no_mangle] extern "system"` functions MUST catch panics** at the boundary.
   Use a `jni_catch_panic` wrapper that catches `std::panic::catch_unwind` and throws to Java.

3. **JNI local references are scoped by the `JNIEnv` lifetime.** Do not store them beyond
   the function call. Global references (`GlobalRef`) for cached classes/methods.

4. **`omp_set_num_threads` calls** → keep as FFI calls to the OpenMP runtime. Alternatively,
   investigate if Faiss can be built without OpenMP and use Rayon, but that's a future concern.
   For now: `extern "C" { fn omp_set_num_threads(n: c_int); }`

5. **SIMD code** (AVX512, NEON) → use `std::arch` intrinsics with `#[target_feature]` guards,
   or keep as a thin C shim and call via FFI. Port to Rust intrinsics where straightforward.

6. **The `JNIReleaseElements` RAII pattern** → use Rust's `Drop` trait or `scopeguard::defer!`.
   The `jni` crate's `AutoElements` already handles release for primitive arrays.

7. **Index pointers passed from Java as `jlong`** are raw pointers to C++ objects (faiss::Index*).
   They are NOT Rust-owned. Never wrap them in Box/Arc. Access via unsafe raw pointer deref.
   Free them by calling the C++ delete (via FFI destructor wrapper).

8. **Thread safety**: JNI functions are called from Java threads. Faiss index read operations
   are generally thread-safe (multiple concurrent queries OK). Write operations are not.
   The Rust code should NOT add synchronization — Java side handles thread safety.

## Global State

The C++ code has a global `JNIUtil` instance with cached classes/methods.
In Rust: use a `static` `OnceCell<GlobalJniState>` initialized in `JNI_OnLoad`.

```rust
static JNI_STATE: OnceCell<GlobalJniState> = OnceCell::new();

struct GlobalJniState {
    vm: JavaVM,
    cached_classes: HashMap<&'static str, GlobalRef>,
    cached_methods: HashMap<&'static str, JMethodID>,
}
```

## Constants

All `extern const std::string` constants become `const &str` or `static` strings:
```rust
pub const SPACE_TYPE: &str = "spaceType";
pub const L2: &str = "l2";
// etc.
```

## Build Integration

The `build.rs` must:
1. Link to pre-built `libfaiss` and `libnmslib` (from the existing CMake build)
2. Generate JNI header bindings OR use hand-written FFI declarations
3. Handle platform-specific linking (macOS/Linux)
4. The resulting `.dylib`/`.so` replaces the old C++ shared library

## Test Strategy

- Port existing C++ Google Test tests → Rust `#[cfg(test)]` unit tests
- Integration tests remain on the Java side (unchanged)
- The Java classes don't change — only the native library implementation
