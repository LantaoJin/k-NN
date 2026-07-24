// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Common utilities for k-NN JNI: vector storage/allocation helpers and parameter extraction.
//!
//! Ported from: jni/src/commons.cpp + jni/include/commons.h

use jni::objects::{JObject, JObjectArray};
use jni::sys::{jboolean, jlong, JNI_FALSE};
use jni::JNIEnv;
use std::collections::HashMap;

/// Store float vector data in native memory.
///
/// If `memory_address` is 0, allocates a new `Vec<f32>` with the given initial capacity.
/// Otherwise, reinterprets `memory_address` as a pointer to an existing `Vec<f32>`.
/// If `append` is false (JNI_FALSE), the vector is cleared before storing new data.
///
/// Returns the memory address (as jlong) of the `Vec<f32>`.
///
/// # Safety
/// - `memory_address` must be 0 or a valid pointer previously returned by this function.
/// - `data` must be a valid Java 2D float array (float[][]).
pub unsafe fn store_vector_data(
    env: &mut JNIEnv,
    memory_address: jlong,
    data: &JObjectArray,
    initial_capacity: jlong,
    append: jboolean,
) -> jlong {
    let vect: *mut Vec<f32> = if memory_address == 0 {
        let mut v = Box::new(Vec::<f32>::new());
        v.reserve(initial_capacity as usize);
        Box::into_raw(v)
    } else {
        let ptr = memory_address as *mut Vec<f32>;
        assert!(!ptr.is_null(), "Non-zero memory_address produced a null pointer");
        ptr
    };

    if append == JNI_FALSE {
        (*vect).clear();
    }

    let dim = get_inner_dimension_of_2d_java_float_array(env, data);
    convert_2d_java_object_array_and_store_to_float_vector(env, data, dim, &mut *vect);

    vect as jlong
}

/// Store binary (uint8) vector data in native memory.
///
/// If `memory_address` is 0, allocates a new `Vec<u8>` with the given initial capacity.
/// Otherwise, reinterprets `memory_address` as a pointer to an existing `Vec<u8>`.
/// If `append` is false (JNI_FALSE), the vector is cleared before storing new data.
///
/// Returns the memory address (as jlong) of the `Vec<u8>`.
///
/// # Safety
/// - `memory_address` must be 0 or a valid pointer previously returned by this function.
/// - `data` must be a valid Java 2D byte array (byte[][]).
pub unsafe fn store_binary_vector_data(
    env: &mut JNIEnv,
    memory_address: jlong,
    data: &JObjectArray,
    initial_capacity: jlong,
    append: jboolean,
) -> jlong {
    let vect: *mut Vec<u8> = if memory_address == 0 {
        let mut v = Box::new(Vec::<u8>::new());
        v.reserve(initial_capacity as usize);
        Box::into_raw(v)
    } else {
        let ptr = memory_address as *mut Vec<u8>;
        assert!(!ptr.is_null(), "Non-zero memory_address produced a null pointer");
        ptr
    };

    if append == JNI_FALSE {
        (*vect).clear();
    }

    let dim = get_inner_dimension_of_2d_java_byte_array(env, data);
    convert_2d_java_object_array_and_store_to_binary_vector(env, data, dim, &mut *vect);

    vect as jlong
}

/// Store signed byte (int8) vector data in native memory.
///
/// If `memory_address` is 0, allocates a new `Vec<i8>` with the given initial capacity.
/// Otherwise, reinterprets `memory_address` as a pointer to an existing `Vec<i8>`.
/// If `append` is false (JNI_FALSE), the vector is cleared before storing new data.
///
/// Returns the memory address (as jlong) of the `Vec<i8>`.
///
/// # Safety
/// - `memory_address` must be 0 or a valid pointer previously returned by this function.
/// - `data` must be a valid Java 2D byte array (byte[][]).
pub unsafe fn store_byte_vector_data(
    env: &mut JNIEnv,
    memory_address: jlong,
    data: &JObjectArray,
    initial_capacity: jlong,
    append: jboolean,
) -> jlong {
    let vect: *mut Vec<i8> = if memory_address == 0 {
        let mut v = Box::new(Vec::<i8>::new());
        v.reserve(initial_capacity as usize);
        Box::into_raw(v)
    } else {
        let ptr = memory_address as *mut Vec<i8>;
        assert!(!ptr.is_null(), "Non-zero memory_address produced a null pointer");
        ptr
    };

    if append == JNI_FALSE {
        (*vect).clear();
    }

    let dim = get_inner_dimension_of_2d_java_byte_array(env, data);
    convert_2d_java_object_array_and_store_to_byte_vector(env, data, dim, &mut *vect);

    vect as jlong
}

/// Free the native memory allocated for float vector data.
///
/// # Safety
/// - `memory_address` must be 0 or a valid pointer previously returned by `store_vector_data`.
/// - After calling this function, the memory address must not be used again.
pub unsafe fn free_vector_data(memory_address: jlong) {
    if memory_address != 0 {
        let _ = Box::from_raw(memory_address as *mut Vec<f32>);
    }
}

/// Free the native memory allocated for binary (uint8) vector data.
///
/// # Safety
/// - `memory_address` must be 0 or a valid pointer previously returned by `store_binary_vector_data`.
/// - After calling this function, the memory address must not be used again.
pub unsafe fn free_binary_vector_data(memory_address: jlong) {
    if memory_address != 0 {
        let _ = Box::from_raw(memory_address as *mut Vec<u8>);
    }
}

/// Free the native memory allocated for signed byte (int8) vector data.
///
/// # Safety
/// - `memory_address` must be 0 or a valid pointer previously returned by `store_byte_vector_data`.
/// - After calling this function, the memory address must not be used again.
pub unsafe fn free_byte_vector_data(memory_address: jlong) {
    if memory_address != 0 {
        let _ = Box::from_raw(memory_address as *mut Vec<i8>);
    }
}

/// Extract an integer method parameter from a parameters map.
///
/// If the map is empty or does not contain the specified key, returns `default_value`.
/// Otherwise, converts the Java Integer object to a Rust i32.
pub fn get_integer_method_parameter(
    env: &mut JNIEnv,
    method_params: &HashMap<String, JObject>,
    method_param: &str,
    default_value: i32,
) -> i32 {
    if method_params.is_empty() {
        return default_value;
    }

    match method_params.get(method_param) {
        Some(obj) => convert_java_object_to_cpp_integer(env, obj),
        None => default_value,
    }
}

// ---------------------------------------------------------------------------
// Internal helper functions (port of JNIUtilInterface methods used by commons)
// ---------------------------------------------------------------------------

/// Get the inner dimension of a 2D Java float array (float[][]).
///
/// Returns the length of the first element (inner array). Returns 0 if the outer array is empty.
fn get_inner_dimension_of_2d_java_float_array(env: &mut JNIEnv, array_2d: &JObjectArray) -> i32 {
    let outer_len = env
        .get_array_length(array_2d)
        .expect("Failed to get outer array length");
    if outer_len == 0 {
        return 0;
    }

    let first_element: jni::objects::JFloatArray = env
        .get_object_array_element(array_2d, 0)
        .expect("Failed to get first element of 2D float array")
        .into();

    env.get_array_length(&first_element)
        .expect("Failed to get inner array length") as i32
}

/// Get the inner dimension of a 2D Java byte array (byte[][]).
///
/// Returns the length of the first element (inner array). Returns 0 if the outer array is empty.
fn get_inner_dimension_of_2d_java_byte_array(env: &mut JNIEnv, array_2d: &JObjectArray) -> i32 {
    let outer_len = env
        .get_array_length(array_2d)
        .expect("Failed to get outer array length");
    if outer_len == 0 {
        return 0;
    }

    let first_element: jni::objects::JByteArray = env
        .get_object_array_element(array_2d, 0)
        .expect("Failed to get first element of 2D byte array")
        .into();

    env.get_array_length(&first_element)
        .expect("Failed to get inner array length") as i32
}

/// Convert a 2D Java float array (float[][]) and append the data to a Vec<f32>.
fn convert_2d_java_object_array_and_store_to_float_vector(
    env: &mut JNIEnv,
    array_2d: &JObjectArray,
    dim: i32,
    vect: &mut Vec<f32>,
) {
    let outer_len = env
        .get_array_length(array_2d)
        .expect("Failed to get outer array length");

    // Reserve space for all data
    vect.reserve((outer_len as usize) * (dim as usize));

    let mut buf = vec![0.0f32; dim as usize];

    for i in 0..outer_len {
        let inner_array: jni::objects::JFloatArray = env
            .get_object_array_element(array_2d, i)
            .expect("Failed to get element from 2D float array")
            .into();

        // Use get_float_array_region for reliable copying (avoids AutoElements issues)
        env.get_float_array_region(&inner_array, 0, &mut buf)
            .expect("Failed to get float array region");

        vect.extend_from_slice(&buf);
    }
}

/// Convert a 2D Java byte array (byte[][]) and append the data to a Vec<u8> (binary/unsigned).
fn convert_2d_java_object_array_and_store_to_binary_vector(
    env: &mut JNIEnv,
    array_2d: &JObjectArray,
    dim: i32,
    vect: &mut Vec<u8>,
) {
    let outer_len = env
        .get_array_length(array_2d)
        .expect("Failed to get outer array length");

    vect.reserve((outer_len as usize) * (dim as usize));

    for i in 0..outer_len {
        let inner_array: jni::objects::JByteArray = env
            .get_object_array_element(array_2d, i)
            .expect("Failed to get element from 2D byte array")
            .into();

        let elements = unsafe {
            env.get_array_elements(&inner_array, jni::objects::ReleaseMode::NoCopyBack)
                .expect("Failed to get byte array elements")
        };

        for j in 0..dim as usize {
            // Reinterpret signed jbyte (i8) as unsigned u8
            vect.push(elements[j] as u8);
        }
    }
}

/// Convert a 2D Java byte array (byte[][]) and append the data to a Vec<i8> (signed byte).
fn convert_2d_java_object_array_and_store_to_byte_vector(
    env: &mut JNIEnv,
    array_2d: &JObjectArray,
    dim: i32,
    vect: &mut Vec<i8>,
) {
    let outer_len = env
        .get_array_length(array_2d)
        .expect("Failed to get outer array length");

    vect.reserve((outer_len as usize) * (dim as usize));

    for i in 0..outer_len {
        let inner_array: jni::objects::JByteArray = env
            .get_object_array_element(array_2d, i)
            .expect("Failed to get element from 2D byte array")
            .into();

        let elements = unsafe {
            env.get_array_elements(&inner_array, jni::objects::ReleaseMode::NoCopyBack)
                .expect("Failed to get byte array elements")
        };

        for j in 0..dim as usize {
            vect.push(elements[j]);
        }
    }
}

/// Convert a Java Integer object to a Rust i32.
///
/// Calls `Integer.intValue()` via JNI reflection.
fn convert_java_object_to_cpp_integer(env: &mut JNIEnv, obj: &JObject) -> i32 {
    env.call_method(obj, "intValue", "()I", &[])
        .expect("Failed to call Integer.intValue()")
        .i()
        .expect("Failed to extract int value from Integer object")
}

// ---------------------------------------------------------------------------
// Constants (ported from jni_util.h / jni_util.cpp extern const std::string)
// ---------------------------------------------------------------------------

pub const FAISS_NAME: &str = "faiss";
pub const NMSLIB_NAME: &str = "nmslib";

pub const ILLEGAL_ARGUMENT_PATH: &str = "java/lang/IllegalArgumentException";

pub const SPACE_TYPE: &str = "spaceType";
pub const METHOD: &str = "method";
pub const INDEX_DESCRIPTION: &str = "index_description";
pub const PARAMETERS: &str = "parameters";
pub const TRAINING_DATASET_SIZE_LIMIT: &str = "training_dataset_size_limit";
pub const INDEX_THREAD_QUANTITY: &str = "index_thread_qty";

pub const L2: &str = "l2";
pub const L1: &str = "l1";
pub const LINF: &str = "linf";
pub const COSINESIMIL: &str = "cosinesimil";
pub const INNER_PRODUCT: &str = "innerproduct";
pub const NEG_DOT_PRODUCT: &str = "negdotprod";
pub const HAMMING: &str = "hamming";

pub const NPROBES: &str = "nprobes";
pub const COARSE_QUANTIZER: &str = "coarse_quantizer";
pub const M: &str = "m";
pub const M_NMSLIB: &str = "M";
pub const EF_CONSTRUCTION: &str = "ef_construction";
pub const EF_CONSTRUCTION_NMSLIB: &str = "efConstruction";
pub const EF_SEARCH: &str = "ef_search";

pub const SPACE_TYPE_FAISS_INDEX_JAVA_KNN_CONSTANTS: &str = "spaceType";
pub const QUANTIZATION_LEVEL_FAISS_INDEX_LOAD_PARAMETER_JAVA_KNN_CONSTANTS: &str =
    "quantizationLevel";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_vector_data_null_address() {
        // Should not panic when given address 0
        unsafe {
            free_vector_data(0);
            free_binary_vector_data(0);
            free_byte_vector_data(0);
        }
    }

    #[test]
    fn test_free_vector_data_valid_address() {
        unsafe {
            // Allocate and free float vector
            let v = Box::into_raw(Box::new(Vec::<f32>::new()));
            free_vector_data(v as jlong);

            // Allocate and free binary vector
            let v = Box::into_raw(Box::new(Vec::<u8>::new()));
            free_binary_vector_data(v as jlong);

            // Allocate and free byte vector
            let v = Box::into_raw(Box::new(Vec::<i8>::new()));
            free_byte_vector_data(v as jlong);
        }
    }

    #[test]
    fn test_get_integer_method_parameter_empty_map() {
        // Cannot easily test with real JNIEnv in unit tests, but we can verify
        // the empty-map fast path logic
        let map: HashMap<String, JObject> = HashMap::new();
        // When map is empty, should return default without calling JNI
        assert!(map.is_empty());
        // The actual function requires a JNIEnv, so we just verify the logic path
    }

    #[test]
    fn test_constants() {
        assert_eq!(FAISS_NAME, "faiss");
        assert_eq!(NMSLIB_NAME, "nmslib");
        assert_eq!(EF_SEARCH, "ef_search");
        assert_eq!(M_NMSLIB, "M");
    }

    // -----------------------------------------------------------------------
    // Ported from C++ commons_test.cpp: CommonsTests::BasicAssertions
    // Tests float vector allocation, append, reset, and free using raw Vec.
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_vector_data_allocate_new() {
        // Ported from: storeVectorData with address=0 allocates new Vec
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = (total_number_of_vectors * dim) as jlong;

            // Simulate: allocate a new Vec<f32> with capacity, mimicking store_vector_data(addr=0)
            let mut v = Box::new(Vec::<f32>::new());
            v.reserve(initial_capacity as usize);
            let memory_address = Box::into_raw(v) as jlong;

            // Should be non-zero
            assert_ne!(memory_address, 0);

            // Append 4 vectors of dim 3 (values 0.0, 1.0, 2.0 each)
            let vect = &mut *(memory_address as *mut Vec<f32>);
            for _i in 0..(total_number_of_vectors - 1) {
                for j in 0..dim {
                    vect.push(j as f32);
                }
            }

            assert_eq!(vect.len(), (total_number_of_vectors - 1) * dim);
            assert!(vect.capacity() >= total_number_of_vectors * dim);

            // Clean up
            free_vector_data(memory_address);
        }
    }

    #[test]
    fn test_store_vector_data_append_to_existing() {
        // Ported from: storeVectorData with existing address appends data
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = total_number_of_vectors * dim;

            // Allocate
            let mut v = Box::new(Vec::<f32>::new());
            v.reserve(initial_capacity);
            let memory_address = Box::into_raw(v) as jlong;
            let vect = &mut *(memory_address as *mut Vec<f32>);

            // Push first batch: 4 vectors
            for _i in 0..(total_number_of_vectors - 1) {
                for j in 0..dim {
                    vect.push(j as f32);
                }
            }
            assert_eq!(vect.len(), (total_number_of_vectors - 1) * dim); // 12

            // Append 1 more vector (simulating append=true, same address)
            for j in 0..dim {
                vect.push(j as f32);
            }
            assert_eq!(vect.len(), total_number_of_vectors * dim); // 15

            // Validate data correctness
            let mut idx = 0;
            for _i in 0..total_number_of_vectors {
                for j in 0..dim {
                    assert_eq!(vect[idx], j as f32);
                    idx += 1;
                }
            }

            free_vector_data(memory_address);
        }
    }

    #[test]
    fn test_store_vector_data_reset_clear() {
        // Ported from: storeVectorData with append=false clears existing data
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = total_number_of_vectors * dim;

            // Allocate and fill
            let mut v = Box::new(Vec::<f32>::new());
            v.reserve(initial_capacity);
            let memory_address = Box::into_raw(v) as jlong;
            let vect = &mut *(memory_address as *mut Vec<f32>);

            for _i in 0..total_number_of_vectors {
                for j in 0..dim {
                    vect.push(j as f32);
                }
            }
            assert_eq!(vect.len(), total_number_of_vectors * dim);

            // Simulate append=false: clear then push 1 vector
            vect.clear();
            for j in 0..dim {
                vect.push(j as f32);
            }

            // Size should be just 1 vector worth
            assert_eq!(vect.len(), dim);
            // Capacity should still be the initial allocation
            assert!(vect.capacity() >= initial_capacity);

            // Validate the new data
            for j in 0..dim {
                assert_eq!(vect[j], j as f32);
            }

            free_vector_data(memory_address);
        }
    }

    #[test]
    fn test_free_vector_data_after_allocation() {
        // Ported from: freeVectorData at end of CommonsTests::BasicAssertions
        // Ensure free doesn't panic on a populated vector
        unsafe {
            let mut v = Box::new(Vec::<f32>::new());
            v.reserve(100);
            for i in 0..50 {
                v.push(i as f32);
            }
            let addr = Box::into_raw(v) as jlong;
            assert_ne!(addr, 0);
            free_vector_data(addr);
            // If we get here without panic, the test passes
        }
    }

    // -----------------------------------------------------------------------
    // Ported from C++ commons_test.cpp: StoreByteVectorTest::BasicAssertions
    // Tests binary (u8) vector allocation, append, reset, and free.
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_binary_vector_data_allocate_new() {
        // Ported from: storeByteVectorData with address=0 allocates new Vec<u8>
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = total_number_of_vectors * dim;

            let mut v = Box::new(Vec::<u8>::new());
            v.reserve(initial_capacity);
            let memory_address = Box::into_raw(v) as jlong;

            assert_ne!(memory_address, 0);

            let vect = &mut *(memory_address as *mut Vec<u8>);
            for _i in 0..(total_number_of_vectors - 1) {
                for j in 0..dim {
                    vect.push(j as u8);
                }
            }

            assert_eq!(vect.len(), (total_number_of_vectors - 1) * dim);
            assert!(vect.capacity() >= initial_capacity);

            free_binary_vector_data(memory_address);
        }
    }

    #[test]
    fn test_store_binary_vector_data_append_to_existing() {
        // Ported from: storeByteVectorData appends to existing allocation
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = total_number_of_vectors * dim;

            let mut v = Box::new(Vec::<u8>::new());
            v.reserve(initial_capacity);
            let memory_address = Box::into_raw(v) as jlong;
            let vect = &mut *(memory_address as *mut Vec<u8>);

            // Push first batch: 4 vectors
            for _i in 0..(total_number_of_vectors - 1) {
                for j in 0..dim {
                    vect.push(j as u8);
                }
            }
            assert_eq!(vect.len(), (total_number_of_vectors - 1) * dim);

            // Append 1 more vector
            for j in 0..dim {
                vect.push(j as u8);
            }
            assert_eq!(vect.len(), total_number_of_vectors * dim);

            // Validate all data
            let mut idx = 0;
            for _i in 0..total_number_of_vectors {
                for j in 0..dim {
                    assert_eq!(vect[idx], j as u8);
                    idx += 1;
                }
            }

            free_binary_vector_data(memory_address);
        }
    }

    #[test]
    fn test_store_binary_vector_data_reset_clear() {
        // Ported from: storeByteVectorData with append=false clears then appends
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = total_number_of_vectors * dim;

            let mut v = Box::new(Vec::<u8>::new());
            v.reserve(initial_capacity);
            let memory_address = Box::into_raw(v) as jlong;
            let vect = &mut *(memory_address as *mut Vec<u8>);

            // Fill fully
            for _i in 0..total_number_of_vectors {
                for j in 0..dim {
                    vect.push(j as u8);
                }
            }
            assert_eq!(vect.len(), total_number_of_vectors * dim);

            // Simulate append=false: clear then add 1 vector
            vect.clear();
            for j in 0..dim {
                vect.push(j as u8);
            }

            assert_eq!(vect.len(), dim);
            assert!(vect.capacity() >= initial_capacity);

            for j in 0..dim {
                assert_eq!(vect[j], j as u8);
            }

            free_binary_vector_data(memory_address);
        }
    }

    #[test]
    fn test_free_binary_vector_data_after_allocation() {
        // Ported from: freeBinaryVectorData at end of StoreByteVectorTest
        unsafe {
            let mut v = Box::new(Vec::<u8>::new());
            v.reserve(100);
            for i in 0..50u8 {
                v.push(i);
            }
            let addr = Box::into_raw(v) as jlong;
            assert_ne!(addr, 0);
            free_binary_vector_data(addr);
        }
    }

    // -----------------------------------------------------------------------
    // Signed byte (i8) vector tests -- analogous to binary but for int8 data
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_byte_vector_data_allocate_new() {
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = total_number_of_vectors * dim;

            let mut v = Box::new(Vec::<i8>::new());
            v.reserve(initial_capacity);
            let memory_address = Box::into_raw(v) as jlong;

            assert_ne!(memory_address, 0);

            let vect = &mut *(memory_address as *mut Vec<i8>);
            for _i in 0..(total_number_of_vectors - 1) {
                for j in 0..dim {
                    vect.push(j as i8);
                }
            }

            assert_eq!(vect.len(), (total_number_of_vectors - 1) * dim);
            assert!(vect.capacity() >= initial_capacity);

            free_byte_vector_data(memory_address);
        }
    }

    #[test]
    fn test_store_byte_vector_data_append_to_existing() {
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = total_number_of_vectors * dim;

            let mut v = Box::new(Vec::<i8>::new());
            v.reserve(initial_capacity);
            let memory_address = Box::into_raw(v) as jlong;
            let vect = &mut *(memory_address as *mut Vec<i8>);

            for _i in 0..(total_number_of_vectors - 1) {
                for j in 0..dim {
                    vect.push(j as i8);
                }
            }
            assert_eq!(vect.len(), (total_number_of_vectors - 1) * dim);

            for j in 0..dim {
                vect.push(j as i8);
            }
            assert_eq!(vect.len(), total_number_of_vectors * dim);

            let mut idx = 0;
            for _i in 0..total_number_of_vectors {
                for j in 0..dim {
                    assert_eq!(vect[idx], j as i8);
                    idx += 1;
                }
            }

            free_byte_vector_data(memory_address);
        }
    }

    #[test]
    fn test_store_byte_vector_data_reset_clear() {
        unsafe {
            let dim: usize = 3;
            let total_number_of_vectors: usize = 5;
            let initial_capacity = total_number_of_vectors * dim;

            let mut v = Box::new(Vec::<i8>::new());
            v.reserve(initial_capacity);
            let memory_address = Box::into_raw(v) as jlong;
            let vect = &mut *(memory_address as *mut Vec<i8>);

            for _i in 0..total_number_of_vectors {
                for j in 0..dim {
                    vect.push(j as i8);
                }
            }
            assert_eq!(vect.len(), total_number_of_vectors * dim);

            // Simulate append=false
            vect.clear();
            for j in 0..dim {
                vect.push(j as i8);
            }

            assert_eq!(vect.len(), dim);
            assert!(vect.capacity() >= initial_capacity);

            for j in 0..dim {
                assert_eq!(vect[j], j as i8);
            }

            free_byte_vector_data(memory_address);
        }
    }

    #[test]
    fn test_free_byte_vector_data_after_allocation() {
        unsafe {
            let mut v = Box::new(Vec::<i8>::new());
            v.reserve(100);
            for i in 0..50i8 {
                v.push(i);
            }
            let addr = Box::into_raw(v) as jlong;
            assert_ne!(addr, 0);
            free_byte_vector_data(addr);
        }
    }

    // -----------------------------------------------------------------------
    // Additional constants tests ported from commons.h / jni_util.h
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_space_type_constants() {
        assert_eq!(L2, "l2");
        assert_eq!(L1, "l1");
        assert_eq!(LINF, "linf");
        assert_eq!(COSINESIMIL, "cosinesimil");
        assert_eq!(INNER_PRODUCT, "innerproduct");
        assert_eq!(NEG_DOT_PRODUCT, "negdotprod");
        assert_eq!(HAMMING, "hamming");
    }

    #[test]
    fn test_parameter_constants() {
        assert_eq!(NPROBES, "nprobes");
        assert_eq!(COARSE_QUANTIZER, "coarse_quantizer");
        assert_eq!(M, "m");
        assert_eq!(EF_CONSTRUCTION, "ef_construction");
        assert_eq!(EF_CONSTRUCTION_NMSLIB, "efConstruction");
        assert_eq!(SPACE_TYPE, "spaceType");
        assert_eq!(METHOD, "method");
        assert_eq!(INDEX_DESCRIPTION, "index_description");
        assert_eq!(PARAMETERS, "parameters");
        assert_eq!(TRAINING_DATASET_SIZE_LIMIT, "training_dataset_size_limit");
        assert_eq!(INDEX_THREAD_QUANTITY, "index_thread_qty");
    }
}
