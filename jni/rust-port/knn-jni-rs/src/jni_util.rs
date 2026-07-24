// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! JNI utility helpers: type conversions, exception handling, cached classes/methods.
//! Ported from jni/src/jni_util.cpp and jni/include/jni_util.h

use jni::objects::{
    GlobalRef, JByteArray, JClass, JFieldID, JFloatArray, JIntArray, JLongArray, JMethodID,
    JObject, JObjectArray, JString,
};
use jni::sys::{jarray, jfieldID, jint, jsize};
// These are used by the gated nonvirtual methods and the primitive array critical methods:
#[cfg(any())]
use jni::sys::{jlong, jvalue};
use jni::JNIEnv;
use jni::JavaVM;
use std::collections::HashMap;

// ------------------------------- CONSTANTS --------------------------------

pub const FAISS_NAME: &str = "faiss";
pub const NMSLIB_NAME: &str = "nmslib";

pub const ILLEGAL_ARGUMENT_PATH: &str = "java/lang/IllegalArgumentException";

pub const SPACE_TYPE: &str = "spaceType";
pub const METHOD: &str = "method";
pub const INDEX_DESCRIPTION: &str = "index_description";
pub const PARAMETERS: &str = "parameters";
pub const TRAINING_DATASET_SIZE_LIMIT: &str = "training_dataset_size_limit";
pub const INDEX_THREAD_QUANTITY: &str = "indexThreadQty";

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

pub const SPACE_TYPE_FAISS_INDEX_JAVA_KNN_CONSTANTS: &str = "space_type";
pub const QUANTIZATION_LEVEL_FAISS_INDEX_LOAD_PARAMETER_JAVA_KNN_CONSTANTS: &str =
    "quantization_level";

// --------------------------------------------------------------------------

/// Quantization level for binary quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BQQuantizationLevel {
    OneBit,
    TwoBit,
    FourBit,
    None,
}

/// Error type for JNI utility operations.
#[derive(Debug, thiserror::Error)]
pub enum JniUtilError {
    #[error("{0}")]
    Runtime(String),

    #[error("JNI error: {0}")]
    Jni(#[from] jni::errors::Error),

    #[error("Null pointer: {0}")]
    NullPointer(String),
}

pub type Result<T> = std::result::Result<T, JniUtilError>;

/// RAII guard that executes a closure on drop, analogous to C++ JNIReleaseElements.
pub struct JniReleaseGuard<F: FnOnce()> {
    release_func: Option<F>,
}

impl<F: FnOnce()> JniReleaseGuard<F> {
    pub fn new(f: F) -> Self {
        JniReleaseGuard {
            release_func: Some(f),
        }
    }
}

impl<F: FnOnce()> Drop for JniReleaseGuard<F> {
    fn drop(&mut self) {
        if let Some(f) = self.release_func.take() {
            // Ignore panics in drop, analogous to catch(...) in C++
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        }
    }
}

/// Cached Java class and method references for JNI operations.
/// This struct mirrors the C++ `knn_jni::JNIUtil` class.
pub struct JNIUtil {
    vm: Option<JavaVM>,
    cached_classes: HashMap<String, GlobalRef>,
    cached_methods: HashMap<String, JMethodID>,
}

impl JNIUtil {
    /// Create a new uninitialized JNIUtil.
    pub fn new() -> Self {
        JNIUtil {
            vm: None,
            cached_classes: HashMap::new(),
            cached_methods: HashMap::new(),
        }
    }

    /// Initialize cached classes and methods. Must be called from a thread attached to the JVM.
    pub fn initialize(&mut self, env: &mut JNIEnv, java_vm: JavaVM) -> Result<()> {
        // Cache IOException
        self.cache_class(env, "java/io/IOException")?;

        // Cache Exception
        self.cache_class(env, "java/lang/Exception")?;

        // Cache Map and its entrySet method
        self.cache_class(env, "java/util/Map")?;
        self.cache_method(env, "java/util/Map", "entrySet", "()Ljava/util/Set;")?;

        // Cache Set and its iterator method
        self.cache_class(env, "java/util/Set")?;
        self.cache_method(env, "java/util/Set", "iterator", "()Ljava/util/Iterator;")?;

        // Cache Iterator and its hasNext/next methods
        self.cache_class(env, "java/util/Iterator")?;
        self.cache_method(env, "java/util/Iterator", "hasNext", "()Z")?;
        self.cache_method(
            env,
            "java/util/Iterator",
            "next",
            "()Ljava/lang/Object;",
        )?;

        // Cache Object
        self.cache_class(env, "java/lang/Object")?;

        // Cache Map$Entry and its getKey/getValue methods
        self.cache_class(env, "java/util/Map$Entry")?;
        self.cache_method(
            env,
            "java/util/Map$Entry",
            "getKey",
            "()Ljava/lang/Object;",
        )?;
        self.cache_method(
            env,
            "java/util/Map$Entry",
            "getValue",
            "()Ljava/lang/Object;",
        )?;

        // Cache Integer and its intValue method
        self.cache_class(env, "java/lang/Integer")?;
        self.cache_method(env, "java/lang/Integer", "intValue", "()I")?;

        // Cache KNNQueryResult and its constructor
        self.cache_class(env, "org/opensearch/knn/index/query/KNNQueryResult")?;
        self.cache_method(
            env,
            "org/opensearch/knn/index/query/KNNQueryResult",
            "<init>",
            "(IF)V",
        )?;

        // Cache MergeAbortChecker (static method)
        self.cache_class(env, "org/apache/lucene/index/MergeAbortChecker")?;
        // Note: static method cached via GetStaticMethodID in C++.
        // In the jni crate, static methods use the same JMethodID type.
        self.cache_static_method(
            env,
            "org/apache/lucene/index/MergeAbortChecker",
            "isMergeAborted",
            "()Z",
        )?;

        self.vm = Some(java_vm);
        Ok(())
    }

    /// Uninitialize: drop all global references.
    pub fn uninitialize(&mut self) {
        self.cached_classes.clear();
        self.cached_methods.clear();
        self.vm = None;
    }

    /// Get the current JNI environment for the calling thread.
    pub fn get_jni_current_env(&self) -> Option<JNIEnv<'_>> {
        let vm = self.vm.as_ref()?;
        // Safety: attach_current_thread_as_daemon is the typical approach
        // to get JNIEnv on an already-attached thread.
        vm.get_env().ok()
    }

    // ======================== EXCEPTION HANDLING ========================

    /// Throw a Java exception of the given type with the given message.
    pub fn throw_java_exception(&self, env: &mut JNIEnv, exception_type: &str, message: &str) {
        // Attempt to throw; if FindClass fails, NoClassDefFoundError is thrown by JVM
        let _ = env.throw_new(exception_type, message);
    }

    /// Check if there is a pending Java exception. If so, return an error.
    pub fn has_exception_in_stack(&self, env: &mut JNIEnv) -> Result<()> {
        self.has_exception_in_stack_with_message(env, "Exception in jni occurred")
    }

    /// Check if there is a pending Java exception with a custom message.
    pub fn has_exception_in_stack_with_message(
        &self,
        env: &mut JNIEnv,
        message: &str,
    ) -> Result<()> {
        if env.exception_check()? {
            Err(JniUtilError::Runtime(message.to_string()))
        } else {
            Ok(())
        }
    }

    /// Catch an index build abort scenario and throw the corresponding Java exception.
    pub fn catch_index_build_abort_exception_and_throw_java(&self, env: &mut JNIEnv) {
        self.throw_java_exception(
            env,
            "org/opensearch/knn/index/codec/nativeindex/IndexBuildAbortedException",
            "Faiss index build aborted",
        );
    }

    /// Catch a Rust/native error and throw the corresponding Java exception.
    /// This is the Rust equivalent of CatchCppExceptionAndThrowJava.
    pub fn catch_rust_error_and_throw_java(&self, env: &mut JNIEnv, error: &JniUtilError) {
        match error {
            JniUtilError::Runtime(msg) => {
                self.throw_java_exception(env, "java/lang/Exception", msg);
            }
            JniUtilError::Jni(e) => {
                self.throw_java_exception(env, "java/lang/Exception", &e.to_string());
            }
            JniUtilError::NullPointer(msg) => {
                self.throw_java_exception(env, "java/lang/Exception", msg);
            }
        }
    }

    /// Catch any panic or error and throw an appropriate Java exception.
    /// Use this at JNI boundaries.
    pub fn catch_and_throw_java(&self, env: &mut JNIEnv, error: Box<dyn std::any::Any + Send>) {
        let message = if let Some(s) = error.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = error.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "Unknown exception occurred".to_string()
        };
        self.throw_java_exception(env, "java/lang/Exception", &message);
    }

    // ======================== JAVA FINDERS ========================

    /// Find a cached Java class by name.
    pub fn find_class(&self, _env: &mut JNIEnv, class_name: &str) -> Result<&GlobalRef> {
        self.cached_classes.get(class_name).ok_or_else(|| {
            JniUtilError::Runtime(format!("Unable to load class \"{}\"", class_name))
        })
    }

    /// Find a cached method by class name and method name.
    pub fn find_method(
        &self,
        _env: &mut JNIEnv,
        class_name: &str,
        method_name: &str,
    ) -> Result<JMethodID> {
        let key = format!("{}:{}", class_name, method_name);
        self.cached_methods.get(&key).copied().ok_or_else(|| {
            JniUtilError::Runtime(format!("Unable to find \"{}\" method", method_name))
        })
    }

    // ======================== JAVA TO RUST CONVERTERS ========================

    /// Convert a Java String to a Rust String.
    pub fn convert_java_string_to_rust_string(
        &self,
        env: &mut JNIEnv,
        java_string: &JString,
    ) -> Result<String> {
        if java_string.is_null() {
            return Err(JniUtilError::NullPointer("String cannot be null".to_string()));
        }
        let rust_string: String = env.get_string(java_string)?.into();
        Ok(rust_string)
    }

    /// Convert a Java Object (assumed to be String) to a Rust String.
    pub fn convert_java_object_to_rust_string(
        &self,
        env: &mut JNIEnv,
        object: &JObject,
    ) -> Result<String> {
        let jstr: JString = JString::from(unsafe { JObject::from_raw(object.as_raw()) });
        self.convert_java_string_to_rust_string(env, &jstr)
    }

    /// Convert a Java String to a BQQuantizationLevel enum.
    pub fn convert_java_string_to_quantization_level(
        &self,
        env: &mut JNIEnv,
        java_string: &JObject,
    ) -> Result<BQQuantizationLevel> {
        if java_string.is_null() {
            return Err(JniUtilError::NullPointer("String cannot be null".to_string()));
        }
        let jstr = JString::from(unsafe { JObject::from_raw(java_string.as_raw()) });
        let rust_str = self.convert_java_string_to_rust_string(env, &jstr)?;

        match rust_str.as_str() {
            "ScalarQuantizationParams_1" => Ok(BQQuantizationLevel::OneBit),
            "ScalarQuantizationParams_2" => Ok(BQQuantizationLevel::TwoBit),
            "ScalarQuantizationParams_4" => Ok(BQQuantizationLevel::FourBit),
            _ => Err(JniUtilError::Runtime(
                "Unable to convert java string to quantization level".to_string(),
            )),
        }
    }

    /// Convert a Java Object (assumed Integer) to a Rust i32.
    pub fn convert_java_object_to_rust_integer(
        &self,
        env: &mut JNIEnv,
        object: &JObject,
    ) -> Result<i32> {
        if object.is_null() {
            return Err(JniUtilError::NullPointer(
                "Object cannot be null".to_string(),
            ));
        }

        let integer_class_ref = self.find_class(env, "java/lang/Integer")?;
        let integer_class = JClass::from(unsafe { JObject::from_raw(integer_class_ref.as_obj().as_raw()) });
        let int_value_method = self.find_method(env, "java/lang/Integer", "intValue")?;

        if !env.is_instance_of(object, &integer_class)? {
            return Err(JniUtilError::Runtime(
                "Cannot call IntMethod on non-integer class".to_string(),
            ));
        }

        // Safety: calling a known method on a validated Integer instance
        let result = unsafe {
            env.call_method_unchecked(
                object,
                int_value_method,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Int),
                &[],
            )?
        };
        self.has_exception_in_stack_with_message(env, "Could not call \"intValue\" method on Integer")?;

        Ok(result.i()?)
    }

    /// Convert a Java Map<String, Object> to a Rust HashMap<String, JObject>.
    /// Note: The returned JObjects are local references valid until the native method returns.
    pub fn convert_java_map_to_rust_map<'local>(
        &self,
        env: &mut JNIEnv<'local>,
        parameters: &JObject,
    ) -> Result<HashMap<String, JObject<'local>>> {
        if parameters.is_null() {
            return Err(JniUtilError::NullPointer(
                "Parameters cannot be null".to_string(),
            ));
        }

        let entry_set_method = self.find_method(env, "java/util/Map", "entrySet")?;
        let iterator_method = self.find_method(env, "java/util/Set", "iterator")?;
        let has_next_method = self.find_method(env, "java/util/Iterator", "hasNext")?;
        let next_method = self.find_method(env, "java/util/Iterator", "next")?;
        let get_key_method = self.find_method(env, "java/util/Map$Entry", "getKey")?;
        let get_value_method = self.find_method(env, "java/util/Map$Entry", "getValue")?;

        // Call entrySet()
        let entry_set = unsafe {
            env.call_method_unchecked(
                parameters,
                entry_set_method,
                jni::signature::ReturnType::Object,
                &[],
            )?
        };
        self.has_exception_in_stack_with_message(
            env,
            "Unable to call \"entrySet\" method on \"java/util/Map\"",
        )?;
        let entry_set_obj = entry_set.l()?;

        // Call iterator()
        let iter = unsafe {
            env.call_method_unchecked(
                &entry_set_obj,
                iterator_method,
                jni::signature::ReturnType::Object,
                &[],
            )?
        };
        self.has_exception_in_stack_with_message(env, "Call to \"iterator\" method failed")?;
        let iter_obj = iter.l()?;

        let mut result_map: HashMap<String, JObject<'local>> = HashMap::new();

        loop {
            // Call hasNext()
            let has_next = unsafe {
                env.call_method_unchecked(
                    &iter_obj,
                    has_next_method,
                    jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
                    &[],
                )?
            };
            if has_next.z()? == false {
                break;
            }

            // Call next()
            let entry = unsafe {
                env.call_method_unchecked(
                    &iter_obj,
                    next_method,
                    jni::signature::ReturnType::Object,
                    &[],
                )?
            };
            self.has_exception_in_stack_with_message(env, "Could not call \"next\" method")?;
            let entry_obj = entry.l()?;

            // Call getKey()
            let key_val = unsafe {
                env.call_method_unchecked(
                    &entry_obj,
                    get_key_method,
                    jni::signature::ReturnType::Object,
                    &[],
                )?
            };
            self.has_exception_in_stack_with_message(env, "Could not call \"getKey\" method")?;
            let key_obj = key_val.l()?;
            let key_jstr = JString::from(key_obj);
            let key_string = self.convert_java_string_to_rust_string(env, &key_jstr)?;

            // Call getValue()
            let value_val = unsafe {
                env.call_method_unchecked(
                    &entry_obj,
                    get_value_method,
                    jni::signature::ReturnType::Object,
                    &[],
                )?
            };
            self.has_exception_in_stack_with_message(env, "Could not call \"getValue\" method")?;
            let value_obj = value_val.l()?;

            result_map.insert(key_string, value_obj);

            env.delete_local_ref(entry_obj)?;
        }

        self.has_exception_in_stack_with_message(env, "Could not call \"hasNext\" method")?;

        env.delete_local_ref(entry_set_obj)?;
        env.delete_local_ref(iter_obj)?;

        Ok(result_map)
    }

    /// Convert a 2D Java float[][] to a flat Vec<f32>.
    pub fn convert_2d_java_object_array_to_float_vector(
        &self,
        env: &mut JNIEnv,
        array_2d: &JObjectArray,
        dim: i32,
    ) -> Result<Vec<f32>> {
        let mut vect = Vec::new();
        self.convert_2d_java_object_array_and_store_to_float_vector(env, array_2d, dim, &mut vect)?;
        Ok(vect)
    }

    /// Convert a 2D Java float[][] and append to an existing Vec<f32>.
    pub fn convert_2d_java_object_array_and_store_to_float_vector(
        &self,
        env: &mut JNIEnv,
        array_2d: &JObjectArray,
        dim: i32,
        vect: &mut Vec<f32>,
    ) -> Result<()> {
        if array_2d.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }

        let num_vectors = env.get_array_length(array_2d)?;
        self.has_exception_in_stack(env)?;

        for i in 0..num_vectors {
            let vector_array_obj = env.get_object_array_element(array_2d, i)?;
            self.has_exception_in_stack_with_message(env, "Unable to get object array element")?;

            let vector_array = JFloatArray::from(vector_array_obj);
            let arr_len = env.get_array_length(&vector_array)?;
            if dim != arr_len {
                return Err(JniUtilError::Runtime(
                    "Dimension of vectors is inconsistent".to_string(),
                ));
            }

            let mut buf = vec![0.0f32; dim as usize];
            env.get_float_array_region(&vector_array, 0, &mut buf)?;

            vect.extend_from_slice(&buf);
        }

        self.has_exception_in_stack(env)?;
        Ok(())
    }

    /// Convert a 2D Java byte[][] to a flat Vec<u8> (binary vectors).
    pub fn convert_2d_java_object_array_and_store_to_binary_vector(
        &self,
        env: &mut JNIEnv,
        array_2d: &JObjectArray,
        dim: i32,
        vect: &mut Vec<u8>,
    ) -> Result<()> {
        if array_2d.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }

        let num_vectors = env.get_array_length(array_2d)?;
        self.has_exception_in_stack(env)?;

        for i in 0..num_vectors {
            let vector_array_obj = env.get_object_array_element(array_2d, i)?;
            self.has_exception_in_stack_with_message(env, "Unable to get object array element")?;

            let vector_array = JByteArray::from(vector_array_obj);
            let arr_len = env.get_array_length(&vector_array)?;
            if dim != arr_len {
                return Err(JniUtilError::Runtime(
                    "Dimension of vectors is inconsistent".to_string(),
                ));
            }

            let mut buf = vec![0i8; dim as usize];
            env.get_byte_array_region(&vector_array, 0, &mut buf)?;

            // Reinterpret i8 as u8
            let ubuf: Vec<u8> = buf.into_iter().map(|b| b as u8).collect();
            vect.extend_from_slice(&ubuf);
        }

        self.has_exception_in_stack(env)?;
        Ok(())
    }

    /// Convert a 2D Java byte[][] to a flat Vec<i8> (signed byte vectors).
    pub fn convert_2d_java_object_array_and_store_to_byte_vector(
        &self,
        env: &mut JNIEnv,
        array_2d: &JObjectArray,
        dim: i32,
        vect: &mut Vec<i8>,
    ) -> Result<()> {
        if array_2d.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }

        let num_vectors = env.get_array_length(array_2d)?;
        self.has_exception_in_stack_with_message(env, "Unable to get array length")?;

        for i in 0..num_vectors {
            let vector_array_obj = env.get_object_array_element(array_2d, i)?;
            self.has_exception_in_stack_with_message(env, "Unable to get object array element")?;

            let vector_array = JByteArray::from(vector_array_obj);
            let arr_len = env.get_array_length(&vector_array)?;
            if dim != arr_len {
                return Err(JniUtilError::Runtime(
                    "Dimension of vectors is inconsistent".to_string(),
                ));
            }

            let mut buf = vec![0i8; dim as usize];
            env.get_byte_array_region(&vector_array, 0, &mut buf)?;

            vect.extend_from_slice(&buf);
        }

        self.has_exception_in_stack(env)?;
        Ok(())
    }

    /// Convert a Java int[] to Vec<i64> (widening from int to long, matching C++ behavior).
    pub fn convert_java_int_array_to_i64_vector(
        &self,
        env: &mut JNIEnv,
        array: &JIntArray,
    ) -> Result<Vec<i64>> {
        if array.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }

        let num_elements = env.get_array_length(array)?;
        self.has_exception_in_stack_with_message(env, "Unable to get array length")?;

        let mut buf = vec![0i32; num_elements as usize];
        env.get_int_array_region(array, 0, &mut buf)?;

        let result: Vec<i64> = buf.into_iter().map(|x| x as i64).collect();
        Ok(result)
    }

    // ======================== MISC HELPERS ========================

    /// Get the inner dimension of a 2D Java float array (float[][]).
    pub fn get_inner_dimension_of_2d_java_float_array(
        &self,
        env: &mut JNIEnv,
        array_2d: &JObjectArray,
    ) -> Result<i32> {
        if array_2d.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }

        let outer_len = env.get_array_length(array_2d)?;
        if outer_len <= 0 {
            return Ok(0);
        }

        let vector_array_obj = env.get_object_array_element(array_2d, 0)?;
        self.has_exception_in_stack(env)?;
        let vector_array = JFloatArray::from(vector_array_obj);
        let dim = env.get_array_length(&vector_array)?;
        self.has_exception_in_stack(env)?;
        Ok(dim)
    }

    /// Get the inner dimension of a 2D Java byte array (byte[][]).
    pub fn get_inner_dimension_of_2d_java_byte_array(
        &self,
        env: &mut JNIEnv,
        array_2d: &JObjectArray,
    ) -> Result<i32> {
        if array_2d.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }

        let outer_len = env.get_array_length(array_2d)?;
        if outer_len <= 0 {
            return Ok(0);
        }

        let vector_array_obj = env.get_object_array_element(array_2d, 0)?;
        self.has_exception_in_stack(env)?;
        let vector_array = JByteArray::from(vector_array_obj);
        let dim = env.get_array_length(&vector_array)?;
        self.has_exception_in_stack(env)?;
        Ok(dim)
    }

    /// Get the length of a Java Object[].
    pub fn get_java_object_array_length(
        &self,
        env: &mut JNIEnv,
        array: &JObjectArray,
    ) -> Result<i32> {
        if array.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }
        let length = env.get_array_length(array)?;
        self.has_exception_in_stack_with_message(env, "Unable to get array length")?;
        Ok(length)
    }

    /// Get the length of a Java int[].
    pub fn get_java_int_array_length(&self, env: &mut JNIEnv, array: &JIntArray) -> Result<i32> {
        if array.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }
        let length = env.get_array_length(array)?;
        self.has_exception_in_stack_with_message(env, "Unable to get array length")?;
        Ok(length)
    }

    /// Get the length of a Java long[].
    pub fn get_java_long_array_length(
        &self,
        env: &mut JNIEnv,
        array: &JLongArray,
    ) -> Result<i32> {
        if array.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }
        let length = env.get_array_length(array)?;
        self.has_exception_in_stack_with_message(env, "Unable to get array length")?;
        Ok(length)
    }

    /// Get the length of a Java byte[].
    pub fn get_java_bytes_array_length(
        &self,
        env: &mut JNIEnv,
        array: &JByteArray,
    ) -> Result<i32> {
        if array.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }
        let length = env.get_array_length(array)?;
        self.has_exception_in_stack_with_message(env, "Unable to get array length")?;
        Ok(length)
    }

    /// Get the length of a Java float[].
    pub fn get_java_float_array_length(
        &self,
        env: &mut JNIEnv,
        array: &JFloatArray,
    ) -> Result<i32> {
        if array.is_null() {
            return Err(JniUtilError::NullPointer(
                "Array cannot be null".to_string(),
            ));
        }
        let length = env.get_array_length(array)?;
        self.has_exception_in_stack_with_message(env, "Unable to get array length")?;
        Ok(length)
    }

    // ======================== DIRECT JNI ENV WRAPPERS ========================

    /// Delete a local reference.
    pub fn delete_local_ref(&self, env: &mut JNIEnv, obj: JObject) -> Result<()> {
        env.delete_local_ref(obj)?;
        Ok(())
    }

    /// Get byte array elements as a Vec<i8>.
    pub fn get_byte_array_elements(
        &self,
        env: &mut JNIEnv,
        array: &JByteArray,
    ) -> Result<Vec<i8>> {
        let len = env.get_array_length(array)?;
        let mut buf = vec![0i8; len as usize];
        env.get_byte_array_region(array, 0, &mut buf)?;
        Ok(buf)
    }

    /// Get float array elements as a Vec<f32>.
    pub fn get_float_array_elements(
        &self,
        env: &mut JNIEnv,
        array: &JFloatArray,
    ) -> Result<Vec<f32>> {
        let len = env.get_array_length(array)?;
        let mut buf = vec![0.0f32; len as usize];
        env.get_float_array_region(array, 0, &mut buf)?;
        Ok(buf)
    }

    /// Get int array elements as a Vec<i32>.
    pub fn get_int_array_elements(
        &self,
        env: &mut JNIEnv,
        array: &JIntArray,
    ) -> Result<Vec<i32>> {
        let len = env.get_array_length(array)?;
        let mut buf = vec![0i32; len as usize];
        env.get_int_array_region(array, 0, &mut buf)?;
        Ok(buf)
    }

    /// Get long array elements as a Vec<i64>.
    pub fn get_long_array_elements(
        &self,
        env: &mut JNIEnv,
        array: &JLongArray,
    ) -> Result<Vec<i64>> {
        let len = env.get_array_length(array)?;
        let mut buf = vec![0i64; len as usize];
        env.get_long_array_region(array, 0, &mut buf)?;
        Ok(buf)
    }

    /// Get an element from a Java Object[].
    pub fn get_object_array_element<'local>(
        &self,
        env: &mut JNIEnv<'local>,
        array: &JObjectArray,
        index: jsize,
    ) -> Result<JObject<'local>> {
        let obj = env.get_object_array_element(array, index)?;
        self.has_exception_in_stack_with_message(env, "Unable to get object")?;
        Ok(obj)
    }

    /// Create a new Java object (KNNQueryResult constructor pattern).
    pub fn new_knn_query_result<'local>(
        &self,
        env: &mut JNIEnv<'local>,
        id: i32,
        distance: f32,
    ) -> Result<JObject<'local>> {
        let class_ref = self.find_class(env, "org/opensearch/knn/index/query/KNNQueryResult")?;
        let method_id = self.find_method(
            env,
            "org/opensearch/knn/index/query/KNNQueryResult",
            "<init>",
        )?;
        let class = JClass::from(unsafe { JObject::from_raw(class_ref.as_obj().as_raw()) });

        let obj = unsafe {
            env.new_object_unchecked(
                &class,
                method_id,
                &[
                    jni::sys::jvalue { i: id },
                    jni::sys::jvalue { f: distance },
                ],
            )?
        };

        if obj.is_null() {
            self.has_exception_in_stack_with_message(env, "Unable to create object")?;
            return Err(JniUtilError::Runtime("Unable to create object".to_string()));
        }

        Ok(obj)
    }

    /// Create a new Java Object[].
    pub fn new_object_array<'local>(
        &self,
        env: &mut JNIEnv<'local>,
        len: jsize,
        class: &JClass,
        init: &JObject,
    ) -> Result<JObjectArray<'local>> {
        let array = env.new_object_array(len, class, init)?;
        self.has_exception_in_stack_with_message(env, "Unable to allocate object array")?;
        Ok(array)
    }

    /// Create a new Java byte[].
    pub fn new_byte_array<'local>(
        &self,
        env: &mut JNIEnv<'local>,
        len: jsize,
    ) -> Result<JByteArray<'local>> {
        let array = env.new_byte_array(len)?;
        self.has_exception_in_stack_with_message(env, "Unable to allocate byte array")?;
        Ok(array)
    }

    /// Set an element in a Java Object[].
    pub fn set_object_array_element(
        &self,
        env: &mut JNIEnv,
        array: &JObjectArray,
        index: jsize,
        val: &JObject,
    ) -> Result<()> {
        env.set_object_array_element(array, index, val)?;
        self.has_exception_in_stack_with_message(env, "Unable to set object array element")?;
        Ok(())
    }

    /// Set a region in a Java byte[].
    pub fn set_byte_array_region(
        &self,
        env: &mut JNIEnv,
        array: &JByteArray,
        start: jsize,
        buf: &[i8],
    ) -> Result<()> {
        env.set_byte_array_region(array, start, buf)?;
        self.has_exception_in_stack_with_message(env, "Unable to set byte array region")?;
        Ok(())
    }

    /// Get an object field from a Java object.
    pub fn get_object_field<'local>(
        &self,
        env: &mut JNIEnv<'local>,
        obj: &JObject,
        field_id: jfieldID,
    ) -> Result<JObject<'local>> {
        // Safety: field_id must be valid for the given object.
        // Convert the raw jfieldID pointer to a JFieldID wrapper for the jni crate API.
        unsafe {
            let jfield_id = JFieldID::from_raw(field_id);
            let raw = env.get_field_unchecked(
                obj,
                jfield_id,
                jni::signature::ReturnType::Object,
            )?;
            Ok(raw.l()?)
        }
    }

    /// Find a class from JNIEnv (not cached) and return a global reference.
    pub fn find_class_from_jni_env(
        &self,
        env: &mut JNIEnv,
        name: &str,
    ) -> Result<GlobalRef> {
        let local_class = env.find_class(name)?;
        let global_ref = env.new_global_ref(local_class)?;
        Ok(global_ref)
    }

    /// Get a method ID from JNIEnv.
    pub fn get_method_id(
        &self,
        env: &mut JNIEnv,
        class: &JClass,
        name: &str,
        sig: &str,
    ) -> Result<JMethodID> {
        let method_id = env.get_method_id(class, name, sig)?;
        Ok(method_id)
    }

    /// Get a field ID from JNIEnv.
    pub fn get_field_id(
        &self,
        env: &mut JNIEnv,
        class: &JClass,
        name: &str,
        sig: &str,
    ) -> Result<jfieldID> {
        let field_id = env.get_field_id(class, name, sig)?;
        // The jni crate returns JFieldID; we need the raw pointer for storage
        Ok(field_id.into_raw())
    }

    /// Call a non-virtual int method.
    /// Note: jni 0.21 does not expose `call_nonvirtual_method_unchecked`.
    /// Gated until a raw JNI implementation is added.
    #[cfg(any())]
    pub fn call_nonvirtual_int_method(
        &self,
        env: &mut JNIEnv,
        obj: &JObject,
        class: &JClass,
        method_id: JMethodID,
        args: &[jvalue],
    ) -> Result<jint> {
        let result = unsafe {
            env.call_nonvirtual_method_unchecked(
                obj,
                class,
                method_id,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Int),
                args,
            )?
        };
        Ok(result.i()?)
    }

    /// Call a non-virtual long method.
    /// Note: jni 0.21 does not expose `call_nonvirtual_method_unchecked`.
    /// Gated until a raw JNI implementation is added.
    #[cfg(any())]
    pub fn call_nonvirtual_long_method(
        &self,
        env: &mut JNIEnv,
        obj: &JObject,
        class: &JClass,
        method_id: JMethodID,
        args: &[jvalue],
    ) -> Result<jlong> {
        let result = unsafe {
            env.call_nonvirtual_method_unchecked(
                obj,
                class,
                method_id,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Long),
                args,
            )?
        };
        Ok(result.j()?)
    }

    /// Call a non-virtual void method.
    /// Note: jni 0.21 does not expose `call_nonvirtual_method_unchecked`.
    /// Gated until a raw JNI implementation is added.
    #[cfg(any())]
    pub fn call_nonvirtual_void_method(
        &self,
        env: &mut JNIEnv,
        obj: &JObject,
        class: &JClass,
        method_id: JMethodID,
        args: &[jvalue],
    ) -> Result<()> {
        unsafe {
            env.call_nonvirtual_method_unchecked(
                obj,
                class,
                method_id,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Void),
                args,
            )?;
        }
        Ok(())
    }

    /// Get primitive array critical (unsafe, pins the array).
    /// The caller MUST call release_primitive_array_critical when done.
    ///
    /// # Safety
    /// The returned pointer is only valid until release_primitive_array_critical is called.
    /// No JNI calls (other than Get/ReleasePrimitiveArrayCritical) may be made while
    /// the array is pinned.
    pub unsafe fn get_primitive_array_critical(
        &self,
        env: &JNIEnv,
        array: jarray,
    ) -> Result<*mut std::ffi::c_void> {
        // Access the raw JNI env pointer for critical array operations
        // which are not directly exposed by the jni crate's safe API
        let raw_env = env.get_raw();
        let ptr = (**raw_env).GetPrimitiveArrayCritical.unwrap()(raw_env, array, std::ptr::null_mut());
        if ptr.is_null() {
            return Err(JniUtilError::Runtime(
                "Unable to get primitive array critical".to_string(),
            ));
        }
        Ok(ptr)
    }

    /// Release primitive array critical.
    ///
    /// # Safety
    /// Must be called with the same array and pointer from get_primitive_array_critical.
    pub unsafe fn release_primitive_array_critical(
        &self,
        env: &JNIEnv,
        array: jarray,
        carray: *mut std::ffi::c_void,
        mode: jint,
    ) {
        let raw_env = env.get_raw();
        (**raw_env).ReleasePrimitiveArrayCritical.unwrap()(raw_env, array, carray, mode);
    }

    // ======================== PRIVATE HELPERS ========================

    fn cache_class(&mut self, env: &mut JNIEnv, class_name: &str) -> Result<()> {
        let class = env.find_class(class_name)?;
        let global_ref = env.new_global_ref(class)?;
        self.cached_classes.insert(class_name.to_string(), global_ref);
        Ok(())
    }

    fn cache_method(
        &mut self,
        env: &mut JNIEnv,
        class_name: &str,
        method_name: &str,
        sig: &str,
    ) -> Result<()> {
        let class_ref = self
            .cached_classes
            .get(class_name)
            .ok_or_else(|| JniUtilError::Runtime(format!("Class {} not cached", class_name)))?;
        let class = JClass::from(unsafe { JObject::from_raw(class_ref.as_obj().as_raw()) });
        let method_id = env.get_method_id(&class, method_name, sig)?;
        let key = format!("{}:{}", class_name, method_name);
        self.cached_methods.insert(key, method_id);
        Ok(())
    }

    fn cache_static_method(
        &mut self,
        env: &mut JNIEnv,
        class_name: &str,
        method_name: &str,
        sig: &str,
    ) -> Result<()> {
        let class_ref = self
            .cached_classes
            .get(class_name)
            .ok_or_else(|| JniUtilError::Runtime(format!("Class {} not cached", class_name)))?;
        let class = JClass::from(unsafe { JObject::from_raw(class_ref.as_obj().as_raw()) });
        let method_id = env.get_static_method_id(&class, method_name, sig)?;
        let key = format!("{}:{}", class_name, method_name);
        // Store static method ID - in the jni crate these are different types,
        // but we store the raw value. For the purpose of this port, we use a separate
        // storage or cast. Using JMethodID storage with unsafe transmute for now.
        // TODO: Consider separate storage for static method IDs.
        let raw_method_id: JMethodID = unsafe { std::mem::transmute(method_id) };
        self.cached_methods.insert(key, raw_method_id);
        Ok(())
    }
}

impl Default for JNIUtil {
    fn default() -> Self {
        Self::new()
    }
}

// ======================== FREE FUNCTIONS ========================

/// Get a JObject from a HashMap or return an error (analogous to C++ GetJObjectFromMapOrThrow).
pub fn get_jobject_from_map_or_throw<'a>(
    map: &'a HashMap<String, JObject<'a>>,
    key: &str,
) -> Result<&'a JObject<'a>> {
    map.get(key).ok_or_else(|| {
        JniUtilError::Runtime(format!("{} not found", key))
    })
}

/// Macro for panic-catching boundary at JNI entry points.
/// Wraps the body in catch_unwind and throws a Java exception on panic.
#[macro_export]
macro_rules! jni_panic_boundary {
    ($env:expr, $jni_util:expr, $default:expr, $body:block) => {{
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        match result {
            Ok(val) => val,
            Err(panic_info) => {
                $jni_util.catch_and_throw_java($env, panic_info);
                $default
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(FAISS_NAME, "faiss");
        assert_eq!(NMSLIB_NAME, "nmslib");
        assert_eq!(L2, "l2");
        assert_eq!(INNER_PRODUCT, "innerproduct");
        assert_eq!(EF_SEARCH, "ef_search");
    }

    #[test]
    fn test_quantization_level_enum() {
        assert_ne!(BQQuantizationLevel::OneBit, BQQuantizationLevel::TwoBit);
        assert_eq!(BQQuantizationLevel::FourBit, BQQuantizationLevel::FourBit);
    }

    #[test]
    fn test_jni_util_default() {
        let util = JNIUtil::default();
        assert!(util.cached_classes.is_empty());
        assert!(util.cached_methods.is_empty());
    }

    // -----------------------------------------------------------------------
    // Ported from C++ tests: BQQuantizationLevel enum value tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bq_quantization_level_all_variants() {
        // Verify all enum variants exist and are distinct
        let one = BQQuantizationLevel::OneBit;
        let two = BQQuantizationLevel::TwoBit;
        let four = BQQuantizationLevel::FourBit;
        let none = BQQuantizationLevel::None;

        assert_ne!(one, two);
        assert_ne!(one, four);
        assert_ne!(one, none);
        assert_ne!(two, four);
        assert_ne!(two, none);
        assert_ne!(four, none);
    }

    #[test]
    fn test_bq_quantization_level_clone_and_copy() {
        let level = BQQuantizationLevel::OneBit;
        let cloned = level.clone();
        let copied = level;
        assert_eq!(level, cloned);
        assert_eq!(level, copied);
    }

    #[test]
    fn test_bq_quantization_level_debug() {
        // Verify Debug trait is implemented
        let level = BQQuantizationLevel::OneBit;
        let debug_str = format!("{:?}", level);
        assert_eq!(debug_str, "OneBit");

        let level = BQQuantizationLevel::TwoBit;
        let debug_str = format!("{:?}", level);
        assert_eq!(debug_str, "TwoBit");

        let level = BQQuantizationLevel::FourBit;
        let debug_str = format!("{:?}", level);
        assert_eq!(debug_str, "FourBit");

        let level = BQQuantizationLevel::None;
        let debug_str = format!("{:?}", level);
        assert_eq!(debug_str, "None");
    }

    // -----------------------------------------------------------------------
    // Constants correctness tests (comprehensive, ported from jni_util.h)
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
    fn test_index_parameter_constants() {
        assert_eq!(NPROBES, "nprobes");
        assert_eq!(COARSE_QUANTIZER, "coarse_quantizer");
        assert_eq!(M, "m");
        assert_eq!(M_NMSLIB, "M");
        assert_eq!(EF_CONSTRUCTION, "ef_construction");
        assert_eq!(EF_CONSTRUCTION_NMSLIB, "efConstruction");
        assert_eq!(EF_SEARCH, "ef_search");
    }

    #[test]
    fn test_java_knn_constants_strings() {
        assert_eq!(SPACE_TYPE_FAISS_INDEX_JAVA_KNN_CONSTANTS, "space_type");
        assert_eq!(
            QUANTIZATION_LEVEL_FAISS_INDEX_LOAD_PARAMETER_JAVA_KNN_CONSTANTS,
            "quantization_level"
        );
    }

    #[test]
    fn test_misc_constants() {
        assert_eq!(SPACE_TYPE, "spaceType");
        assert_eq!(METHOD, "method");
        assert_eq!(INDEX_DESCRIPTION, "index_description");
        assert_eq!(PARAMETERS, "parameters");
        assert_eq!(TRAINING_DATASET_SIZE_LIMIT, "training_dataset_size_limit");
        assert_eq!(INDEX_THREAD_QUANTITY, "indexThreadQty");
        assert_eq!(ILLEGAL_ARGUMENT_PATH, "java/lang/IllegalArgumentException");
    }

    // -----------------------------------------------------------------------
    // JNIUtil struct tests (pure logic, no JVM needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_jni_util_new_is_empty() {
        let util = JNIUtil::new();
        assert!(util.cached_classes.is_empty());
        assert!(util.cached_methods.is_empty());
        assert!(util.vm.is_none());
    }

    #[test]
    fn test_jni_util_uninitialize_clears() {
        let mut util = JNIUtil::new();
        // Nothing to clear but should not panic
        util.uninitialize();
        assert!(util.cached_classes.is_empty());
        assert!(util.cached_methods.is_empty());
        assert!(util.vm.is_none());
    }

    #[test]
    fn test_jni_util_get_jni_current_env_returns_none_without_vm() {
        let util = JNIUtil::new();
        // Without a JVM, should return None
        assert!(util.get_jni_current_env().is_none());
    }

    #[test]
    fn test_jni_util_find_class_not_cached_returns_error() {
        // JNIUtil without initialization -- no cached classes
        let util = JNIUtil::new();
        // We cannot call find_class without a real JNIEnv, but we can test
        // the internal logic by checking cached_classes directly
        assert!(!util.cached_classes.contains_key("java/lang/Integer"));
    }

    // -----------------------------------------------------------------------
    // JniUtilError tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jni_util_error_display() {
        let err = JniUtilError::Runtime("test error".to_string());
        assert_eq!(err.to_string(), "test error");

        let err = JniUtilError::NullPointer("null arg".to_string());
        assert_eq!(err.to_string(), "Null pointer: null arg");
    }

    // -----------------------------------------------------------------------
    // get_jobject_from_map_or_throw tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_jobject_from_map_or_throw_key_not_found() {
        let map: HashMap<String, JObject> = HashMap::new();
        let result = get_jobject_from_map_or_throw(&map, "missing_key");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("missing_key not found"));
    }

    // -----------------------------------------------------------------------
    // JniReleaseGuard tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jni_release_guard_executes_on_drop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let was_called = Arc::new(AtomicBool::new(false));
        let was_called_clone = was_called.clone();

        {
            let _guard = JniReleaseGuard::new(move || {
                was_called_clone.store(true, Ordering::SeqCst);
            });
        }
        // After the guard goes out of scope, the closure should have been called
        assert!(was_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_jni_release_guard_handles_panic_in_closure() {
        // The guard should not propagate panics from the release function
        {
            let _guard = JniReleaseGuard::new(|| {
                panic!("intentional panic in release");
            });
        }
        // If we reach here, the panic was caught
    }
}
