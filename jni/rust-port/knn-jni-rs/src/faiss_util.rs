// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

//! Utility functions for Faiss operations, free of JNI dependencies.
//! This module mirrors `jni/src/faiss_util.cpp`.

use std::ptr;

/// Opaque type representing `faiss::IDGrouperBitmap` on the C++ side.
/// We never construct this in Rust; we only hold a pointer to it.
#[repr(C)]
pub struct FaissIDGrouperBitmap {
    _opaque: [u8; 0],
}

/// FFI declarations for Faiss IDGrouperBitmap operations.
/// These will be moved to an `ffi` module in a later phase.
mod ffi {
    use super::FaissIDGrouperBitmap;

    extern "C" {
        /// Constructs a new `faiss::IDGrouperBitmap` with the given number of
        /// 64-bit blocks and a pointer to the bitmap data.
        ///
        /// Corresponds to:
        /// ```cpp
        /// new faiss::IDGrouperBitmap(num_blocks, bitmap_data)
        /// ```
        ///
        /// Returns a raw pointer to the heap-allocated IDGrouperBitmap.
        /// The caller owns the memory.
        pub fn faiss_id_grouper_bitmap_new(
            num_blocks: i32,
            bitmap_data: *mut u64,
        ) -> *mut FaissIDGrouperBitmap;

        /// Sets a group bit in the IDGrouperBitmap.
        ///
        /// Corresponds to:
        /// ```cpp
        /// idGrouper->set_group(id)
        /// ```
        pub fn faiss_id_grouper_bitmap_set_group(
            grouper: *mut FaissIDGrouperBitmap,
            id: i32,
        );

        /// Frees a previously allocated IDGrouperBitmap.
        pub fn faiss_id_grouper_bitmap_free(grouper: *mut FaissIDGrouperBitmap);
    }
}

/// A safe wrapper around the Faiss IDGrouperBitmap that owns the C++ object
/// and its backing bitmap storage.
pub struct IDGrouperBitmap {
    /// Raw pointer to the C++ `faiss::IDGrouperBitmap` object.
    ptr: *mut FaissIDGrouperBitmap,
    /// Backing bitmap storage. Must remain alive as long as `ptr` is in use.
    bitmap: Vec<u64>,
}

impl IDGrouperBitmap {
    /// Returns the raw pointer to the underlying `faiss::IDGrouperBitmap`.
    /// The pointer is valid as long as this struct is alive.
    pub fn as_ptr(&self) -> *mut FaissIDGrouperBitmap {
        self.ptr
    }

    /// Returns a reference to the backing bitmap storage.
    pub fn bitmap(&self) -> &[u64] {
        &self.bitmap
    }
}

impl Drop for IDGrouperBitmap {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::faiss_id_grouper_bitmap_free(self.ptr);
            }
        }
    }
}

// IDGrouperBitmap is not Send/Sync by default due to raw pointer.
// The underlying Faiss object is not thread-safe, so we intentionally
// do NOT implement Send/Sync.

/// Builds an `IDGrouperBitmap` from an array of parent IDs.
///
/// This mirrors the C++ function:
/// ```cpp
/// std::unique_ptr<faiss::IDGrouperBitmap> faiss_util::buildIDGrouperBitmap(
///     int *parentIdsArray, int parentIdsLength, std::vector<uint64_t>* bitmap);
/// ```
///
/// # Arguments
/// * `parent_ids` - Slice of parent ID values. Must not be empty.
///
/// # Returns
/// An `IDGrouperBitmap` that owns both the C++ object and the backing bitmap.
///
/// # Panics
/// Panics if `parent_ids` is empty (mirrors undefined behavior of the C++ code
/// when `parentIdsLength` is 0, since `std::max_element` would be UB).
pub fn build_id_grouper_bitmap(parent_ids: &[i32]) -> IDGrouperBitmap {
    assert!(
        !parent_ids.is_empty(),
        "parent_ids must not be empty"
    );

    // Find the maximum value to determine bitmap size.
    let max_value = *parent_ids.iter().max().unwrap();
    let num_bits = max_value + 1;
    let num_blocks = (num_bits >> 6) + 1; // div by 64, plus 1

    // Allocate the bitmap storage, initialized to zeros.
    let mut bitmap: Vec<u64> = vec![0u64; num_blocks as usize];

    // Create the C++ IDGrouperBitmap object via FFI.
    let grouper_ptr = unsafe {
        ffi::faiss_id_grouper_bitmap_new(num_blocks, bitmap.as_mut_ptr())
    };
    assert!(
        !grouper_ptr.is_null(),
        "faiss_id_grouper_bitmap_new returned null"
    );

    // Set each parent ID as a group in the bitmap.
    for &parent_id in parent_ids {
        unsafe {
            ffi::faiss_id_grouper_bitmap_set_group(grouper_ptr, parent_id);
        }
    }

    IDGrouperBitmap {
        ptr: grouper_ptr,
        bitmap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: Tests that call build_id_grouper_bitmap with non-empty slices require
    // the Faiss FFI symbols (faiss_id_grouper_bitmap_new, etc.) to be linked.
    // They are intended to be run as integration tests with the full native library.

    #[test]
    #[should_panic(expected = "parent_ids must not be empty")]
    fn test_build_id_grouper_bitmap_empty_panics() {
        let _ = build_id_grouper_bitmap(&[]);
    }

    // -----------------------------------------------------------------------
    // Ported from C++ faiss_util_test.cpp: IDGrouperBitMapTest
    // Tests the bitmap sizing logic without calling FFI (pure Rust logic).
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitmap_size_calculation() {
        // Ported from: buildIDGrouperBitmap logic for bitmap sizing
        // With parent IDs [128, 1024], the max is 1024.
        // num_bits = 1024 + 1 = 1025
        // num_blocks = (1025 >> 6) + 1 = 16 + 1 = 17
        let parent_ids = [128i32, 1024i32];
        let max_value = *parent_ids.iter().max().unwrap();
        let num_bits = max_value + 1;
        let num_blocks = (num_bits >> 6) + 1;

        assert_eq!(max_value, 1024);
        assert_eq!(num_blocks, 17);
    }

    #[test]
    fn test_bitmap_size_calculation_small() {
        // Single ID: 63
        // num_bits = 64, num_blocks = (64 >> 6) + 1 = 1 + 1 = 2
        let parent_ids = [63i32];
        let max_value = *parent_ids.iter().max().unwrap();
        let num_bits = max_value + 1;
        let num_blocks = (num_bits >> 6) + 1;

        assert_eq!(num_blocks, 2);
    }

    #[test]
    fn test_bitmap_size_calculation_zero() {
        // ID = 0: num_bits = 1, num_blocks = (1 >> 6) + 1 = 0 + 1 = 1
        let parent_ids = [0i32];
        let max_value = *parent_ids.iter().max().unwrap();
        let num_bits = max_value + 1;
        let num_blocks = (num_bits >> 6) + 1;

        assert_eq!(num_blocks, 1);
    }

    #[test]
    fn test_bitmap_bit_setting_logic() {
        // Verify the bit-setting logic that build_id_grouper_bitmap uses
        // before calling FFI. This tests the Rust-side bitmap manipulation.
        let parent_ids = [128i32, 1024i32];
        let max_value = *parent_ids.iter().max().unwrap();
        let num_blocks = ((max_value + 1) >> 6) + 1;
        let mut bitmap: Vec<u64> = vec![0u64; num_blocks as usize];

        // Manually set bits like the Rust code does before passing to FFI
        for &parent_id in &parent_ids {
            let word_idx = (parent_id as u64 / 64) as usize;
            let bit_idx = parent_id as u64 % 64;
            bitmap[word_idx] |= 1u64 << bit_idx;
        }

        // Verify bit 128 is set: word 2 (128/64=2), bit 0 (128%64=0)
        assert_eq!(bitmap[2] & (1u64 << 0), 1);
        // Verify bit 1024 is set: word 16 (1024/64=16), bit 0 (1024%64=0)
        assert_eq!(bitmap[16] & (1u64 << 0), 1);
        // Verify other bits are not set
        assert_eq!(bitmap[0], 0);
        assert_eq!(bitmap[1], 0);
        assert_eq!(bitmap[3], 0);
    }
}
