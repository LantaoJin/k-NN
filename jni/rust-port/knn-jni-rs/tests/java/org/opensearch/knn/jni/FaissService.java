// SPDX-License-Identifier: Apache-2.0
// Minimal FaissService class for integration testing the Rust JNI library.
// Only declares the native methods we want to test.

package org.opensearch.knn.jni;

public class FaissService {
    static {
        System.loadLibrary("opensearchknn_faiss");
    }

    // Index lifecycle
    public static native void initLibrary();
    public static native void free(long indexPointer, boolean isBinaryIndex);
    public static native boolean isSharedIndexStateRequired(long indexPointer);

    // Load index from file path
    public static native long loadIndex(String indexPath);
    public static native long loadBinaryIndex(String indexPath);
}
