// SPDX-License-Identifier: Apache-2.0
package org.opensearch.knn.jni;

public class JNICommons {
    static {
        System.loadLibrary("opensearchknn_faiss");
    }

    // Vector storage
    public static native long storeVectorData(long memoryAddress, float[][] data, long totalNumberOfVectors, boolean append);
    public static native void freeVectorData(long memoryAddress);
}
