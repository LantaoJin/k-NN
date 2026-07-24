package org.opensearch.knn.jni;

public class JNICommons {
    static {
        System.loadLibrary("opensearchknn_util");
        System.loadLibrary("opensearchknn_common");
    }
    public static native long storeVectorData(long memoryAddress, float[][] data, long totalNumberOfVectors, boolean append);
    public static native void freeVectorData(long memoryAddress);
}
