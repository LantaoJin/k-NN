package org.opensearch.knn.jni;

public class FaissService {
    static { System.loadLibrary("opensearchknn_faiss"); }

    public static native void initLibrary();
    public static native long initIndex(long numDocs, int dim, java.util.Map<String, Object> parameters);
    public static native void insertToIndex(int[] ids, long vectorsAddress, int dim, long indexAddress, int threadCount);
    public static native void free(long indexPointer, boolean isBinaryIndex);
    public static native boolean isSharedIndexStateRequired(long indexPointer);
    public static native long loadIndex(String indexPath);

    // Query
    public static native org.opensearch.knn.index.query.KNNQueryResult[] queryIndex(
        long indexPointer, float[] queryVector, int k, java.util.Map<String, Object> methodParams, int[] parentIds);
}
