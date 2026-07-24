// SPDX-License-Identifier: Apache-2.0
//
// Full benchmark comparing C++ JNI vs Rust JNI performance.
// Covers: JNI overhead, vector storage, index creation, query, and free.
//
// Usage:
//   cd jni/rust-port/knn-jni-rs/benches/java
//   javac org/opensearch/knn/jni/*.java org/opensearch/knn/index/query/*.java org/apache/lucene/index/*.java
//   DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:<release_dir> \
//     java -Djava.library.path=<release_dir> -cp . org.opensearch.knn.jni.NativeJniBenchmark

package org.opensearch.knn.jni;

import org.opensearch.knn.index.query.KNNQueryResult;
import java.util.HashMap;
import java.util.Map;

public class NativeJniBenchmark {

    static final int WARMUP = 50;
    static final int ITERATIONS = 500;
    static final int DIM = 128;
    static final int NUM_VECTORS = 1000;

    public static void main(String[] args) {
        System.out.println("╔═══════════════════════════════════════════════════════════════╗");
        System.out.println("║        OpenSearch k-NN JNI Performance Benchmark             ║");
        System.out.println("╠═══════════════════════════════════════════════════════════════╣");
        System.out.println("║  Vectors: " + NUM_VECTORS + " x " + DIM + "d | Warmup: " + WARMUP + " | Measure: " + ITERATIONS + "          ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════╝");
        System.out.println();

        // 1. JNI boundary overhead
        bench("initLibrary (no-op JNI overhead)", ITERATIONS * 2, () -> {
            FaissService.initLibrary();
        });

        // 2. Vector storage
        float[][] vectors = generateVectors(NUM_VECTORS, DIM);
        bench("storeVectorData (" + NUM_VECTORS + " x " + DIM + "d)", ITERATIONS, () -> {
            long addr = JNICommons.storeVectorData(0, vectors, (long) NUM_VECTORS * DIM, true);
            JNICommons.freeVectorData(addr);
        });

        // 3. Store with append (batched)
        float[][] batch = generateVectors(100, DIM);
        bench("storeVectorData append (10x100)", ITERATIONS, () -> {
            long addr = 0;
            for (int i = 0; i < 10; i++) {
                addr = JNICommons.storeVectorData(addr, batch, (long) 1000 * DIM, true);
            }
            JNICommons.freeVectorData(addr);
        });

        // 4. Index creation (initIndex + insertToIndex)
        Map<String, Object> params = new HashMap<>();
        params.put("spaceType", "l2");
        params.put("index_description", "HNSW32,Flat");
        Map<String, Object> subParams = new HashMap<>();
        subParams.put("ef_construction", 128);
        subParams.put("m", 16);
        params.put("parameters", subParams);

        bench("initIndex + insertToIndex (" + NUM_VECTORS + " vectors)", 20, () -> {
            long vecAddr = JNICommons.storeVectorData(0, vectors, (long) NUM_VECTORS * DIM, true);
            int[] ids = generateIds(NUM_VECTORS);
            long indexAddr = FaissService.initIndex(0, DIM, params);
            FaissService.insertToIndex(ids, vecAddr, DIM, indexAddr, 1);
            FaissService.free(indexAddr, false);
            JNICommons.freeVectorData(vecAddr);
        });

        // 5. Query (build index once, query many times)
        long vecAddr = JNICommons.storeVectorData(0, vectors, (long) NUM_VECTORS * DIM, true);
        int[] ids = generateIds(NUM_VECTORS);
        long indexAddr = FaissService.initIndex(0, DIM, params);
        FaissService.insertToIndex(ids, vecAddr, DIM, indexAddr, 1);

        float[] query = new float[DIM];
        for (int d = 0; d < DIM; d++) query[d] = (float) Math.sin(d * 0.01);

        bench("queryIndex k=10 (pre-built index)", ITERATIONS, () -> {
            KNNQueryResult[] results = FaissService.queryIndex(indexAddr, query, 10, null, null);
        });

        bench("queryIndex k=100 (pre-built index)", ITERATIONS, () -> {
            KNNQueryResult[] results = FaissService.queryIndex(indexAddr, query, 100, null, null);
        });

        // 6. Free
        bench("free (non-null index)", 20, () -> {
            long vAddr = JNICommons.storeVectorData(0, vectors, (long) NUM_VECTORS * DIM, true);
            int[] docIds = generateIds(NUM_VECTORS);
            long idx = FaissService.initIndex(0, DIM, params);
            FaissService.insertToIndex(docIds, vAddr, DIM, idx, 1);
            FaissService.free(idx, false);
            JNICommons.freeVectorData(vAddr);
        });

        bench("free(0) null guard", ITERATIONS * 2, () -> {
            FaissService.free(0, false);
        });

        // 7. isSharedIndexStateRequired
        bench("isSharedIndexStateRequired(0)", ITERATIONS * 2, () -> {
            FaissService.isSharedIndexStateRequired(0);
        });

        // Cleanup
        FaissService.free(indexAddr, false);
        JNICommons.freeVectorData(vecAddr);

        System.out.println("\nDone.");
    }

    // ─────────────────────────────────────────────────────────────────────────

    static void bench(String name, int iterations, Runnable fn) {
        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            try { fn.run(); } catch (Exception e) { break; }
        }

        // Measure
        long start = System.nanoTime();
        int successful = 0;
        for (int i = 0; i < iterations; i++) {
            try {
                fn.run();
                successful++;
            } catch (Exception e) {
                if (successful == 0) {
                    System.out.printf("  %-45s  SKIPPED (%s)%n", name, e.getClass().getSimpleName());
                    return;
                }
                break;
            }
        }
        long elapsed = System.nanoTime() - start;

        double perCall;
        String unit;
        if (elapsed / successful < 1000) {
            perCall = (double) elapsed / successful;
            unit = "ns";
        } else if (elapsed / successful < 1_000_000) {
            perCall = (double) elapsed / successful / 1000.0;
            unit = "us";
        } else {
            perCall = (double) elapsed / successful / 1_000_000.0;
            unit = "ms";
        }

        System.out.printf("  %-45s  %8.1f %s/call%n", name, perCall, unit);
    }

    static float[][] generateVectors(int n, int dim) {
        float[][] v = new float[n][dim];
        for (int i = 0; i < n; i++)
            for (int d = 0; d < dim; d++)
                v[i][d] = (float) Math.sin((i * dim + d) * 0.01);
        return v;
    }

    static int[] generateIds(int n) {
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) ids[i] = i;
        return ids;
    }
}
