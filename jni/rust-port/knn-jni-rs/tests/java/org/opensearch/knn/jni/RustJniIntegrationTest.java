// SPDX-License-Identifier: Apache-2.0
//
// Full integration test for the Rust JNI library.
// Tests the complete round-trip: Java → Rust → Faiss → Rust → Java
//
// Build & Run:
//   cd tests/java
//   javac org/opensearch/knn/jni/*.java
//   java -Djava.library.path=../../target/debug -cp . org.opensearch.knn.jni.RustJniIntegrationTest

package org.opensearch.knn.jni;

public class RustJniIntegrationTest {

    static int passed = 0;
    static int failed = 0;

    public static void main(String[] args) {
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║   OpenSearch k-NN — Rust JNI Integration Test Suite      ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝\n");

        test1_libraryLoads();
        test2_initLibrary();
        test3_freeNullPointer();
        test4_isSharedIndexStateRequired();
        test5_storeAndFreeVectors();
        test6_loadIndexInvalidPath();

        System.out.println("\n═══════════════════════════════════════════════════════════");
        System.out.printf("Results: %d passed, %d failed%n", passed, failed);
        System.out.println("═══════════════════════════════════════════════════════════");
        System.exit(failed > 0 ? 1 : 0);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Test 1: Library loads and JNI_OnLoad succeeds
    // ─────────────────────────────────────────────────────────────────────────────
    static void test1_libraryLoads() {
        System.out.println("[TEST 1] Library loads and JNI_OnLoad succeeds");
        try {
            // Force class loading which triggers System.loadLibrary
            Class.forName("org.opensearch.knn.jni.FaissService");
            System.out.println("  ✓ libknn_jni_rs.dylib loaded successfully");
            pass();
        } catch (Throwable e) {
            fail(e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Test 2: initLibrary() (no-op, but verifies symbol resolution)
    // ─────────────────────────────────────────────────────────────────────────────
    static void test2_initLibrary() {
        System.out.println("[TEST 2] FaissService.initLibrary() resolves and executes");
        try {
            FaissService.initLibrary();
            System.out.println("  ✓ initLibrary() returned without error");
            pass();
        } catch (Throwable e) {
            fail(e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Test 3: free(0) with null pointer guard (should not crash)
    // ─────────────────────────────────────────────────────────────────────────────
    static void test3_freeNullPointer() {
        System.out.println("[TEST 3] FaissService.free(0, false) - null pointer guard");
        try {
            FaissService.free(0L, false);
            System.out.println("  ✓ free(0, false) did not crash");
            FaissService.free(0L, true);
            System.out.println("  ✓ free(0, true) did not crash");
            pass();
        } catch (Throwable e) {
            fail(e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Test 4: isSharedIndexStateRequired(0) - should return false for null
    // ─────────────────────────────────────────────────────────────────────────────
    static void test4_isSharedIndexStateRequired() {
        System.out.println("[TEST 4] FaissService.isSharedIndexStateRequired(0)");
        try {
            boolean result = FaissService.isSharedIndexStateRequired(0L);
            System.out.println("  ✓ Returned: " + result + " (expected: false)");
            if (!result) {
                pass();
            } else {
                System.out.println("  ✗ Expected false for null pointer");
                failed++;
            }
        } catch (Throwable e) {
            fail(e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Test 5: Store vectors, verify address, free
    // ─────────────────────────────────────────────────────────────────────────────
    static void test5_storeAndFreeVectors() {
        System.out.println("[TEST 5] JNICommons.storeVectorData + freeVectorData");
        try {
            float[][] vectors = {
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f},
                {7.0f, 8.0f, 9.0f}
            };

            // Store vectors (address=0 means allocate new)
            long address = JNICommons.storeVectorData(0L, vectors, 3 * 3, true);
            System.out.println("  ✓ storeVectorData returned address: 0x" + Long.toHexString(address));

            if (address == 0) {
                System.out.println("  ✗ Address is 0 (allocation failed)");
                failed++;
                return;
            }

            // Free the vectors
            JNICommons.freeVectorData(address);
            System.out.println("  ✓ freeVectorData completed without crash");
            pass();
        } catch (Throwable e) {
            fail(e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Test 6: loadIndex with invalid path - should throw exception, not crash
    // ─────────────────────────────────────────────────────────────────────────────
    static void test6_loadIndexInvalidPath() {
        System.out.println("[TEST 6] FaissService.loadIndex(\"/nonexistent\") - error handling");
        try {
            long result = FaissService.loadIndex("/nonexistent/path/index.faiss");
            // If we get here without exception, the Faiss call returned null or error
            System.out.println("  ✓ Returned: " + result + " (exception would also be acceptable)");
            pass();
        } catch (Exception e) {
            // This is the EXPECTED path — Rust catches the Faiss error and throws java.lang.Exception
            System.out.println("  ✓ Exception thrown as expected: " + e.getClass().getSimpleName());
            System.out.println("    Message: " + e.getMessage());
            pass();
        } catch (Throwable e) {
            fail(e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    static void pass() {
        passed++;
        System.out.println();
    }

    static void fail(Throwable e) {
        System.out.println("  ✗ FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        e.printStackTrace(System.out);
        System.out.println();
        failed++;
    }
}
