#!/bin/bash
# Run the JNI benchmark comparing C++ vs Rust implementations.
#
# Usage:
#   ./run_benchmark.sh cpp    # Benchmark with C++ JNI
#   ./run_benchmark.sh rust   # Benchmark with Rust JNI
#   ./run_benchmark.sh both   # Run both and show side-by-side

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DIR="$SCRIPT_DIR/../../../../build/release"
RUST_CRATE_DIR="$SCRIPT_DIR/../.."

# Compile benchmark
cd "$SCRIPT_DIR"
javac org/opensearch/knn/jni/NativeJniBenchmark.java \
      org/opensearch/knn/index/query/KNNQueryResult.java \
      org/apache/lucene/index/MergeAbortChecker.java

run_bench() {
    local label=$1
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Running benchmark with: $label JNI"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$RELEASE_DIR" \
        java -Djava.library.path="$RELEASE_DIR" -cp . \
        org.opensearch.knn.jni.NativeJniBenchmark
}

case "${1:-both}" in
    cpp)
        echo "Deploying C++ JNI..."
        cd "$SCRIPT_DIR/../../../.." && ./gradlew buildJniLib -x cmakeJniLib 2>/dev/null
        run_bench "C++"
        ;;
    rust)
        echo "Deploying Rust JNI..."
        cd "$SCRIPT_DIR/../../../.." && ./gradlew deployRustJniLib 2>/dev/null
        run_bench "Rust"
        ;;
    both)
        # Run C++ first
        echo "Step 1: Deploying C++ JNI..."
        cd "$SCRIPT_DIR/../../../.." && ./gradlew buildJniLib -x cmakeJniLib 2>/dev/null
        cd "$SCRIPT_DIR"
        run_bench "C++" | tee /tmp/bench_cpp.txt

        # Run Rust
        echo ""
        echo "Step 2: Deploying Rust JNI..."
        cd "$SCRIPT_DIR/../../../.." && ./gradlew deployRustJniLib 2>/dev/null
        cd "$SCRIPT_DIR"
        run_bench "Rust" | tee /tmp/bench_rust.txt

        # Compare
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  COMPARISON"
        echo "════════════════════════════════════════════════════════════"
        echo ""
        paste <(grep "ns/call\|us/call\|MB/s\|us/iteration" /tmp/bench_cpp.txt) \
              <(grep "ns/call\|us/call\|MB/s\|us/iteration" /tmp/bench_rust.txt) \
            | awk '{print "C++: " $0}'
        ;;
    *)
        echo "Usage: $0 [cpp|rust|both]"
        exit 1
        ;;
esac
