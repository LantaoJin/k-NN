# OpenSearch k-NN Plugin - Learning Guide

A comprehensive guide to understanding the OpenSearch k-NN plugin codebase, organized from foundational concepts to advanced internals.

---

## Table of Contents

- [Chapter 1: Overview and Architecture](#chapter-1-overview-and-architecture)
- [Chapter 2: Core Concepts - Vectors, Spaces, and Data Types](#chapter-2-core-concepts---vectors-spaces-and-data-types)
- [Chapter 3: KNN Engines - Faiss, Lucene, and NMSLIB](#chapter-3-knn-engines---faiss-lucene-and-nmslib)
- [Chapter 4: Index Mapping and Field Types](#chapter-4-index-mapping-and-field-types)
- [Chapter 5: Indexing Pipeline - Codecs and Native Index Building](#chapter-5-indexing-pipeline---codecs-and-native-index-building)
- [Chapter 6: Search Pipeline - Queries and Scoring](#chapter-6-search-pipeline---queries-and-scoring)
- [Chapter 7: JNI Layer - Native Library Integration](#chapter-7-jni-layer---native-library-integration)
- [Chapter 8: Memory Management and Caching](#chapter-8-memory-management-and-caching)
- [Chapter 9: Memory-Optimized Search](#chapter-9-memory-optimized-search)
- [Chapter 10: Quantization](#chapter-10-quantization)
- [Chapter 11: Model Training](#chapter-11-model-training)
- [Chapter 12: Plugin Framework - REST APIs and Transport Actions](#chapter-12-plugin-framework---rest-apis-and-transport-actions)
- [Chapter 13: Advanced Features](#chapter-13-advanced-features)
- [Chapter 14: Build System and Development](#chapter-14-build-system-and-development)

---

## Chapter 1: Overview and Architecture

### What is OpenSearch k-NN?

OpenSearch k-NN is a plugin that enables **approximate nearest neighbor (ANN) search** on billions of documents across thousands of dimensions. It integrates with OpenSearch's query engine to allow vector similarity search alongside traditional text search.

### Use Cases

- Product recommendations
- Fraud detection
- Image and video search
- Semantic / related document search
- Retrieval-Augmented Generation (RAG) for LLMs

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenSearch Core                            │
├─────────────────────────────────────────────────────────────┤
│                     k-NN Plugin                              │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  REST   │  │ Transport│  │  Mapper  │  │   Codec    │  │
│  │  Layer  │  │  Actions │  │  Layer   │  │   Layer    │  │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │             │             │              │           │
│  ┌────┴─────────────┴─────────────┴──────────────┴────────┐ │
│  │              Engine Abstraction Layer                    │ │
│  │    ┌─────────┐    ┌────────┐    ┌─────────────┐       │ │
│  │    │  Faiss  │    │ Lucene │    │   NMSLIB    │       │ │
│  │    └────┬────┘    └────┬───┘    └──────┬──────┘       │ │
│  └─────────┼──────────────┼───────────────┼──────────────┘ │
│            │              │               │                  │
│  ┌─────────┴──────────────┴───────────────┴──────────────┐ │
│  │                  JNI Layer (C++)                        │ │
│  │        faiss_wrapper  │  nmslib_wrapper                │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
k-NN/
├── src/main/java/org/opensearch/knn/
│   ├── plugin/          # Plugin entry point, REST handlers, transport
│   ├── index/           # Core indexing: mapper, codec, query, engine, memory
│   ├── jni/             # Java JNI bindings to native libraries
│   ├── memoryoptsearch/ # Memory-optimized search (reads index from disk)
│   ├── quantization/    # Vector quantization framework
│   ├── training/        # Model training for IVF/PQ methods
│   ├── indices/         # Model storage and management
│   ├── search/          # Search extensions (MMR, processors)
│   ├── common/          # Shared utilities, exceptions, feature flags
│   └── profile/         # Query profiling/metrics
├── jni/                 # C++ native code (Faiss/NMSLIB wrappers)
│   ├── src/             # JNI implementation files
│   ├── include/         # Header files
│   ├── external/        # Git submodules for faiss, nmslib
│   └── patches/         # Patches applied to external libraries
└── src/test/            # Unit and integration tests
```

### Entry Point

The plugin entry point is `KNNPlugin.java` which implements OpenSearch's plugin interfaces:
- Registers the `knn_vector` field type
- Registers REST endpoints and transport actions
- Sets up the custom codec for vector indexing
- Initializes native libraries and memory management

---

## Chapter 2: Core Concepts - Vectors, Spaces, and Data Types

### Key Files

- `src/main/java/org/opensearch/knn/index/SpaceType.java`
- `src/main/java/org/opensearch/knn/index/VectorDataType.java`
- `src/main/java/org/opensearch/knn/index/KNNVectorSimilarityFunction.java`

### Vector Data Types

The plugin supports multiple data types for vectors:

| Type | Description | Storage per Dimension |
|------|-------------|----------------------|
| `FLOAT` | 32-bit floating point | 4 bytes |
| `BYTE` | 8-bit signed integer (-128 to 127) | 1 byte |
| `BINARY` | Binary vectors (bit-level) | 1/8 byte |

### Space Types (Distance Metrics)

Space types define how similarity between vectors is measured:

| Space Type | Description | Score Translation |
|------------|-------------|-------------------|
| `L2` (Euclidean) | Squared L2 distance | `1 / (1 + rawScore)` |
| `L1` (Manhattan) | Sum of absolute differences | `1 / (1 + rawScore)` |
| `LINF` (Chebyshev) | Maximum absolute difference | `1 / (1 + rawScore)` |
| `COSINESIMIL` | Cosine similarity | `1 / (1 + rawScore)` for nmslib; special for faiss |
| `INNER_PRODUCT` | Dot product | Varies by engine |
| `HAMMING` | Hamming distance (binary) | `1 / (1 + rawScore)` |

### Score Translation

Each space type implements `scoreTranslation(float rawScore)` which converts the raw distance/similarity from the native library into an OpenSearch-compatible score (higher = more relevant).

---

## Chapter 3: KNN Engines - Faiss, Lucene, and NMSLIB

### Key Files

- `src/main/java/org/opensearch/knn/index/engine/KNNEngine.java`
- `src/main/java/org/opensearch/knn/index/engine/KNNLibrary.java`
- `src/main/java/org/opensearch/knn/index/engine/faiss/Faiss.java`
- `src/main/java/org/opensearch/knn/index/engine/lucene/Lucene.java`
- `src/main/java/org/opensearch/knn/index/engine/nmslib/Nmslib.java`

### Engine Overview

The plugin supports three backends ("engines"):

#### Faiss (Facebook AI Similarity Search)
- Most feature-rich engine
- Supports: HNSW, IVF, Product Quantization (PQ), Scalar Quantization (SQ)
- Native C++ library accessed via JNI
- Supports training-based methods (IVF requires training)
- Supports binary vectors and all distance metrics

#### Lucene
- Built into the JVM (no native library needed)
- Supports: HNSW, Flat (brute force)
- Good for smaller datasets or when JNI is not available
- Supports byte and float vectors

#### NMSLIB (Non-Metric Space Library)
- Legacy engine (original engine for the plugin)
- Supports: HNSW only
- Native C++ library accessed via JNI
- Being phased out in favor of Faiss

### Engine Resolution

`EngineResolver.java` selects the appropriate engine based on:
1. User-specified engine in mapping
2. Method type (some methods only available on certain engines)
3. Vector data type compatibility
4. Compression level requirements

### Algorithm: HNSW (Hierarchical Navigable Small World)

The primary algorithm used across all engines:

```
Parameters:
- M: Number of connections per node (graph connectivity)
- ef_construction: Size of dynamic candidate list during indexing
- ef_search: Size of dynamic candidate list during search

Trade-offs:
- Higher M → better recall, more memory, slower indexing
- Higher ef_construction → better graph quality, slower indexing
- Higher ef_search → better recall, slower search
```

### Algorithm: IVF (Inverted File Index)

Available in Faiss only:

```
Parameters:
- nlist: Number of cluster centroids
- nprobes: Number of clusters to search at query time

Requires training phase to learn cluster centroids.
```

---

## Chapter 4: Index Mapping and Field Types

### Key Files

- `src/main/java/org/opensearch/knn/index/mapper/KNNVectorFieldMapper.java`
- `src/main/java/org/opensearch/knn/index/mapper/KNNVectorFieldType.java`
- `src/main/java/org/opensearch/knn/index/mapper/Mode.java`
- `src/main/java/org/opensearch/knn/index/mapper/CompressionLevel.java`

### The `knn_vector` Field Type

When you create a mapping with `"type": "knn_vector"`, `KNNVectorFieldMapper` handles:
1. Parsing mapping parameters (dimension, method, model_id, etc.)
2. Validating vector data at index time
3. Storing vectors in doc values for search

### Mapping Modes

| Mode | Description |
|------|-------------|
| `NOT_CONFIGURED` | Default - engine decides |
| `IN_MEMORY` | Traditional approach - load index into memory |
| `ON_DISK` | Memory-optimized search from disk |

### Compression Levels

| Level | Description | Typical Encoding |
|-------|-------------|-----------------|
| `x1` | No compression | Flat (original vectors) |
| `x2` | 2x compression | FP16 scalar quantization |
| `x4` | 4x compression | 8-bit scalar quantization |
| `x8` | 8x compression | 4-bit quantization |
| `x16` | 16x compression | 2-bit or binary quantization |
| `x32` | 32x compression | 1-bit binary quantization |

### Example Mapping

```json
{
  "properties": {
    "my_vector": {
      "type": "knn_vector",
      "dimension": 768,
      "method": {
        "name": "hnsw",
        "space_type": "l2",
        "engine": "faiss",
        "parameters": {
          "m": 16,
          "ef_construction": 256,
          "encoder": {
            "name": "sq",
            "parameters": { "type": "fp16" }
          }
        }
      }
    }
  }
}
```

---

## Chapter 5: Indexing Pipeline - Codecs and Native Index Building

### Key Files

- `src/main/java/org/opensearch/knn/index/codec/KNN80Codec/` (and other codec versions)
- `src/main/java/org/opensearch/knn/index/codec/nativeindex/`
- `src/main/java/org/opensearch/knn/index/codec/KNNCodecService.java`
- `src/main/java/org/opensearch/knn/index/codec/transfer/`

### How Vectors Get Indexed

```
Document Indexing Flow:
                                                                
1. Document arrives → KNNVectorFieldMapper validates & parses vector
2. Vector stored in Lucene doc values (binary format)
3. During segment flush/merge:
   a. KNN codec reads vectors from doc values
   b. Vectors transferred to native library via JNI
   c. Native library builds the index (HNSW graph / IVF clusters)
   d. Native index written as a sidecar file alongside the segment
```

### Codec Versioning

Each OpenSearch version has its own codec (e.g., `KNN1040Codec`). Backward-compatible codecs exist for reading older segments:
- `KNN80Codec` - Oldest supported
- `KNN910Codec`, `KNN920Codec`, ... (track Lucene/OpenSearch versions)
- `KNN1040Codec` - Latest

### Native Index Build Strategies

The `nativeindex/` package handles building the native ANN index:

1. **Default Strategy**: Build index on the data node during segment flush
2. **Remote Build Strategy**: Offload index building to a remote service (for large indices)

### The KNNCodecService

Wraps the standard Lucene codec to inject:
- Custom `KNNVectorsFormat` for writing/reading vector data
- Custom doc values format for vector storage
- Format negotiation based on engine type

---

## Chapter 6: Search Pipeline - Queries and Scoring

### Key Files

- `src/main/java/org/opensearch/knn/index/query/KNNQueryBuilder.java`
- `src/main/java/org/opensearch/knn/index/query/KNNQuery.java`
- `src/main/java/org/opensearch/knn/index/query/KNNWeight.java`
- `src/main/java/org/opensearch/knn/index/query/nativelib/NativeEngineKnnVectorQuery.java`
- `src/main/java/org/opensearch/knn/index/query/exactsearch/ExactSearcher.java`

### Query Flow

```
User Query (JSON) 
    → KNNQueryBuilder (parses request)
    → KNNQuery (Lucene Query object)
    → KNNWeight (per-segment execution)
        ├── Native Engine path: JNI call to search the ANN index
        ├── Lucene Engine path: Lucene's built-in kNN search
        └── Exact Search path: brute-force when filter is very selective
    → Scored documents returned to OpenSearch
```

### KNNQueryBuilder

The entry point for k-NN queries. Supports:
- `k`: Number of nearest neighbors to return
- `vector`: The query vector
- `filter`: Pre-filter documents before ANN search
- `method_parameters`: Runtime search params (e.g., `ef_search`)
- `rescore`: Two-phase approach (ANN first, then exact re-scoring)
- `min_score` / `max_distance`: Radial search (radius-based, not top-k)

### Example Query

```json
{
  "query": {
    "knn": {
      "my_vector": {
        "vector": [0.1, 0.2, ...],
        "k": 10,
        "filter": { "term": { "category": "electronics" } },
        "method_parameters": { "ef_search": 512 }
      }
    }
  }
}
```

### Search Strategies

1. **ANN Search (Default)**: Uses the pre-built graph/index for fast approximate results
2. **Exact Search**: Brute-force scan when filters eliminate most documents
3. **Rescore Search**: ANN first with oversampling, then exact re-rank top candidates
4. **Radial Search**: Find all vectors within a distance/score threshold

### KNNWeight - The Heart of Search

`KNNWeight` (and `DefaultKNNWeight`) implements `Weight.scorer()`:
- Loads the native index from cache (or disk)
- Decides between ANN vs exact search based on filter ratio
- Calls JNI to execute the search on native libraries
- Converts results to Lucene scorers

---

## Chapter 7: JNI Layer - Native Library Integration

### Key Files

- `src/main/java/org/opensearch/knn/jni/` (Java side)
- `jni/src/` (C++ side)
- `jni/include/` (C++ headers)

### Architecture

```
Java Layer                          C++ Layer
─────────────────                   ─────────────────
FaissService.java    ──JNI──→    org_opensearch_knn_jni_FaissService.cpp
NmslibService.java   ──JNI──→    org_opensearch_knn_jni_NmslibService.cpp
JNICommons.java      ──JNI──→    org_opensearch_knn_jni_JNICommons.cpp
                                         │
                                    faiss_wrapper.cpp
                                    nmslib_wrapper.cpp
                                         │
                                  ┌──────┴──────┐
                                  │  libfaiss   │
                                  │  libnmslib  │
                                  └─────────────┘
```

### Key JNI Operations

| Operation | Description |
|-----------|-------------|
| `createIndex` | Build an ANN index from vectors |
| `loadIndex` | Load a pre-built index into memory |
| `queryIndex` | Search the loaded index for nearest neighbors |
| `freeIndex` | Release memory for a loaded index |
| `trainIndex` | Train an IVF/PQ model from sample data |

### SIMD Optimizations

The `jni/src/simd/` directory contains platform-specific optimizations:
- Distance computation kernels optimized with SIMD instructions
- Exposed via `SimdVectorComputeService`

---

## Chapter 8: Memory Management and Caching

### Key Files

- `src/main/java/org/opensearch/knn/index/memory/NativeMemoryCacheManager.java`
- `src/main/java/org/opensearch/knn/index/memory/NativeMemoryAllocation.java`
- `src/main/java/org/opensearch/knn/index/memory/NativeMemoryLoadStrategy.java`
- `src/main/java/org/opensearch/knn/index/KNNCircuitBreaker.java`

### Native Memory Cache

When a segment is searched, its native ANN index is loaded into off-heap memory:

```
Search Request
    → Check NativeMemoryCacheManager
        ├── Cache HIT: return pointer to loaded index
        └── Cache MISS: 
            1. Check circuit breaker (enough memory?)
            2. Load index from disk via JNI
            3. Store pointer in cache
            4. Return pointer
```

### Cache Eviction

- **LRU-based**: Least recently used indices evicted first
- **Circuit Breaker**: Prevents loading if total native memory exceeds threshold
- **Expiry**: Indices can expire after a configurable time
- **Explicit Clear**: Via the `/_plugins/_knn/clear_cache` API

### Memory Settings

Key settings in `KNNSettings.java`:
- `knn.memory.circuit_breaker.limit`: Maximum native memory percentage (default: 50%)
- `knn.memory.circuit_breaker.enabled`: Enable/disable the breaker
- `knn.cache.item.expiry.enabled`: Enable time-based cache expiry

### Warmup

The warmup API (`/_plugins/_knn/warmup/{index}`) pre-loads all native indices into memory before search traffic arrives, avoiding cold-start latency.

---

## Chapter 9: Memory-Optimized Search

### Key Files

- `src/main/java/org/opensearch/knn/memoryoptsearch/` (entire package)
- `src/main/java/org/opensearch/knn/index/query/memoryoptsearch/MemoryOptimizedKNNWeight.java`

### Concept

Traditional k-NN search loads the entire ANN graph into off-heap memory. Memory-optimized search reads the graph structure directly from disk using memory-mapped files, dramatically reducing memory requirements.

### How It Works

```
Traditional (in-memory):
  Disk → [Load full index into RAM] → Search in RAM

Memory-Optimized (on-disk):
  Disk → [Memory-map the file] → Traverse graph by reading pages on demand
```

### Key Components

- **FaissMemoryOptimizedSearcher**: Reads Faiss HNSW index structure from mmap'd files
- **FaissHnswGraph**: Parses the HNSW graph structure (layers, neighbors) from raw bytes
- **MMapVectorValues / MMapFloatVectorValues**: Provides vector data from mmap'd storage
- **MemorySegmentAddressExtractor**: JDK version-specific APIs for memory segment access

### Trade-offs

| Aspect | In-Memory | Memory-Optimized |
|--------|-----------|------------------|
| Memory usage | High (full index in RAM) | Low (OS page cache) |
| First query latency | High (load time) | Low (no preload) |
| Steady-state latency | Lower | Slightly higher (page faults) |
| Best for | Hot, frequently-searched indices | Large, infrequently-searched indices |

---

## Chapter 10: Quantization

### Key Files

- `src/main/java/org/opensearch/knn/quantization/quantizer/` (quantizer implementations)
- `src/main/java/org/opensearch/knn/quantization/models/` (state and params)
- `src/main/java/org/opensearch/knn/quantization/factory/` (factory pattern)
- `src/main/java/org/opensearch/knn/quantization/sampler/` (training data sampling)

### What is Quantization?

Quantization reduces the memory footprint of vectors by representing them with fewer bits, trading some accuracy for significant space savings.

### Supported Quantization Types

#### Scalar Quantization (SQ)
Represents each dimension with fewer bits:
- **1-bit**: Each dimension → 0 or 1 (32x compression)
- **2-bit**: Each dimension → 0,1,2,3 (16x compression)
- **4-bit**: Each dimension → 0..15 (8x compression)

#### In Faiss Engine
- **SQ (fp16)**: Half-precision float (2x compression)
- **SQ (8-bit)**: 8-bit integer per dimension (4x compression)
- **PQ (Product Quantization)**: Splits vector into sub-vectors, encodes each with a codebook

### Quantization Pipeline

```
1. Training Phase:
   Sample vectors → Compute quantization parameters (thresholds, codebooks)
   → Store as QuantizationState

2. Indexing Phase:
   Full vector → Apply quantizer → Store compressed representation

3. Search Phase:
   Query vector → Compare against compressed vectors (with asymmetric distance)
```

### QuantizerFactory

Uses registry pattern to create quantizers based on `QuantizationParams`:
```java
QuantizerFactory.getQuantizer(ScalarQuantizationParams(SQType.ONE_BIT))
  → returns OneBitScalarQuantizer
```

---

## Chapter 11: Model Training

### Key Files

- `src/main/java/org/opensearch/knn/training/TrainingJob.java`
- `src/main/java/org/opensearch/knn/training/TrainingJobRunner.java`
- `src/main/java/org/opensearch/knn/training/VectorReader.java`
- `src/main/java/org/opensearch/knn/indices/Model.java`
- `src/main/java/org/opensearch/knn/indices/ModelDao.java`

### Why Training?

Some ANN algorithms (IVF, PQ) require a **training phase** to learn data distribution before indexing. Training produces a **model** that is then used during index creation.

### Training Flow

```
1. User calls POST /_plugins/_knn/models/{model_id}/_train
   - Specifies: training_index, training_field, method (ivf/pq params)

2. TrainingJobRouter selects a node to run training

3. VectorReader reads training vectors from the specified index

4. TrainingJob:
   a. Reads vectors into memory (FloatTrainingDataConsumer)
   b. Calls JNI trainIndex() with vectors and parameters
   c. Native library learns cluster centroids / codebooks
   d. Model (binary blob) stored in system index .opensearch-knn-models

5. Model can now be referenced in mappings via model_id
```

### Model Lifecycle

```
States: CREATED → TRAINING → CREATED (success) / FAILED

Storage: .opensearch-knn-models system index
Cache: ModelCache for fast access during indexing
Deletion: ModelGraveyard tracks deleted models for cluster-wide cleanup
```

### Using a Trained Model

```json
{
  "properties": {
    "my_vector": {
      "type": "knn_vector",
      "model_id": "my-ivf-model"
    }
  }
}
```

---

## Chapter 12: Plugin Framework - REST APIs and Transport Actions

### Key Files

- `src/main/java/org/opensearch/knn/plugin/KNNPlugin.java`
- `src/main/java/org/opensearch/knn/plugin/rest/` (REST handlers)
- `src/main/java/org/opensearch/knn/plugin/transport/` (transport actions)
- `src/main/java/org/opensearch/knn/plugin/stats/` (statistics)

### REST API Endpoints

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `GET /_plugins/_knn/stats` | RestKNNStatsHandler | Plugin statistics |
| `POST /_plugins/_knn/warmup/{index}` | RestKNNWarmupHandler | Pre-load indices into memory |
| `POST /_plugins/_knn/models/{id}/_train` | RestTrainModelHandler | Train a new model |
| `GET /_plugins/_knn/models/{id}` | RestGetModelHandler | Get model metadata |
| `DELETE /_plugins/_knn/models/{id}` | RestDeleteModelHandler | Delete a model |
| `POST /_plugins/_knn/models/_search` | RestSearchModelHandler | Search models |
| `POST /_plugins/_knn/{index}/_clear_cache` | RestClearCacheHandler | Evict indices from cache |

### Transport Actions Pattern

Each API follows the OpenSearch transport action pattern:

```
REST Handler → Action (e.g., KNNStatsAction)
    → TransportAction (e.g., KNNStatsTransportAction)
        → Executes on appropriate node(s)
        → Returns Response
```

### Statistics

The stats system tracks:
- **Cluster-level**: Total graph count, circuit breaker status, model index status
- **Node-level**: Cache stats (hits, misses, size), graph stats, request counts
- **Counters**: Script compilations, errors, query counts

---

## Chapter 13: Advanced Features

### Filtered Search

k-NN supports pre-filtering with any OpenSearch query:
- For Lucene engine: native integration with Lucene's filtered kNN
- For Faiss/NMSLIB: filter evaluated first, then ANN search on filtered set
- Threshold-based strategy: switches to exact search if filter is very selective

### Rescore (Two-Phase Search)

```json
{
  "knn": {
    "my_vector": {
      "vector": [...],
      "k": 10,
      "rescore": {
        "oversample_factor": 3.0
      }
    }
  }
}
```
1. First phase: ANN search for `k * oversample_factor` candidates
2. Second phase: Exact distance computation on candidates, return top `k`

### Radial Search

Find all vectors within a distance/score threshold (not just top-k):
- `min_score`: Return vectors with score >= threshold
- `max_distance`: Return vectors with distance <= threshold

### MMR (Maximal Marginal Relevance)

Located in `search/processor/mmr/`:
- Diversifies search results by balancing relevance and diversity
- Implemented as a search pipeline processor

### Painless Scripting

`plugin/script/KNNScoringScriptEngine.java` enables vector operations in Painless scripts:
- Custom scoring functions using vector similarity
- Supports L2, cosine, hamming distance computations in scripts

### Remote Index Build

`index/remote/` package supports offloading index building:
- Sends vectors to a remote service for index construction
- Polls for completion
- Useful for very large indices that are expensive to build locally

### Concurrent Search

`KNNConcurrentSearchRequestDecider` enables parallel segment search:
- Multiple segments can be searched concurrently
- Improves latency for indices with many segments

---

## Chapter 14: Build System and Development

### Build Structure

```
build.gradle              # Main build file
├── JNI compilation       # Native C++ build via CMake
├── Java compilation      # Standard Gradle Java plugin
├── Integration tests     # OpenSearch test framework
└── Packaging            # Plugin zip for distribution
```

### Key Gradle Tasks

```bash
./gradlew build              # Full build
./gradlew test               # Unit tests
./gradlew integTest          # Integration tests
./gradlew run                # Run OpenSearch with plugin
./gradlew assemble           # Build plugin zip
```

### Prerequisites

- Java 21+
- C++ compiler (for JNI)
- CMake (for native build)
- OpenSearch source (for integration)

### Testing

- **Unit Tests**: `src/test/java/` - standard JUnit tests
- **Integration Tests**: Full cluster tests via OpenSearch test framework
- **Test Fixtures**: `src/testFixtures/` - shared test utilities
- **JNI Tests**: `jni/tests/` - C++ unit tests with Google Test

### Codec Compatibility

When making codec changes:
1. Create a new codec version (e.g., `KNN1050Codec`)
2. Move the old version to `backward_codecs/`
3. Ensure old segments can still be read
4. Register the new codec in `META-INF/services`

---

## Appendix: Reading Order for New Contributors

If you're new to the codebase, read in this order:

1. **`KNNPlugin.java`** - See everything the plugin registers
2. **`SpaceType.java`** + **`VectorDataType.java`** - Core value types
3. **`KNNEngine.java`** - How engines are abstracted
4. **`KNNVectorFieldMapper.java`** - How mappings work
5. **`KNNQueryBuilder.java`** - How queries are parsed
6. **`KNNWeight.java`** / **`DefaultKNNWeight.java`** - How search executes
7. **`NativeMemoryCacheManager.java`** - How indices are loaded/cached
8. **`jni/src/faiss_wrapper.cpp`** - How JNI calls the native library
9. **`TrainingJob.java`** - How model training works
10. **`FaissMemoryOptimizedSearcher.java`** - Memory-optimized search internals

---

## Appendix: Glossary

| Term | Definition |
|------|-----------|
| **ANN** | Approximate Nearest Neighbor - find "close enough" vectors efficiently |
| **HNSW** | Hierarchical Navigable Small World - graph-based ANN algorithm |
| **IVF** | Inverted File Index - partition-based ANN algorithm |
| **PQ** | Product Quantization - vector compression by sub-vector codebook encoding |
| **SQ** | Scalar Quantization - per-dimension bit reduction |
| **ef_search** | HNSW parameter: search-time beam width (larger = better recall, slower) |
| **ef_construction** | HNSW parameter: build-time beam width (larger = better graph, slower build) |
| **M** | HNSW parameter: max connections per node in the graph |
| **nlist** | IVF parameter: number of clusters |
| **nprobes** | IVF parameter: number of clusters to visit during search |
| **Circuit Breaker** | Mechanism to prevent excessive native memory usage |
| **Warmup** | Pre-loading native indices into memory before search |
| **Segment** | A Lucene index partition; each has its own ANN index |
| **Doc Values** | Columnar storage format where raw vectors are persisted |
| **Codec** | Lucene's format abstraction for reading/writing index data |
