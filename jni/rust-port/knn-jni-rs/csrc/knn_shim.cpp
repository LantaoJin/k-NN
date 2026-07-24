// SPDX-License-Identifier: Apache-2.0
//
// C++ shim library for the Rust k-NN JNI port.
// Exposes C-callable functions for operations that require C++ features
// (virtual class inheritance, RTTI dynamic_cast, template specializations)
// that cannot be implemented in pure Rust FFI.
//
// This file is compiled alongside libfaiss and linked into the Rust cdylib.

#include "faiss/Index.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexIVF.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/impl/io.h"
#include "faiss/index_io.h"

#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

// NMSLIB includes
#include "init.h"
#include "index.h"
#include "params.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "spacefactory.h"
#include "space.h"
#include "object.h"
#include "hnswquery.h"
#include "method/hnsw.h"

// ---------------------------------------------------------------------------
// 1. RTTI helpers: detect index types and set parameters
// ---------------------------------------------------------------------------

extern "C" {

/// Check if the underlying index (unwrapping IndexIDMap) is IndexHNSW.
int knn_shim_is_index_hnsw(void* index_ptr) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return 0;
    // Unwrap IndexIDMap if present
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        return dynamic_cast<faiss::IndexHNSW*>(idmap->index) != nullptr ? 1 : 0;
    }
    return dynamic_cast<faiss::IndexHNSW*>(index) != nullptr ? 1 : 0;
}

/// Check if the underlying index (unwrapping IndexIDMap) is IndexIVF.
int knn_shim_is_index_ivf(void* index_ptr) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return 0;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        return dynamic_cast<faiss::IndexIVF*>(idmap->index) != nullptr ? 1 : 0;
    }
    return dynamic_cast<faiss::IndexIVF*>(index) != nullptr ? 1 : 0;
}

/// Check if the underlying index is IndexIVFPQ with L2 metric.
int knn_shim_is_index_ivfpq_l2(void* index_ptr) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return 0;
    faiss::Index* candidate = index;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        candidate = idmap->index;
    }
    if (auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(candidate)) {
        return ivfpq->metric_type == faiss::METRIC_L2 ? 1 : 0;
    }
    return 0;
}

/// Set efConstruction on an HNSW index (unwraps IndexIDMap).
void knn_shim_set_hnsw_ef_construction(void* index_ptr, int ef_construction) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return;
    faiss::IndexHNSW* hnsw = nullptr;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        hnsw = dynamic_cast<faiss::IndexHNSW*>(idmap->index);
    } else {
        hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
    }
    if (hnsw) {
        hnsw->hnsw.efConstruction = ef_construction;
    }
}

/// Set efSearch on an HNSW index (unwraps IndexIDMap).
void knn_shim_set_hnsw_ef_search(void* index_ptr, int ef_search) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return;
    faiss::IndexHNSW* hnsw = nullptr;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        hnsw = dynamic_cast<faiss::IndexHNSW*>(idmap->index);
    } else {
        hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
    }
    if (hnsw) {
        hnsw->hnsw.efSearch = ef_search;
    }
}

/// Set nprobe on an IVF index (unwraps IndexIDMap).
void knn_shim_set_ivf_nprobe(void* index_ptr, int nprobe) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return;
    faiss::IndexIVF* ivf = nullptr;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        ivf = dynamic_cast<faiss::IndexIVF*>(idmap->index);
    } else {
        ivf = dynamic_cast<faiss::IndexIVF*>(index);
    }
    if (ivf) {
        ivf->nprobe = nprobe;
    }
}

/// Get efSearch from an HNSW index. Returns -1 if not HNSW.
int knn_shim_get_hnsw_ef_search(void* index_ptr) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return -1;
    faiss::IndexHNSW* hnsw = nullptr;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        hnsw = dynamic_cast<faiss::IndexHNSW*>(idmap->index);
    } else {
        hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
    }
    return hnsw ? hnsw->hnsw.efSearch : -1;
}

/// Get nprobe from an IVF index. Returns -1 if not IVF.
int knn_shim_get_ivf_nprobe(void* index_ptr) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return -1;
    faiss::IndexIVF* ivf = nullptr;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        ivf = dynamic_cast<faiss::IndexIVF*>(idmap->index);
    } else {
        ivf = dynamic_cast<faiss::IndexIVF*>(index);
    }
    return ivf ? (int)ivf->nprobe : -1;
}

// ---------------------------------------------------------------------------
// 2. IVFPQ precomputed table
// ---------------------------------------------------------------------------

/// Initialize the IVFPQ precomputed table. Returns a pointer to the
/// allocated AlignedTable<float> (as void*), or nullptr on failure.
void* knn_shim_init_ivfpq_precomputed_table(void* index_ptr) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index) return nullptr;

    faiss::Index* candidate = index;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        candidate = idmap->index;
    }

    auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(candidate);
    if (!ivfpq || ivfpq->metric_type != faiss::METRIC_L2) {
        return nullptr;
    }

    auto* table = new faiss::AlignedTable<float>();
    int use_precomputed_table = 0;
    faiss::initialize_IVFPQ_precomputed_table(
        use_precomputed_table,
        ivfpq->quantizer,
        ivfpq->pq,
        *table,
        ivfpq->by_residual,
        ivfpq->verbose
    );
    return table;
}

/// Set the precomputed table on an IVFPQ index.
void knn_shim_set_ivfpq_precomputed_table(void* index_ptr, void* table_ptr) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    if (!index || !table_ptr) return;

    faiss::Index* candidate = index;
    if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        candidate = idmap->index;
    }

    auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(candidate);
    if (!ivfpq) return;

    auto* table = reinterpret_cast<faiss::AlignedTable<float>*>(table_ptr);
    int use_precomputed_table = 1;
    ivfpq->set_precomputed_table(table, use_precomputed_table);
}

/// Free an IVFPQ precomputed table.
void knn_shim_free_ivfpq_precomputed_table(void* table_ptr) {
    if (!table_ptr) return;
    delete reinterpret_cast<faiss::AlignedTable<float>*>(table_ptr);
}

// ---------------------------------------------------------------------------
// 3. Faiss IOReader/IOWriter from callbacks (for stream support)
// ---------------------------------------------------------------------------

// Callback types matching the Rust side (3-param: ctx, ptr, nbytes)
typedef size_t (*knn_read_callback_t)(void* ctx, void* dest, size_t nbytes);
typedef size_t (*knn_write_callback_t)(void* ctx, const void* src, size_t nbytes);
typedef void (*knn_flush_callback_t)(void* ctx);

/// IOReader backed by a Rust callback.
struct KnnCallbackIOReader : faiss::IOReader {
    void* ctx;
    knn_read_callback_t read_fn;

    KnnCallbackIOReader(void* _ctx, knn_read_callback_t _read_fn)
        : ctx(_ctx), read_fn(_read_fn) {
        name = "KnnCallbackIOReader";
    }

    size_t operator()(void* ptr, size_t size, size_t nitems) override {
        // Faiss passes (size, nitems); we pass total bytes to the Rust callback
        size_t total = size * nitems;
        size_t bytes_read = read_fn(ctx, ptr, total);
        // Return nitems (Faiss expects item count, not byte count)
        return (size > 0) ? bytes_read / size : 0;
    }

    int filedescriptor() override {
        return -1;
    }
};

/// IOWriter backed by a Rust callback.
struct KnnCallbackIOWriter : faiss::IOWriter {
    void* ctx;
    knn_write_callback_t write_fn;
    knn_flush_callback_t flush_fn;

    KnnCallbackIOWriter(void* _ctx, knn_write_callback_t _write_fn, knn_flush_callback_t _flush_fn)
        : ctx(_ctx), write_fn(_write_fn), flush_fn(_flush_fn) {
        name = "KnnCallbackIOWriter";
    }

    size_t operator()(const void* ptr, size_t size, size_t nitems) override {
        // Faiss passes (size, nitems); we pass total bytes to the Rust callback
        size_t total = size * nitems;
        size_t bytes_written = write_fn(ctx, ptr, total);
        // Return nitems (Faiss expects item count)
        return (size > 0) ? bytes_written / size : 0;
    }

    int filedescriptor() override {
        return -1;
    }
};

/// Create a Faiss IOReader from a read callback + context pointer.
void* knn_shim_create_io_reader(void* ctx, knn_read_callback_t read_fn) {
    return new KnnCallbackIOReader(ctx, read_fn);
}

/// Create a Faiss IOWriter from write/flush callbacks + context pointer.
void* knn_shim_create_io_writer(void* ctx, knn_write_callback_t write_fn, knn_flush_callback_t flush_fn) {
    return new KnnCallbackIOWriter(ctx, write_fn, flush_fn);
}

/// Free an IOReader created by knn_shim_create_io_reader.
void knn_shim_free_io_reader(void* reader_ptr) {
    delete reinterpret_cast<KnnCallbackIOReader*>(reader_ptr);
}

/// Free an IOWriter created by knn_shim_create_io_writer.
void knn_shim_free_io_writer(void* writer_ptr) {
    delete reinterpret_cast<KnnCallbackIOWriter*>(writer_ptr);
}

/// Flush an IOWriter (calls the flush callback).
void knn_shim_flush_io_writer(void* writer_ptr) {
    auto* writer = reinterpret_cast<KnnCallbackIOWriter*>(writer_ptr);
    if (writer && writer->flush_fn) {
        writer->flush_fn(writer->ctx);
    }
}

/// Read an index from an IOReader. Returns the index pointer (caller owns).
void* knn_shim_read_index_from_reader(void* reader_ptr, int io_flags) {
    auto* reader = reinterpret_cast<KnnCallbackIOReader*>(reader_ptr);
    if (!reader) return nullptr;
    return faiss::read_index(reader, io_flags);
}

/// Read a binary index from an IOReader.
void* knn_shim_read_binary_index_from_reader(void* reader_ptr, int io_flags) {
    auto* reader = reinterpret_cast<KnnCallbackIOReader*>(reader_ptr);
    if (!reader) return nullptr;
    return faiss::read_index_binary(reader, io_flags);
}

/// Write an index to an IOWriter.
void knn_shim_write_index_to_writer(void* index_ptr, void* writer_ptr) {
    auto* index = reinterpret_cast<faiss::Index*>(index_ptr);
    auto* writer = reinterpret_cast<KnnCallbackIOWriter*>(writer_ptr);
    if (index && writer) {
        faiss::write_index(index, writer);
    }
}

/// Write a binary index to an IOWriter.
void knn_shim_write_binary_index_to_writer(void* index_ptr, void* writer_ptr) {
    auto* index = reinterpret_cast<faiss::IndexBinary*>(index_ptr);
    auto* writer = reinterpret_cast<KnnCallbackIOWriter*>(writer_ptr);
    if (index && writer) {
        faiss::write_index_binary(index, writer);
    }
}

// ---------------------------------------------------------------------------
// 3b. NMSLIB IndexWrapper and stream reader for the C shim
// ---------------------------------------------------------------------------

/// Wrapper struct that holds an NMSLIB space and index together.
/// Mirrors knn_jni::nmslib_wrapper::IndexWrapper from the main codebase.
struct NmslibIndexWrapper {
    similarity::ObjectVector data;  // empty; needed for method creation
    similarity::Space<float>* space = nullptr;
    similarity::Index<float>* index = nullptr;

    ~NmslibIndexWrapper() {
        delete index;
        delete space;
    }
};

/// An NmslibIOReader backed by a std::istream (from CallbackInputStreambuf).
/// Used for nmslib_load_index_with_stream when the reader is an IStreamWrapper.
struct IStreamNmslibIOReader : public similarity::NmslibIOReader {
    std::istream* is;

    explicit IStreamNmslibIOReader(std::istream* _is) : is(_is) {}

    void read(char* bytes, size_t len) override {
        is->read(bytes, len);
    }

    size_t remainingBytes() override {
        // For stream-based readers, we cannot easily determine remaining bytes.
        // Return a large value to indicate "not at EOF".
        if (is->eof() || is->fail()) return 0;
        return std::numeric_limits<size_t>::max();
    }
};

// ---------------------------------------------------------------------------
// 4. NMSLIB stream support (callback-based I/O)
// ---------------------------------------------------------------------------

// NMSLIB uses std::ostream/std::istream for serialization. We provide a
// custom streambuf that delegates to Rust callbacks.

#include <streambuf>
#include <ostream>
#include <istream>

class CallbackOutputStreambuf : public std::streambuf {
    void* ctx;
    knn_write_callback_t write_fn;
    char buffer[8192];

public:
    CallbackOutputStreambuf(void* _ctx, knn_write_callback_t _write_fn)
        : ctx(_ctx), write_fn(_write_fn) {
        setp(buffer, buffer + sizeof(buffer));
    }

    ~CallbackOutputStreambuf() override {
        sync();
    }

protected:
    int overflow(int ch) override {
        if (pptr() > pbase()) {
            size_t n = pptr() - pbase();
            write_fn(ctx, pbase(), n);
        }
        setp(buffer, buffer + sizeof(buffer));
        if (ch != EOF) {
            *pptr() = (char)ch;
            pbump(1);
        }
        return ch;
    }

    int sync() override {
        if (pptr() > pbase()) {
            size_t n = pptr() - pbase();
            write_fn(ctx, pbase(), n);
            setp(buffer, buffer + sizeof(buffer));
        }
        return 0;
    }
};

class CallbackInputStreambuf : public std::streambuf {
    void* ctx;
    knn_read_callback_t read_fn;
    char buffer[8192];

public:
    CallbackInputStreambuf(void* _ctx, knn_read_callback_t _read_fn)
        : ctx(_ctx), read_fn(_read_fn) {
        setg(buffer, buffer, buffer); // empty initially
    }

protected:
    int underflow() override {
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }
        size_t n = read_fn(ctx, buffer, sizeof(buffer));
        if (n == 0) {
            return EOF;
        }
        setg(buffer, buffer, buffer + n);
        return traits_type::to_int_type(*gptr());
    }
};

/// Create an ostream backed by a write callback (for NMSLIB SaveIndex).
void* knn_shim_create_ostream(void* ctx, knn_write_callback_t write_fn) {
    auto* buf = new CallbackOutputStreambuf(ctx, write_fn);
    auto* os = new std::ostream(buf);
    // Store both so we can free them. Pack into a simple struct.
    struct OStreamWrapper { std::ostream* os; CallbackOutputStreambuf* buf; };
    auto* wrapper = new OStreamWrapper{os, buf};
    return wrapper;
}

/// Get the raw ostream pointer from the wrapper (for passing to NMSLIB).
void* knn_shim_get_ostream_ptr(void* wrapper_ptr) {
    struct OStreamWrapper { std::ostream* os; CallbackOutputStreambuf* buf; };
    auto* wrapper = reinterpret_cast<OStreamWrapper*>(wrapper_ptr);
    return wrapper ? wrapper->os : nullptr;
}

/// Free an ostream created by knn_shim_create_ostream.
void knn_shim_free_ostream(void* wrapper_ptr) {
    struct OStreamWrapper { std::ostream* os; CallbackOutputStreambuf* buf; };
    auto* wrapper = reinterpret_cast<OStreamWrapper*>(wrapper_ptr);
    if (wrapper) {
        delete wrapper->os;
        delete wrapper->buf;
        delete wrapper;
    }
}

/// Create an istream backed by a read callback (for NMSLIB LoadIndex).
void* knn_shim_create_istream(void* ctx, knn_read_callback_t read_fn) {
    auto* buf = new CallbackInputStreambuf(ctx, read_fn);
    auto* is = new std::istream(buf);
    struct IStreamWrapper { std::istream* is; CallbackInputStreambuf* buf; };
    auto* wrapper = new IStreamWrapper{is, buf};
    return wrapper;
}

/// Get the raw istream pointer from the wrapper.
void* knn_shim_get_istream_ptr(void* wrapper_ptr) {
    struct IStreamWrapper { std::istream* is; CallbackInputStreambuf* buf; };
    auto* wrapper = reinterpret_cast<IStreamWrapper*>(wrapper_ptr);
    return wrapper ? wrapper->is : nullptr;
}

/// Free an istream created by knn_shim_create_istream.
void knn_shim_free_istream(void* wrapper_ptr) {
    struct IStreamWrapper { std::istream* is; CallbackInputStreambuf* buf; };
    auto* wrapper = reinterpret_cast<IStreamWrapper*>(wrapper_ptr);
    if (wrapper) {
        delete wrapper->is;
        delete wrapper->buf;
        delete wrapper;
    }
}

// ---------------------------------------------------------------------------
// 5. ADC index loading (LoadIndexWithStreamADC)
// ---------------------------------------------------------------------------
// This is the most complex operation: loads a binary HNSW index, extracts
// the graph structure, wraps it in a float IndexHNSW with a custom distance
// computer (FaissIndexBQ). Too complex for pure FFI — exposed as one call.

// Forward declare FaissIndexBQ from the main knn code
// (This will need to link against the knn faiss wrapper library or be inlined here)
// For now, we provide a stub that returns nullptr, requiring the main build to supply it.

void* knn_shim_load_index_adc(void* io_reader_ptr, int metric_type) {
    // This operation requires FaissIndexBQ which is part of the k-NN custom code.
    // It needs the binary index loaded, HNSW graph extracted, and wrapped.
    // TODO: Link against the knn faiss extension code to implement this.
    (void)io_reader_ptr;
    (void)metric_type;
    return nullptr;
}

// ---------------------------------------------------------------------------
// 6. Merge interrupt callback
// ---------------------------------------------------------------------------

/// Set the Faiss abort callback that checks Java's MergeAbortChecker.
/// This requires a JNIEnv pointer to call back into Java.
void knn_shim_set_merge_interrupt_callback(void* jni_env_ptr) {
    // The original C++ sets a faiss::InterruptCallback that calls
    // MergeAbortChecker.isMergeAborted() via JNI. We store the env
    // and install the callback.
    // TODO: Implement when needed. For now, no-op (merges won't be abortable).
    (void)jni_env_ptr;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// 7. SQ index operations (stubs — full implementation needs faiss_sq_flat.h)
// ---------------------------------------------------------------------------

extern "C" {

int64_t knn_shim_init_sq_index(int32_t total_live_docs, int32_t dim, float centroid_dp, int32_t quantized_vec_bytes) {
    // TODO: Instantiate FaissSQHnsw or FaissSQFlat based on parameters.
    // For now, return 0 (failure) — the Java side will get an error.
    (void)total_live_docs; (void)dim; (void)centroid_dp; (void)quantized_vec_bytes;
    return 0;
}

void knn_shim_sq_add_docs(int64_t index_memory_address, int32_t num_docs, int32_t num_added) {
    (void)index_memory_address; (void)num_docs; (void)num_added;
}

void knn_shim_sq_pass_vectors(int64_t index_memory_address, const uint8_t* buffer, int32_t num_elements) {
    (void)index_memory_address; (void)buffer; (void)num_elements;
}

void knn_shim_sq_release_index(int64_t index_memory_address) {
    (void)index_memory_address;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Aliases for stream_support.rs (uses different names than knn_shim_*)
// ---------------------------------------------------------------------------

extern "C" {

void* faiss_io_reader_from_callback(void* ctx, knn_read_callback_t read_fn) {
    return knn_shim_create_io_reader(ctx, read_fn);
}

void* faiss_io_writer_from_callback(void* ctx, knn_write_callback_t write_fn, knn_flush_callback_t flush_fn) {
    return knn_shim_create_io_writer(ctx, write_fn, flush_fn);
}

void faiss_io_reader_free(void* reader) {
    knn_shim_free_io_reader(reader);
}

void faiss_io_writer_free(void* writer) {
    knn_shim_free_io_writer(writer);
}

// NMSLIB real implementations for functions referenced from nmslib_service_jni.rs
void* nmslib_create_index_wrapper(const char* space_type) {
    if (!space_type) return nullptr;
    try {
        std::string spaceTypeCpp(space_type);
        auto* wrapper = new NmslibIndexWrapper();
        wrapper->space = similarity::SpaceFactoryRegistry<float>::Instance().CreateSpace(
            spaceTypeCpp, similarity::AnyParams());
        // Create the HNSW method with an empty data vector (for loading)
        wrapper->index = similarity::MethodFactoryRegistry<float>::Instance().CreateMethod(
            false, "hnsw", spaceTypeCpp, *(wrapper->space), wrapper->data);
        return wrapper;
    } catch (const std::exception& e) {
        fprintf(stderr, "[nmslib_shim] nmslib_create_index_wrapper exception: %s\n", e.what());
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void nmslib_free_index_wrapper(void* wrapper) {
    if (!wrapper) return;
    auto* w = reinterpret_cast<NmslibIndexWrapper*>(wrapper);
    delete w;
}

int nmslib_load_index_with_stream(void* wrapper, void* reader, const char** params, int num_params) {
    if (!wrapper || !reader) return -1;
    try {
        auto* w = reinterpret_cast<NmslibIndexWrapper*>(wrapper);

        // The reader is an IStreamWrapper* created by knn_shim_create_istream.
        // Extract the raw istream pointer from it.
        struct IStreamWrapper { std::istream* is; CallbackInputStreambuf* buf; };
        auto* istreamWrapper = reinterpret_cast<IStreamWrapper*>(reader);
        std::istream* is = istreamWrapper->is;

        // Create an NmslibIOReader adapter over the istream
        IStreamNmslibIOReader ioReader(is);

        // Set query time params
        std::vector<std::string> queryParams;
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                queryParams.push_back(std::string(params[i]));
            }
        }
        if (!queryParams.empty()) {
            w->index->SetQueryTimeParams(similarity::AnyParams(queryParams));
        }

        // Load via stream - must downcast to Hnsw
        auto* hnswIndex = dynamic_cast<similarity::Hnsw<float>*>(w->index);
        if (!hnswIndex) {
            fprintf(stderr, "[nmslib_shim] Index is not HNSW type\n");
            return -1;
        }
        hnswIndex->LoadIndexWithStream(ioReader);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[nmslib_shim] nmslib_load_index_with_stream exception: %s\n", e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Missing C API aliases (these exist on the base class but Rust declares them on IndexIDMap)
// ---------------------------------------------------------------------------
extern "C" {

int faiss_IndexIDMap_add_with_ids(void* id_map, int64_t n, const float* x, const int64_t* ids) {
    auto* index = reinterpret_cast<faiss::Index*>(id_map);
    if (!index) return -1;
    try {
        index->add_with_ids(n, x, ids);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[knn_shim] add_with_ids exception: %s\n", e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

void faiss_IndexIDMap_set_own_fields(void* id_map, int val) {
    auto* index = dynamic_cast<faiss::IndexIDMap*>(reinterpret_cast<faiss::Index*>(id_map));
    if (index) index->own_fields = (val != 0);
}

void* faiss_IndexIDMap_sub_index(void* id_map) {
    auto* index = dynamic_cast<faiss::IndexIDMap*>(reinterpret_cast<faiss::Index*>(id_map));
    return index ? index->index : nullptr;
}

int faiss_IndexIDMap_new(void** p_id_map, void* sub_index) {
    try {
        auto* idx = new faiss::IndexIDMap(reinterpret_cast<faiss::Index*>(sub_index));
        *p_id_map = idx;
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[knn_shim] add_with_ids exception: %s\n", e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}

int faiss_Index_is_trained(void* index) {
    auto* idx = reinterpret_cast<faiss::Index*>(index);
    return (idx && idx->is_trained) ? 1 : 0;
}

void faiss_Index_free(void* index) {
    delete reinterpret_cast<faiss::Index*>(index);
}

void faiss_index_binary_free(void* index) {
    delete reinterpret_cast<faiss::IndexBinary*>(index);
}

}  // extern "C"

// Auto-generated stubs for missing link symbols
extern "C" {
int faiss_index_add_with_ids(void* idx, int64_t n, const float* x, const int64_t* ids) {
    auto* index = reinterpret_cast<faiss::Index*>(idx); if (!index) return -1;
    try { index->add_with_ids(n, x, ids); return 0; } catch (...) { return -1; }
}
int faiss_index_search(void* idx, int64_t n, const float* x, int64_t k, float* d, int64_t* labels) {
    auto* index = reinterpret_cast<faiss::Index*>(idx); if (!index) return -1;
    try { index->search(n, x, k, d, labels); return 0; } catch (...) { return -1; }
}
int faiss_index_train(void* idx, int64_t n, const float* x) {
    auto* index = reinterpret_cast<faiss::Index*>(idx); if (!index) return -1;
    try { index->train(n, x); return 0; } catch (...) { return -1; }
}
int faiss_index_is_trained(void* idx) {
    auto* index = reinterpret_cast<faiss::Index*>(idx);
    return (index && index->is_trained) ? 1 : 0;
}
int faiss_index_metric_type(void* idx) {
    auto* index = reinterpret_cast<faiss::Index*>(idx);
    return index ? (int)index->metric_type : -1;
}
void faiss_index_free(void* idx) { delete reinterpret_cast<faiss::Index*>(idx); }
int faiss_index_range_search(void* idx, int64_t n, const float* x, float radius, void* result) {
    (void)idx; (void)n; (void)x; (void)radius; (void)result; return -1; // stub
}
int faiss_index_id_map_new(void** out, void* sub) {
    try { *out = new faiss::IndexIDMap(reinterpret_cast<faiss::Index*>(sub)); return 0; } catch (...) { return -1; }
}
int faiss_index_binary_add_with_ids(void* idx, int64_t n, const uint8_t* x, const int64_t* ids) {
    auto* index = reinterpret_cast<faiss::IndexBinary*>(idx); if (!index) return -1;
    try { index->add_with_ids(n, x, ids); return 0; } catch (...) { return -1; }
}
int faiss_index_binary_search(void* idx, int64_t n, const uint8_t* x, int64_t k, int32_t* d, int64_t* labels) {
    auto* index = reinterpret_cast<faiss::IndexBinary*>(idx); if (!index) return -1;
    try { index->search(n, x, k, d, labels); return 0; } catch (...) { return -1; }
}
int faiss_index_binary_train(void* idx, int64_t n, const uint8_t* x) {
    auto* index = reinterpret_cast<faiss::IndexBinary*>(idx); if (!index) return -1;
    try { index->train(n, x); return 0; } catch (...) { return -1; }
}
int faiss_index_binary_is_trained(void* idx) {
    auto* index = reinterpret_cast<faiss::IndexBinary*>(idx);
    return (index && index->is_trained) ? 1 : 0;
}
int faiss_index_binary_id_map_new(void** out, void* sub) {
    try { *out = new faiss::IndexBinaryIDMap(reinterpret_cast<faiss::IndexBinary*>(sub)); return 0; } catch (...) { return -1; }
}
void* faiss_index_to_hnsw(void* idx) { return dynamic_cast<faiss::IndexHNSW*>(reinterpret_cast<faiss::Index*>(idx)); }
void* faiss_index_to_ivf(void* idx) { return dynamic_cast<faiss::IndexIVF*>(reinterpret_cast<faiss::Index*>(idx)); }
void* faiss_index_to_ivfpq(void* idx) { return dynamic_cast<faiss::IndexIVFPQ*>(reinterpret_cast<faiss::Index*>(idx)); }
void faiss_index_hnsw_set_ef_construction(void* hnsw, int val) { if(hnsw) ((faiss::IndexHNSW*)hnsw)->hnsw.efConstruction = val; }
void faiss_index_hnsw_set_ef_search(void* hnsw, int val) { if(hnsw) ((faiss::IndexHNSW*)hnsw)->hnsw.efSearch = val; }
void faiss_index_ivf_set_nprobe(void* ivf, int val) { if(ivf) ((faiss::IndexIVF*)ivf)->nprobe = val; }
void* faiss_vector_io_reader_new() { try { return new faiss::VectorIOReader(); } catch (...) { return nullptr; } }
void faiss_vector_io_reader_free(void* r) { delete reinterpret_cast<faiss::VectorIOReader*>(r); }
void faiss_vector_io_reader_set_data(void* r, const uint8_t* data, size_t n) { auto* vr = (faiss::VectorIOReader*)r; vr->data.assign(data, data+n); }
int faiss_read_index_from_vector_io_reader(void** out, void* reader, int flags) { try { *out = faiss::read_index((faiss::VectorIOReader*)reader, flags); return 0; } catch (...) { return -1; } }
int faiss_read_index_binary_from_vector_io_reader(void** out, void* reader, int flags) { try { *out = faiss::read_index_binary((faiss::VectorIOReader*)reader, flags); return 0; } catch (...) { return -1; } }
int faiss_read_index_from_reader(void** out, void* reader, int flags) { try { *out = faiss::read_index((faiss::IOReader*)reader, flags); return 0; } catch (...) { return -1; } }
int faiss_read_index_binary_from_reader(void** out, void* reader, int flags) { try { *out = faiss::read_index_binary((faiss::IOReader*)reader, flags); return 0; } catch (...) { return -1; } }
void* faiss_vector_io_writer_new() { try { return new faiss::VectorIOWriter(); } catch (...) { return nullptr; } }
void faiss_vector_io_writer_free(void* w) { delete reinterpret_cast<faiss::VectorIOWriter*>(w); }
void faiss_vector_io_writer_get_data(void* w, const uint8_t** data, size_t* len) { auto* vw = (faiss::VectorIOWriter*)w; *data = vw->data.data(); *len = vw->data.size(); }
int faiss_write_index_to_vector_io_writer(void* idx, void* writer) { try { faiss::write_index((faiss::Index*)idx, (faiss::VectorIOWriter*)writer); return 0; } catch (...) { return -1; } }
int faiss_write_index_binary_to_vector_io_writer(void* idx, void* writer) { try { faiss::write_index_binary((faiss::IndexBinary*)idx, (faiss::VectorIOWriter*)writer); return 0; } catch (...) { return -1; } }
int faiss_write_index_to_IOWriter(void* idx, void* writer) { try { faiss::write_index((faiss::Index*)idx, (faiss::IOWriter*)writer); return 0; } catch (...) { return -1; } }
int faiss_write_index_binary_to_IOWriter(void* idx, void* writer) { try { faiss::write_index_binary((faiss::IndexBinary*)idx, (faiss::IOWriter*)writer); return 0; } catch (...) { return -1; } }
int faiss_range_search_result_new(void** out, int64_t nq) { (void)out; (void)nq; return -1; }
void faiss_range_search_result_free(void* r) { (void)r; }
int faiss_range_search_result_get_lims(void* r, size_t** lims) { (void)r; (void)lims; return -1; }
int faiss_range_search_result_get_labels(void* r, int64_t** labels) { (void)r; (void)labels; return -1; }
int faiss_range_search_result_get_distances(void* r, float** distances) { (void)r; (void)distances; return -1; }
void knn_simd_calculate_similarity(void* ctx, int id, float* score) { (void)ctx; (void)id; *score = 0; }
void knn_simd_calculate_similarity_in_bulk(void* ctx, int32_t* ids, float* scores, int n) { (void)ctx; (void)ids; (void)scores; (void)n; }
void* knn_simd_get_search_context() { return nullptr; }
void knn_simd_save_search_context(void* ctx) { (void)ctx; }
void knn_simd_set_sq_correction_factors(void* ctx, const void* factors, int n) { (void)ctx; (void)factors; (void)n; }
void nmslib_init_library() {
    similarity::initLibrary();
}
int nmslib_load_index(void* w, const char* path, const char** params, int n) {
    if (!w || !path) return -1;
    try {
        auto* wrapper = reinterpret_cast<NmslibIndexWrapper*>(w);

        // Load the index from file
        wrapper->index->LoadIndex(std::string(path));

        // Set query time params (e.g., "efSearch=100")
        std::vector<std::string> queryParams;
        for (int i = 0; i < n; i++) {
            if (params && params[i]) {
                queryParams.push_back(std::string(params[i]));
            }
        }
        if (!queryParams.empty()) {
            wrapper->index->SetQueryTimeParams(similarity::AnyParams(queryParams));
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[nmslib_shim] nmslib_load_index exception: %s\n", e.what());
        return -1;
    } catch (...) {
        return -1;
    }
}
int nmslib_query_index(void* w, const float* q, int d, int k, int ef, int* ids, float* dists) {
    if (!w || !q || !ids || !dists) return 0;
    try {
        auto* wrapper = reinterpret_cast<NmslibIndexWrapper*>(w);

        // Create query object: Object(id, label, datalength, data)
        std::unique_ptr<const similarity::Object> queryObject(
            new similarity::Object(-1, -1, d * sizeof(float), q));

        // Create KNNQuery or HNSWQuery depending on ef_search
        std::unique_ptr<similarity::KNNQuery<float>> query;
        if (ef > 0) {
            query.reset(new similarity::HNSWQuery<float>(
                *(wrapper->space), queryObject.get(), (unsigned)k, (unsigned)ef));
        } else {
            query.reset(new similarity::KNNQuery<float>(
                *(wrapper->space), queryObject.get(), (unsigned)k));
        }

        // Execute search
        wrapper->index->Search(query.get());

        // Extract results from the priority queue (max-heap by distance)
        std::unique_ptr<similarity::KNNQueue<float>> neighbors(query->Result()->Clone());
        int resultSize = (int)neighbors->Size();

        // Results come out in descending distance order (max-heap),
        // fill arrays from the end to get ascending order
        for (int i = resultSize - 1; i >= 0; --i) {
            dists[i] = neighbors->TopDistance();
            const similarity::Object* obj = neighbors->Pop();
            ids[i] = (int)obj->id();
        }

        return resultSize;
    } catch (const std::exception& e) {
        fprintf(stderr, "[nmslib_shim] nmslib_query_index exception: %s\n", e.what());
        return 0;
    } catch (...) {
        return 0;
    }
}
void faiss_IndexIDMap_free(void* idx) { delete reinterpret_cast<faiss::IndexIDMap*>(idx); }
void faiss_IndexBinaryIDMap_free(void* idx) { delete reinterpret_cast<faiss::IndexBinaryIDMap*>(idx); }
void faiss_IndexBinaryIDMap_set_own_fields(void* idx, int val) { auto* m = dynamic_cast<faiss::IndexBinaryIDMap*>(reinterpret_cast<faiss::IndexBinary*>(idx)); if(m) m->own_fields = (val!=0); }
int faiss_IndexBinaryIDMap_add_with_ids(void* idx, int64_t n, const uint8_t* x, const int64_t* ids) { auto* index = reinterpret_cast<faiss::IndexBinary*>(idx); if (!index) return -1; try { index->add_with_ids(n, x, ids); return 0; } catch (...) { return -1; } }
int faiss_IndexBinaryIDMap_new(void** out, void* sub) { try { *out = new faiss::IndexBinaryIDMap(reinterpret_cast<faiss::IndexBinary*>(sub)); return 0; } catch (...) { return -1; } }
}

// Fix: faiss_read_index from file path (the Rust code calls this with a path string)
// The official C API uses faiss_read_index_fname for this, but our Rust FFI declares faiss_read_index
extern "C" int faiss_read_index(const char* fname, int io_flags, void** p_out) {
    if (!fname || !p_out) return -1;
    try {
        *p_out = faiss::read_index(fname, io_flags);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[shim] faiss_read_index(%s) exception: %s\n", fname, e.what());
        *p_out = nullptr;
        return -1;
    } catch (...) {
        *p_out = nullptr;
        return -1;
    }
}

extern "C" int faiss_read_index_binary(const char* fname, int io_flags, void** p_out) {
    if (!fname || !p_out) return -1;
    try {
        *p_out = faiss::read_index_binary(fname, io_flags);
        return 0;
    } catch (...) {
        *p_out = nullptr;
        return -1;
    }
}
