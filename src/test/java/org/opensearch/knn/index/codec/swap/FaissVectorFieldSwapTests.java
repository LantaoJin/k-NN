/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.swap;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Proves the vector-field swap works against the <b>native FAISS</b> engine — the engine OpenSearch
 * k-NN uses by default — and not only against Lucene HNSW.
 *
 * <p>The mechanism under test is {@link VectorFieldSubstitutingCodecReader} passed to
 * {@link IndexWriter#addIndexes(CodecReader...)}. The claim being verified is that k-NN's native
 * writer consumes vectors through {@code mergeState.knnVectorsReaders}, so substituting at the
 * {@code KnnVectorsReader} seam causes FAISS to re-quantize and rebuild its native index from the new
 * values, while every non-vector field is carried across by the merge's bulk-copy path.
 *
 * <p>Distinct from {@code SnapshotVectorReembedCoreTests}: that test builds a second index from
 * scratch with new vectors (which requires possessing every field's source value). This test performs
 * a real <i>swap</i> — the destination is built from the source segment, so the non-vector fields are
 * never re-supplied and never re-analyzed.
 *
 * <p>Runs in-process with the k-NN {@link NativeEngines990KnnVectorsFormat} (what {@code faiss} /
 * {@code on_disk} uses: {@code BHNSW32} + 1-bit scalar quantization), which requires the JNI libs but
 * no cluster, no {@code MapperService} and no {@code ClusterService}.
 */
public class FaissVectorFieldSwapTests extends KNNTestCase {

    private static final Codec CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(0));
    private static final String VEC = "vec";
    private static final String ID = "id";
    private static final int DIM = 8;
    private static final int NUM_DOCS = 6;
    private static final String FAISS_EXT = ".faiss";

    /** The "old model" embedding for doc i. */
    private static float[] oldVec(int i) {
        float[] v = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            v[d] = (i + 1) + d;
        }
        return v;
    }

    /** The "new model" embedding: clearly distinguishable from the old one. */
    private static float[] newVec(int i) {
        float[] v = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            v[d] = oldVec(i)[d] + 100f;
        }
        return v;
    }

    /**
     * The headline test: swap the FAISS vector field in place of a rebuild, and confirm both halves of
     * the claim — new vectors through FAISS, old non-vector data preserved.
     */
    @SneakyThrows
    public void testFaissVectorFieldSwapRebuildsNativeIndexAndPreservesOtherFields() {
        try (Directory src = newFSDirectory(createTempDir()); Directory dest = newFSDirectory(createTempDir())) {
            // The native engine has no codec-level search wired up in unit tests.
            ((BaseDirectoryWrapper) src).setCheckIndexOnClose(false);
            ((BaseDirectoryWrapper) dest).setCheckIndexOnClose(false);

            // ---- source: a real faiss/on_disk segment with the OLD vectors ----
            try (IndexWriter w = new IndexWriter(src, config())) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    w.addDocument(doc(i, oldVec(i)));
                }
                w.commit();
            }
            assertEquals("source must have a native faiss index", 1, filesWithExt(src, FAISS_EXT).size());
            Map<String, Long> srcNonVectorFiles = nonVectorFileSizes(src);
            assertFalse("source must have non-vector files to preserve", srcNonVectorFiles.isEmpty());

            // ---- the swap: build dest FROM src, substituting only the vector field ----
            try (DirectoryReader reader = DirectoryReader.open(src); IndexWriter w = new IndexWriter(dest, config())) {
                List<CodecReader> wrapped = new ArrayList<>();
                for (LeafReaderContext ctx : reader.leaves()) {
                    // Resolve docId -> logical id on this thread; the supplier runs on the merge thread.
                    final int[] ids = docIdToLogicalId(ctx.reader());
                    wrapped.add(
                        new VectorFieldSubstitutingCodecReader((SegmentReader) ctx.reader(), VEC, (docId, dim) -> newVec(ids[docId]))
                    );
                }
                w.addIndexes(wrapped.toArray(new CodecReader[0]));
                w.commit();
            }

            // 1. FAISS actually rebuilt its native index in the destination.
            List<String> destFaiss = filesWithExt(dest, FAISS_EXT);
            assertEquals("destination must have a rebuilt native faiss index", 1, destFaiss.size());
            assertTrue("the faiss file must belong to the vector field", destFaiss.get(0).contains(VEC));

            try (DirectoryReader destReader = DirectoryReader.open(dest)) {
                assertEquals("all documents carried over", NUM_DOCS, destReader.numDocs());

                // 2. Every vector is the NEW model's value, and none of the old values survive.
                for (int i = 0; i < NUM_DOCS; i++) {
                    float[] actual = vectorFor(destReader, "doc-" + i);
                    assertArrayEquals("FAISS must serve the new vector for doc-" + i, newVec(i), actual, 0.0f);
                    assertFalse("doc-" + i + " must not retain its old vector", Arrays.equals(oldVec(i), actual));
                }

                // 3. Non-vector fields survived without being re-supplied.
                try (DirectoryReader srcReader = DirectoryReader.open(src)) {
                    StoredFields srcStored = srcReader.storedFields();
                    StoredFields destStored = destReader.storedFields();
                    for (int i = 0; i < NUM_DOCS; i++) {
                        assertEquals("id preserved", srcStored.document(i).get(ID), destStored.document(i).get(ID));
                        assertEquals("body preserved", srcStored.document(i).get("body"), destStored.document(i).get("body"));
                        assertEquals("payload preserved", srcStored.document(i).get("payload"), destStored.document(i).get("payload"));
                    }
                }
            }
        }
    }

    /**
     * The FAISS write path must receive the substituted vectors specifically through
     * {@code getMergeInstance()}, which is how {@code MergeState} exposes readers to
     * {@code KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues} — the call k-NN's
     * {@code AbstractNativeEnginesKnnVectorsWriter.doMergeOneField} uses.
     *
     * <p>Guards the failure mode where the merge instance reverts to the delegate and the swap
     * silently writes the OLD vectors while reporting success.
     */
    @SneakyThrows
    public void testFaissMergePathSeesSubstitutedVectors() {
        try (Directory src = newFSDirectory(createTempDir())) {
            ((BaseDirectoryWrapper) src).setCheckIndexOnClose(false);
            try (IndexWriter w = new IndexWriter(src, config())) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    w.addDocument(doc(i, oldVec(i)));
                }
                w.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(src)) {
                for (LeafReaderContext ctx : reader.leaves()) {
                    final int[] ids = docIdToLogicalId(ctx.reader());
                    VectorFieldSubstitutingCodecReader wrapped = new VectorFieldSubstitutingCodecReader(
                        (SegmentReader) ctx.reader(),
                        VEC,
                        (docId, dim) -> newVec(ids[docId])
                    );

                    // Exactly MergeState.java:165-167.
                    org.apache.lucene.codecs.KnnVectorsReader mergeInstance = wrapped.getVectorReader().getMergeInstance();
                    assertNotNull(mergeInstance);

                    // Exactly what k-NN's native writer reads.
                    FloatVectorValues values = mergeInstance.getFloatVectorValues(VEC);
                    assertNotNull("the FAISS merge path must see the vector field", values);

                    int seen = 0;
                    KnnVectorValues.DocIndexIterator it = values.iterator();
                    for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                        float[] actual = values.vectorValue(it.index());
                        assertArrayEquals("FAISS writer must receive the new vector", newVec(ids[doc]), actual, 0.0f);
                        assertFalse("FAISS writer must not receive the old vector", Arrays.equals(oldVec(ids[doc]), actual));
                        seen++;
                    }
                    assertEquals("every doc in the leaf was checked", ctx.reader().maxDoc(), seen);
                }
            }
        }
    }

    // ---- helpers ----

    private IndexWriterConfig config() {
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(CODEC);
        iwc.setUseCompoundFile(false);
        iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        return iwc;
    }

    private Document doc(int i, float[] vector) {
        Document d = new Document();
        d.add(new StringField(ID, "doc-" + i, Field.Store.YES));
        // Non-vector content the swap must carry across without re-analysis.
        d.add(new TextField("body", "document " + i + " storage segment merge amplification quantization recall", Field.Store.YES));
        d.add(new StoredField("payload", "payload-for-doc-" + i + "-" + "y".repeat(48)));
        d.add(new NumericDocValuesField("rank", i));
        d.add(new KnnFloatVectorField(VEC, vector, faissOnDiskFieldType()));
        return d;
    }

    /** A faiss/on_disk float vector field: native FAISS engine + BHNSW32 + 1-bit scalar quantization. */
    private FieldType faissOnDiskFieldType() {
        FieldType ft = new FieldType();
        ft.setTokenized(false);
        ft.setIndexOptions(IndexOptions.NONE);
        ft.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        ft.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        ft.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        ft.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        ft.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());
        ft.putAttribute(KNNConstants.PARAMETERS, "{ \"index_description\":\"BHNSW32\", \"spaceType\": \"l2\"}");
        QuantizationConfig qc = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build();
        ft.putAttribute(KNNConstants.QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(qc));
        ft.setVectorAttributes(DIM, VectorEncoding.FLOAT32, SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction());
        ft.freeze();
        return ft;
    }

    private static int[] docIdToLogicalId(LeafReader leaf) throws IOException {
        StoredFields sf = leaf.storedFields();
        int[] ids = new int[leaf.maxDoc()];
        for (int doc = 0; doc < leaf.maxDoc(); doc++) {
            // Stored ids are "doc-<n>".
            ids[doc] = Integer.parseInt(sf.document(doc).get(ID).substring("doc-".length()));
        }
        return ids;
    }

    private static List<String> filesWithExt(Directory dir, String ext) throws Exception {
        return Arrays.stream(dir.listAll()).filter(f -> f.contains(ext)).collect(Collectors.toList());
    }

    /** File name -> length for every file that does not hold vector data. */
    private static Map<String, Long> nonVectorFileSizes(Directory dir) throws IOException {
        Map<String, Long> sizes = new LinkedHashMap<>();
        for (String f : dir.listAll()) {
            if (f.contains(FAISS_EXT)
                || f.endsWith(".vec")
                || f.endsWith(".vex")
                || f.endsWith(".vem")
                || f.endsWith(".vemf")
                || f.endsWith(".vemq")
                || f.endsWith(".veq")) {
                continue;
            }
            sizes.put(f, dir.fileLength(f));
        }
        return sizes;
    }

    private static float[] vectorFor(DirectoryReader reader, String idValue) throws IOException {
        for (LeafReaderContext ctx : reader.leaves()) {
            LeafReader leaf = ctx.reader();
            FloatVectorValues values = leaf.getFloatVectorValues(VEC);
            if (values == null) {
                continue;
            }
            StoredFields stored = leaf.storedFields();
            KnnVectorValues.DocIndexIterator it = values.iterator();
            for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                if (idValue.equals(stored.document(doc).get(ID))) {
                    return values.vectorValue(it.index());
                }
            }
        }
        throw new AssertionError("doc not found: " + idValue);
    }
}
