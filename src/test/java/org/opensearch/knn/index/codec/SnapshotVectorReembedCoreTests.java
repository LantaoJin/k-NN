/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
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

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Non-cluster proof of the CORE of the snapshot vector re-embed flow for <b>faiss + on_disk</b>
 * (binary-quantized) KNN vectors.
 *
 * <p>The cluster-based {@code SnapshotVectorReembedIT} can't run in every environment (it needs a
 * test node + free JMX port). This unit test isolates the part that actually matters and is unique
 * to faiss/on_disk: <i>can the k-NN codec regenerate a valid native Faiss binary-quantized segment
 * when the vector field is rebuilt with brand-new vectors?</i> That is exactly what the re-embed
 * tool relies on — it never byte-patches {@code .faiss}; it re-indexes the field and lets the codec
 * (re-)quantize and (re-)build the native index.
 *
 * <p>This drives an in-process Lucene {@link RandomIndexWriter} with the k-NN
 * {@link NativeEngines990KnnVectorsFormat} (the format faiss/on_disk uses: {@code BHNSW32} +
 * 1-bit scalar quantization), mirroring {@code NativeEngines990KnnVectorsFormatTests}. It builds a
 * segment with ORIGINAL vectors, then a second segment with REPLACEMENT vectors (the "new embedding
 * model"), and asserts the replacement segment is a real {@code .faiss} index whose float vectors
 * read back as the new values.
 */
public class SnapshotVectorReembedCoreTests extends KNNTestCase {

    private static final Codec CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(0));
    private static final String VEC = "vec";
    private static final String ID = "id";
    private static final int DIM = 8;
    private static final int NUM_DOCS = 6;
    private static final String FAISS_EXT = ".faiss";

    /** Original embedding for doc i. */
    private static float[] origVec(int i) {
        float[] v = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            v[d] = (i + 1) + d; // distinct, non-zero
        }
        return v;
    }

    /** PoC "new embedding model": full replacement = original + 100. */
    private static float[] reembed(float[] old) {
        float[] v = new float[old.length];
        for (int d = 0; d < old.length; d++) {
            v[d] = old[d] + 100f;
        }
        return v;
    }

    @SneakyThrows
    public void testReembedRegeneratesFaissOnDiskField() {
        try (Directory dir = newFSDirectory(createTempDir())) {
            // Native engine format has no codec-level search wired in unit tests; skip check-on-close.
            ((BaseDirectoryWrapper) dir).setCheckIndexOnClose(false);

            // ---- build the ORIGINAL faiss/on_disk segment ----
            try (RandomIndexWriter w = newWriter(dir)) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    w.addDocument(doc(i, origVec(i)));
                }
                w.flush();
                w.commit();

                try (IndexReader reader = w.getReader()) {
                    // a real native faiss index file exists for the field
                    List<String> faissFiles = filesWithExt(dir, FAISS_EXT);
                    assertEquals("expected one .faiss file for the vector field", 1, faissFiles.size());
                    assertTrue(faissFiles.get(0).contains(VEC));

                    // and the original float vectors read back exactly
                    assertVector(reader, "doc-0", origVec(0));
                    assertVector(reader, "doc-3", origVec(3));
                }
            }
        }

        // ---- re-embed: build a NEW index with REPLACEMENT vectors via the SAME codec ----
        // (models the tool writing the rebuilt field into a fresh index; the codec re-quantizes
        // and rebuilds the native faiss index for the new vectors.)
        try (Directory destDir = newFSDirectory(createTempDir())) {
            ((BaseDirectoryWrapper) destDir).setCheckIndexOnClose(false);
            try (RandomIndexWriter w = newWriter(destDir)) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    w.addDocument(doc(i, reembed(origVec(i))));
                }
                w.flush();
                w.commit();

                try (IndexReader reader = w.getReader()) {
                    // still a real faiss/on_disk segment...
                    List<String> faissFiles = filesWithExt(destDir, FAISS_EXT);
                    assertEquals(1, faissFiles.size());
                    assertTrue(faissFiles.get(0).contains(VEC));

                    // ...and every vector is now the re-embedded value, none of the originals remain.
                    for (int i = 0; i < NUM_DOCS; i++) {
                        assertVector(reader, "doc-" + i, reembed(origVec(i)));
                    }

                    // sanity: a search reader opens cleanly over the regenerated segment
                    IndexSearcher searcher = new IndexSearcher(reader);
                    assertEquals(NUM_DOCS, searcher.getIndexReader().numDocs());
                }
            }
        }
    }

    // ---- helpers ----

    private RandomIndexWriter newWriter(Directory dir) throws Exception {
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setMergeScheduler(new SerialMergeScheduler());
        iwc.setCodec(CODEC);
        iwc.setUseCompoundFile(false);
        iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        return new RandomIndexWriter(random(), dir, iwc);
    }

    private Document doc(int i, float[] vector) {
        Document d = new Document();
        d.add(new StringField(ID, "doc-" + i, Field.Store.YES));
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
        // BHNSW32 == binary-quantized HNSW (this is what on_disk/32x compression produces for faiss)
        ft.putAttribute(KNNConstants.PARAMETERS, "{ \"index_description\":\"BHNSW32\", \"spaceType\": \"l2\"}");
        QuantizationConfig qc = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build();
        ft.putAttribute(KNNConstants.QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(qc));
        ft.setVectorAttributes(DIM, VectorEncoding.FLOAT32, SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction());
        ft.freeze();
        return ft;
    }

    private static List<String> filesWithExt(Directory dir, String ext) throws Exception {
        return Arrays.stream(dir.listAll()).filter(f -> f.contains(ext)).collect(Collectors.toList());
    }

    /** Find the doc by stored id and assert its float vector equals expected. */
    private static void assertVector(IndexReader reader, String idValue, float[] expected) throws Exception {
        for (var ctx : reader.leaves()) {
            LeafReader leaf = ctx.reader();
            FloatVectorValues values = leaf.getFloatVectorValues(VEC);
            if (values == null) {
                continue;
            }
            KnnVectorValues.DocIndexIterator it = values.iterator();
            var stored = leaf.storedFields();
            for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                if (idValue.equals(stored.document(doc).get(ID))) {
                    assertArrayEquals("vector for " + idValue, expected, values.vectorValue(it.index()), 0.0f);
                    return;
                }
            }
        }
        fail("doc not found: " + idValue);
    }
}
