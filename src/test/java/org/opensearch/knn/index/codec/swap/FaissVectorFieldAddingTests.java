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
import org.apache.lucene.index.FieldInfo;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Verifies {@link VectorFieldAddingCodecReader} against the <b>native FAISS</b> engine: populating a
 * vector field that does not exist in the source segment, so that FAISS builds a real native index for
 * it, while every existing field is bulk-copied.
 *
 * <p>This completes the "field pair + alias flip" story on the engine OpenSearch actually ships. The
 * critical detail FAISS adds over plain Lucene HNSW is that the native writer derives what to build from
 * {@code FieldInfo.attributes()} — {@code NativeIndexWriter.getParameters} reads
 * {@code KNNConstants.PARAMETERS} (carrying {@code index_description}), {@code SPACE_TYPE} and
 * {@code VECTOR_DATA_TYPE_FIELD}, and the quantization framework reads {@code QFRAMEWORK_CONFIG}. A
 * synthesized field with empty attributes would therefore silently produce the wrong index type rather
 * than failing, so these tests assert the attributes survive onto the added field.
 */
public class FaissVectorFieldAddingTests extends KNNTestCase {

    private static final Codec CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(0));
    private static final String OLD_FIELD = "embedding_foo";
    private static final String NEW_FIELD = "embedding_bar";
    private static final String ID = "id";
    private static final int DIM = 8;
    private static final int NUM_DOCS = 6;
    private static final String FAISS_EXT = ".faiss";

    private static float[] oldVec(int i) {
        float[] v = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            v[d] = (i + 1) + d;
        }
        return v;
    }

    private static float[] newVec(int i) {
        float[] v = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            v[d] = oldVec(i)[d] + 100f;
        }
        return v;
    }

    /**
     * The headline test: FAISS builds a native index for a field that did not exist in the source, and
     * the pre-existing FAISS field plus all non-vector data survive untouched.
     */
    @SneakyThrows
    public void testFaissBuildsNativeIndexForAnAddedField() {
        try (Directory src = newFSDirectory(createTempDir()); Directory dest = newFSDirectory(createTempDir())) {
            ((BaseDirectoryWrapper) src).setCheckIndexOnClose(false);
            ((BaseDirectoryWrapper) dest).setCheckIndexOnClose(false);

            // ---- source: one populated faiss/on_disk field; the second is not present at all ----
            try (IndexWriter w = new IndexWriter(src, config())) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    w.addDocument(doc(i));
                }
                w.commit();
            }

            List<String> srcFaiss = filesWithExt(src, FAISS_EXT);
            assertEquals("source has exactly one native faiss index", 1, srcFaiss.size());
            assertTrue("...belonging to the old field", srcFaiss.get(0).contains(OLD_FIELD));

            try (DirectoryReader r = DirectoryReader.open(src)) {
                for (LeafReaderContext ctx : r.leaves()) {
                    assertNull("the new field does not exist yet", ctx.reader().getFloatVectorValues(NEW_FIELD));
                }
            }

            // ---- add the new field, carrying the faiss attributes the native writer needs ----
            try (DirectoryReader reader = DirectoryReader.open(src); IndexWriter w = new IndexWriter(dest, config())) {
                List<CodecReader> wrapped = new ArrayList<>();
                for (LeafReaderContext ctx : reader.leaves()) {
                    final int[] ids = docIdToLogicalId(ctx.reader());
                    wrapped.add(
                        new VectorFieldAddingCodecReader(
                            (SegmentReader) ctx.reader(),
                            NEW_FIELD,
                            DIM,
                            SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction(),
                            faissAttributes(),
                            docId -> newVec(ids[docId])
                        )
                    );
                }
                w.addIndexes(wrapped.toArray(new CodecReader[0]));
                w.commit();
            }

            // 1. FAISS built a SECOND native index — one per field.
            List<String> destFaiss = filesWithExt(dest, FAISS_EXT);
            assertEquals("destination has a native faiss index per vector field", 2, destFaiss.size());
            assertTrue("the added field has its own .faiss file: " + destFaiss, destFaiss.stream().anyMatch(f -> f.contains(NEW_FIELD)));
            assertTrue("the original field keeps its .faiss file: " + destFaiss, destFaiss.stream().anyMatch(f -> f.contains(OLD_FIELD)));

            try (DirectoryReader destReader = DirectoryReader.open(dest)) {
                assertEquals("all documents carried over", NUM_DOCS, destReader.numDocs());

                // 2. The added field serves the new model's vectors.
                for (int i = 0; i < NUM_DOCS; i++) {
                    assertArrayEquals(
                        "added field holds the new vector for doc-" + i,
                        newVec(i),
                        vectorFor(destReader, NEW_FIELD, i),
                        0.0f
                    );
                }

                // 3. The pre-existing FAISS field is untouched — the rollback target.
                for (int i = 0; i < NUM_DOCS; i++) {
                    assertArrayEquals("old field keeps its vector for doc-" + i, oldVec(i), vectorFor(destReader, OLD_FIELD, i), 0.0f);
                }

                // 4. Non-vector fields survived without being re-supplied.
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
     * The FAISS-specific hazard: the native writer configures itself from {@code FieldInfo.attributes()},
     * so those attributes must reach the added field. Without them the engine falls back to defaults and
     * quietly produces a different index type than intended.
     */
    @SneakyThrows
    public void testFaissAttributesReachTheAddedField() {
        try (Directory src = newFSDirectory(createTempDir())) {
            ((BaseDirectoryWrapper) src).setCheckIndexOnClose(false);
            try (IndexWriter w = new IndexWriter(src, config())) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    w.addDocument(doc(i));
                }
                w.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(src)) {
                for (LeafReaderContext ctx : reader.leaves()) {
                    final int[] ids = docIdToLogicalId(ctx.reader());
                    VectorFieldAddingCodecReader wrapped = new VectorFieldAddingCodecReader(
                        (SegmentReader) ctx.reader(),
                        NEW_FIELD,
                        DIM,
                        SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction(),
                        faissAttributes(),
                        docId -> newVec(ids[docId])
                    );

                    FieldInfo added = wrapped.getFieldInfos().fieldInfo(NEW_FIELD);
                    assertNotNull("the added field must be advertised in FieldInfos", added);
                    assertEquals(DIM, added.getVectorDimension());
                    assertEquals(VectorEncoding.FLOAT32, added.getVectorEncoding());

                    // These are exactly the keys NativeIndexWriter.getParameters and the quantization
                    // framework read; a missing one means a silently wrong native index.
                    assertEquals("true", added.attributes().get(KNNVectorFieldMapper.KNN_FIELD));
                    assertEquals(KNNEngine.FAISS.getName(), added.attributes().get(KNNConstants.KNN_ENGINE));
                    assertEquals(SpaceType.L2.getValue(), added.attributes().get(KNNConstants.SPACE_TYPE));
                    assertEquals(VectorDataType.FLOAT.getValue(), added.attributes().get(KNNConstants.VECTOR_DATA_TYPE_FIELD));
                    assertNotNull("parameters carry index_description", added.attributes().get(KNNConstants.PARAMETERS));
                    assertTrue(
                        "index_description must describe the intended faiss index",
                        added.attributes().get(KNNConstants.PARAMETERS).contains("BHNSW32")
                    );
                    assertNotNull("quantization config must be present", added.attributes().get(KNNConstants.QFRAMEWORK_CONFIG));

                    // The pre-existing field's attributes are untouched.
                    FieldInfo old = wrapped.getFieldInfos().fieldInfo(OLD_FIELD);
                    assertEquals(KNNEngine.FAISS.getName(), old.attributes().get(KNNConstants.KNN_ENGINE));
                }
            }
        }
    }

    /**
     * The added field must survive {@code getMergeInstance()}, which is how {@code MergeState} exposes
     * readers to k-NN's native writer. Same trap that produced a real bug in the substituting reader.
     */
    @SneakyThrows
    public void testAddedFieldSurvivesFaissMergePath() {
        try (Directory src = newFSDirectory(createTempDir())) {
            ((BaseDirectoryWrapper) src).setCheckIndexOnClose(false);
            try (IndexWriter w = new IndexWriter(src, config())) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    w.addDocument(doc(i));
                }
                w.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(src)) {
                for (LeafReaderContext ctx : reader.leaves()) {
                    final int[] ids = docIdToLogicalId(ctx.reader());
                    VectorFieldAddingCodecReader wrapped = new VectorFieldAddingCodecReader(
                        (SegmentReader) ctx.reader(),
                        NEW_FIELD,
                        DIM,
                        SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction(),
                        faissAttributes(),
                        docId -> newVec(ids[docId])
                    );

                    org.apache.lucene.codecs.KnnVectorsReader mergeInstance = wrapped.getVectorReader().getMergeInstance();
                    FloatVectorValues values = mergeInstance.getFloatVectorValues(NEW_FIELD);
                    assertNotNull("the FAISS merge path must see the added field", values);
                    assertEquals("every document has a value", ctx.reader().maxDoc(), values.size());

                    int seen = 0;
                    KnnVectorValues.DocIndexIterator it = values.iterator();
                    for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                        assertArrayEquals(newVec(ids[doc]), values.vectorValue(it.index()), 0.0f);
                        assertFalse(
                            "must not serve the old field's values",
                            Arrays.equals(oldVec(ids[doc]), values.vectorValue(it.index()))
                        );
                        seen++;
                    }
                    assertEquals(ctx.reader().maxDoc(), seen);

                    // The delegate's FAISS field is still reachable through the same merge instance.
                    assertNotNull("the pre-existing faiss field is still served", mergeInstance.getFloatVectorValues(OLD_FIELD));
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

    private Document doc(int i) {
        Document d = new Document();
        d.add(new StringField(ID, "doc-" + i, Field.Store.YES));
        d.add(new TextField("body", "document " + i + " storage segment merge quantization recall latency", Field.Store.YES));
        d.add(new StoredField("payload", "payload-for-doc-" + i + "-" + "y".repeat(48)));
        d.add(new NumericDocValuesField("rank", i));
        d.add(new KnnFloatVectorField(OLD_FIELD, oldVec(i), faissOnDiskFieldType()));
        // NEW_FIELD deliberately absent — it is added later by the codec reader.
        return d;
    }

    /**
     * The attribute map a faiss/on_disk field carries. Mirrors what {@code KNNVectorFieldMapper} puts on
     * the {@link FieldType}, which is where the native writer reads its configuration from.
     */
    private static Map<String, String> faissAttributes() {
        Map<String, String> a = new HashMap<>();
        a.put(KNNVectorFieldMapper.KNN_FIELD, "true");
        a.put(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        a.put(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        a.put(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        a.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());
        a.put(KNNConstants.PARAMETERS, "{ \"index_description\":\"BHNSW32\", \"spaceType\": \"l2\"}");
        QuantizationConfig qc = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.ONE_BIT).build();
        a.put(KNNConstants.QFRAMEWORK_CONFIG, QuantizationConfigParser.toCsv(qc));
        return a;
    }

    /** A faiss/on_disk float vector field: native FAISS + BHNSW32 + 1-bit scalar quantization. */
    private FieldType faissOnDiskFieldType() {
        FieldType ft = new FieldType();
        ft.setTokenized(false);
        ft.setIndexOptions(IndexOptions.NONE);
        for (Map.Entry<String, String> e : faissAttributes().entrySet()) {
            ft.putAttribute(e.getKey(), e.getValue());
        }
        ft.setVectorAttributes(DIM, VectorEncoding.FLOAT32, SpaceType.L2.getKnnVectorSimilarityFunction().getVectorSimilarityFunction());
        ft.freeze();
        return ft;
    }

    private static int[] docIdToLogicalId(LeafReader leaf) throws IOException {
        StoredFields sf = leaf.storedFields();
        int[] ids = new int[leaf.maxDoc()];
        for (int doc = 0; doc < leaf.maxDoc(); doc++) {
            ids[doc] = Integer.parseInt(sf.document(doc).get(ID).substring("doc-".length()));
        }
        return ids;
    }

    private static List<String> filesWithExt(Directory dir, String ext) throws Exception {
        return Arrays.stream(dir.listAll()).filter(f -> f.contains(ext)).collect(Collectors.toList());
    }

    private static float[] vectorFor(DirectoryReader reader, String field, int logicalId) throws IOException {
        for (LeafReaderContext ctx : reader.leaves()) {
            LeafReader leaf = ctx.reader();
            FloatVectorValues values = leaf.getFloatVectorValues(field);
            if (values == null) {
                continue;
            }
            StoredFields stored = leaf.storedFields();
            KnnVectorValues.DocIndexIterator it = values.iterator();
            for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                if (("doc-" + logicalId).equals(stored.document(doc).get(ID))) {
                    return values.vectorValue(it.index());
                }
            }
        }
        throw new AssertionError("no vector for field [" + field + "] doc-" + logicalId);
    }
}
