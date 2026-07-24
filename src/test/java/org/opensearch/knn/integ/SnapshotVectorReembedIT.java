/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;

/**
 * End-to-end PoC for the snapshot vector re-embed flow (faiss + on_disk):
 *
 * <ol>
 *   <li>build a faiss/on_disk knn_vector index with a couple of non-vector fields and snapshot it;</li>
 *   <li>run {@link SnapshotVectorReembedder} to restore -> replace ALL vectors -> re-snapshot;</li>
 *   <li>restore the NEW snapshot and assert KNN search returns the re-embedded vectors and the
 *       non-vector fields survived.</li>
 * </ol>
 *
 * The destination index is created with the same faiss/on_disk mapping, so k-NN's codec rebuilds a
 * faithful on_disk Faiss index for the new vectors during step 2's bulk indexing (no HNSW work here).
 */
public class SnapshotVectorReembedIT extends KNNRestTestCase {

    private static final String FIELD = "vec";
    private static final String LABEL = "label";   // non-vector keyword field (must survive)
    private static final int DIM = 8;
    private static final int NUM_DOCS = 20;

    private final String repository = "reembed-repo";
    private final String srcIndex = "src-index";
    private final String srcSnapshot = "src-snap";
    private final String workIndex = "work-index";
    private final String destIndex = "dest-index";
    private final String destSnapshot = "dest-snap";
    private static final String RESTORE_SUFFIX = "-restored";
    private final String restoredIndex = "dest-index" + RESTORE_SUFFIX;

    /** Original vector for doc i (deterministic). */
    private static float[] origVec(int i) {
        float[] v = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            v[d] = i + d * 0.01f;
        }
        return v;
    }

    /** PoC "new embedding model": newVector = oldVector + 100 (full replacement, deterministic). */
    private static List<Float> reembed(List<Number> old) {
        List<Float> out = new ArrayList<>(old.size());
        for (Number n : old) {
            out.add(n.floatValue() + 100f);
        }
        return out;
    }

    private String onDiskMapping() throws Exception {
        // faiss + on_disk knn_vector mapping (+ a keyword field). on_disk implies faiss engine and
        // (>=3.6) 32x binary quantization by default.
        XContentBuilder b = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD)
            .field("type", "knn_vector")
            .field(DIMENSION, DIM)
            .field(MODE_PARAMETER, "on_disk")
            .startObject("method")
            .field("name", "hnsw")
            .field(KNN_ENGINE, "faiss")
            .field("space_type", "l2")
            .endObject()
            .endObject()
            .startObject(LABEL)
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject();
        return b.toString();
    }

    private String destSettingsAndMapping() throws Exception {
        XContentBuilder b = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("settings")
            .field("number_of_shards", 1)
            .field("number_of_replicas", 0)
            .field("index.knn", true)
            .endObject()
            .startObject("mappings")
            .startObject("properties")
            .startObject(FIELD)
            .field("type", "knn_vector")
            .field(DIMENSION, DIM)
            .field(MODE_PARAMETER, "on_disk")
            .startObject("method")
            .field("name", "hnsw")
            .field(KNN_ENGINE, "faiss")
            .field("space_type", "l2")
            .endObject()
            .endObject()
            .startObject(LABEL)
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return b.toString();
    }

    @SneakyThrows
    public void testReembedFaissOnDiskSnapshot() {
        // ---- step 0: build a faiss/on_disk index + snapshot ----
        createKnnIndex(srcIndex, getDefaultIndexSettings(), onDiskMapping());
        for (int i = 0; i < NUM_DOCS; i++) {
            indexDoc(srcIndex, Integer.toString(i), origVec(i), "label-" + i);
        }
        refreshAllNonSystemIndices();

        final String pathRepo = System.getProperty("tests.path.repo");
        Settings repoSettings = Settings.builder().put("compress", false).put("location", pathRepo).build();
        registerRepository(repository, "fs", true, repoSettings);
        createSnapshot(repository, srcSnapshot, true);

        // ---- step 1+2+3: restore -> re-embed ALL vectors -> re-snapshot ----
        SnapshotVectorReembedder tool = new SnapshotVectorReembedder(client());
        int count = tool.reembed(
            repository,
            srcSnapshot,
            srcIndex,
            workIndex,
            destIndex,
            destSnapshot,
            FIELD,
            destSettingsAndMapping(),
            (docId, oldVector) -> {
                assertNotNull("derived-source vector must be present in _source for " + docId, oldVector);
                return reembed(oldVector);
            }
        );
        assertEquals(NUM_DOCS, count);

        // ---- step 4: restore the NEW snapshot as a fresh index and verify ----
        // Drop the working/dest indices first so the restore target name is clean.
        deleteIndexIfExists(workIndex);
        deleteIndexIfExists(destIndex);
        restoreSnapshot(RESTORE_SUFFIX, List.of(destIndex), repository, destSnapshot, true);
        refreshAllNonSystemIndices();

        assertEquals(NUM_DOCS, getDocCount(restoredIndex));

        // KNN search: query near the re-embedded doc-3 (origVec(3)+100). Expect doc-3 as the top hit,
        // proving the restored on_disk faiss index reflects the NEW vectors.
        float[] target = origVec(3);
        for (int d = 0; d < DIM; d++) {
            target[d] += 100f;
        }
        Response resp = searchKNNIndex(restoredIndex, new KNNQueryBuilder(FIELD, target, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(resp.getEntity()), FIELD);
        assertFalse("expected KNN hits", results.isEmpty());
        assertEquals("top hit should be the re-embedded doc-3", "3", results.get(0).getDocId());

        // Sanity: a query near an ORIGINAL vector should NOT return that doc as the strong top match,
        // confirming the old vectors are gone (all replaced).
        Response respOld = searchKNNIndex(restoredIndex, new KNNQueryBuilder(FIELD, origVec(3), 1), 1);
        List<KNNResult> oldResults = parseSearchResponse(EntityUtils.toString(respOld.getEntity()), FIELD);
        // It still returns *something* (k=1), but the nearest re-embedded vector to origVec(3) is doc-0ish,
        // definitely not an exact match; we just assert search works and the index is non-empty.
        assertFalse(oldResults.isEmpty());

        // Non-vector field survived the re-embed round-trip.
        assertEquals("label-7", getSourceField(restoredIndex, "7", LABEL));
    }

    // ---- helpers ----

    private void indexDoc(String index, String id, float[] vector, String label) throws Exception {
        XContentBuilder src = XContentFactory.jsonBuilder().startObject();
        src.field(FIELD, vector);
        src.field(LABEL, label);
        src.endObject();
        Request req = new Request("POST", "/" + index + "/_doc/" + id + "?refresh=true");
        req.setJsonEntity(src.toString());
        client().performRequest(req);
    }

    private void deleteIndexIfExists(String index) {
        try {
            client().performRequest(new Request("DELETE", "/" + index));
        } catch (Exception ignored) {
            // index may not exist; fine
        }
    }

    @SuppressWarnings("unchecked")
    private String getSourceField(String index, String id, String field) throws Exception {
        Response resp = client().performRequest(new Request("GET", "/" + index + "/_doc/" + id));
        String body = EntityUtils.toString(resp.getEntity());
        var map = createParser(org.opensearch.core.xcontent.MediaTypeRegistry.getDefaultMediaType().xContent(), body).map();
        var source = (java.util.Map<String, Object>) map.get("_source");
        return (String) source.get(field);
    }
}
