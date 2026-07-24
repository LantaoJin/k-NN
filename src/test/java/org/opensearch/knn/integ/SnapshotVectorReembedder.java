/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.RestClient;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * PoC orchestrator that "rewrites" the single {@code knn_vector} field of an OpenSearch index that
 * lives inside a snapshot, for an embedding-model upgrade where <b>every</b> vector is replaced.
 *
 * <h2>Why this shape (faiss + on_disk)</h2>
 * For {@code engine: faiss}, {@code mode: on_disk} the vector field is stored as a native Faiss
 * binary-quantized HNSW index ({@code .faiss}) plus Lucene quantized flat files ({@code .vec}/{@code
 * .veb}). These are produced by k-NN's Lucene codec ({@code Faiss1040ScalarQuantizedKnnVectorsWriter}
 * + the native build strategy) using <i>field-global</i> quantization parameters. You cannot
 * byte-patch them; the only correct way to change the vectors is to let k-NN's codec re-build the
 * field. So this tool does <b>not</b> edit snapshot blobs. It orchestrates, entirely over the REST
 * API of a running node:
 *
 * <ol>
 *   <li><b>restore</b> the source snapshot to a working index;</li>
 *   <li><b>re-embed</b>: read every document's {@code _source} (derived-source keeps the original
 *       float vector in {@code _source}), apply a transform to the vector field, and bulk-index into
 *       a fresh index that has the <i>same</i> faiss/on_disk mapping — so the destination's k-NN
 *       codec regenerates a faithful on_disk Faiss index for the new vectors;</li>
 *   <li><b>snapshot</b> the rebuilt index into a (new) snapshot, ready to restore as the upgraded
 *       index.</li>
 * </ol>
 *
 * The graph/quantization are rebuilt by k-NN during step 2's indexing, exactly as a normal ingest
 * would, so no HNSW work is done here.
 *
 * <p>This is a test-scoped utility (it drives the same {@link RestClient} the ITs use). A production
 * version would be a small client app pointed at a real cluster + repository.
 */
public final class SnapshotVectorReembedder {

    /** Transform applied to each document's vector (full replacement; PoC may ignore the old value). */
    @FunctionalInterface
    public interface VectorTransform {
        /**
         * @param docId the document _id
         * @param oldVector the existing vector as a list of Numbers (from _source), or null if absent
         * @return the new vector as a list of Float
         */
        List<Float> apply(String docId, List<Number> oldVector);
    }

    private final RestClient client;

    public SnapshotVectorReembedder(RestClient client) {
        this.client = client;
    }

    /**
     * Restore {@code srcSnapshot} from {@code repository}, replace all values of {@code vectorField}
     * via {@code transform}, write the result to {@code destIndex} (created with {@code destMapping},
     * which must carry the desired faiss/on_disk knn_vector mapping + {@code index.knn:true}
     * settings), and snapshot {@code destIndex} as {@code destSnapshot} in {@code repository}.
     *
     * @return number of documents re-embedded
     */
    public int reembed(
        String repository,
        String srcSnapshot,
        String srcIndex,
        String workIndex,
        String destIndex,
        String destSnapshot,
        String vectorField,
        String destSettingsAndMappingJson,
        VectorTransform transform
    ) throws IOException {
        // 1) Restore the snapshot to a working index (rename srcIndex -> workIndex).
        restore(repository, srcSnapshot, srcIndex, workIndex);
        refresh(workIndex);

        // 2) Create the destination index with the faiss/on_disk mapping, then re-embed via bulk.
        createIndex(destIndex, destSettingsAndMappingJson);
        int count = reembedAllDocs(workIndex, destIndex, vectorField, transform);
        refresh(destIndex);

        // 3) Snapshot the rebuilt index.
        snapshot(repository, destSnapshot, destIndex);
        return count;
    }

    // ---- step 1: restore ----

    private void restore(String repository, String snapshot, String srcIndex, String workIndex) throws IOException {
        XContentBuilder cmd = JsonXContent.contentBuilder().startObject();
        cmd.field("indices", srcIndex);
        cmd.field("rename_pattern", "(.+)");
        cmd.field("rename_replacement", workIndex);
        cmd.endObject();
        Request req = new Request("POST", "/_snapshot/" + repository + "/" + snapshot + "/_restore");
        req.addParameter("wait_for_completion", "true");
        req.setJsonEntity(cmd.toString());
        check(client.performRequest(req), 200, "restore");
    }

    // ---- step 2: re-embed ----

    private void createIndex(String index, String settingsAndMappingJson) throws IOException {
        Request req = new Request("PUT", "/" + index);
        req.setJsonEntity(settingsAndMappingJson);
        check(client.performRequest(req), 200, "create dest index");
    }

    @SuppressWarnings("unchecked")
    private int reembedAllDocs(String workIndex, String destIndex, String vectorField, VectorTransform transform) throws IOException {
        // Scroll through every document's _source. PoC volumes are small, so a single large search is fine.
        Request search = new Request("GET", "/" + workIndex + "/_search");
        search.addParameter("size", "10000");
        search.setJsonEntity("{\"query\":{\"match_all\":{}}}");
        Map<String, Object> resp = asMap(client.performRequest(search));
        Map<String, Object> hitsOuter = (Map<String, Object>) resp.get("hits");
        List<Map<String, Object>> hits = (List<Map<String, Object>>) hitsOuter.get("hits");

        StringBuilder bulk = new StringBuilder();
        int count = 0;
        for (Map<String, Object> hit : hits) {
            String id = (String) hit.get("_id");
            Map<String, Object> source = (Map<String, Object>) hit.get("_source");

            List<Number> oldVector = null;
            Object existing = source.get(vectorField);
            if (existing instanceof List) {
                oldVector = (List<Number>) existing;
            }
            List<Float> newVector = transform.apply(id, oldVector);
            source.put(vectorField, newVector);

            // bulk action + source line
            XContentBuilder action = JsonXContent.contentBuilder().startObject().startObject("index");
            action.field("_index", destIndex);
            action.field("_id", id);
            action.endObject().endObject();
            bulk.append(action.toString()).append("\n");
            bulk.append(toJson(source)).append("\n");
            count++;
        }

        if (count > 0) {
            Request bulkReq = new Request("POST", "/_bulk");
            bulkReq.addParameter("refresh", "true");
            bulkReq.setJsonEntity(bulk.toString());
            Response bulkResp = client.performRequest(bulkReq);
            check(bulkResp, 200, "bulk re-embed");
            Map<String, Object> bulkMap = asMap(bulkResp);
            if (Boolean.TRUE.equals(bulkMap.get("errors"))) {
                throw new IOException("bulk re-embed reported errors: " + bodyOf(bulkResp));
            }
        }
        return count;
    }

    // ---- step 3: snapshot ----

    private void snapshot(String repository, String snapshot, String index) throws IOException {
        XContentBuilder cmd = JsonXContent.contentBuilder().startObject();
        cmd.field("indices", index);
        cmd.field("include_global_state", false);
        cmd.endObject();
        Request req = new Request("PUT", "/_snapshot/" + repository + "/" + snapshot);
        req.addParameter("wait_for_completion", "true");
        req.setJsonEntity(cmd.toString());
        check(client.performRequest(req), 200, "snapshot dest");
    }

    // ---- helpers ----

    private void refresh(String index) throws IOException {
        client.performRequest(new Request("POST", "/" + index + "/_refresh"));
    }

    private static void check(Response resp, int expected, String what) throws IOException {
        int code = resp.getStatusLine().getStatusCode();
        if (code != expected) {
            throw new IOException(what + " failed: HTTP " + code + " -> " + bodyOf(resp));
        }
    }

    /** Reads a response body, converting Apache's checked {@link ParseException} into {@link IOException}. */
    private static String bodyOf(Response resp) throws IOException {
        try {
            return EntityUtils.toString(resp.getEntity());
        } catch (ParseException e) {
            throw new IOException("failed to parse response entity", e);
        }
    }

    private static Map<String, Object> asMap(Response resp) throws IOException {
        String body = bodyOf(resp);
        try (
            XContentParser parser = MediaTypeRegistry.JSON.xContent()
                .createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, body)
        ) {
            return parser.map();
        }
    }

    private static String toJson(Map<String, Object> map) throws IOException {
        XContentBuilder b = JsonXContent.contentBuilder();
        b.map(map);
        return b.toString();
    }
}
