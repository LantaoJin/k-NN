# Approach B — vector-only snapshot-blob rewrite

Replace **all vectors** of one `knn_vector` field (faiss/on_disk) inside an OpenSearch snapshot,
producing a new snapshot, **without re-indexing any other field**. Unlike
[approach A](../README.md) (which `_bulk`-reindexes whole documents), this regenerates *only* the
vector segment files and copies every other file byte-for-byte, then splices the result into the
snapshot's blobs. A normal `_restore` then rebuilds the shard from those blobs.

This is the approach you want when the index has expensive/large non-vector fields (text, doc-values,
points, …) that must not be rebuilt.

## Status: proven end-to-end

On a real OpenSearch 3.7.0 node (k-NN + FAISS), faiss/on_disk index:

- restore of the rewritten snapshot is **GREEN** (no translog issue — see below);
- restored index has the **new** vectors; **4/4 KNN self-match** (each doc's own new vector returns
  itself, score 1.0);
- querying an **old** vector scores ~`1e-5` (no match) → all old vectors are gone;
- the **13 non-vector segment files are byte-identical** to the source;
- the **source repo is untouched** (still restores the old vectors).

## The two tools

Both are standalone Java programs compiled against the jars bundled in the OpenSearch container
(`/usr/share/opensearch/lib/*` + `plugins/opensearch-knn/*` + `plugins/opensearch-knn/lib/*`) — no
gradle, no running node required for the rewrite itself.

| File | Role |
|------|------|
| `VectorOnlySegmentRewriter.java` | Reads a shard's Lucene segment dir; regenerates **only** the vector field's files via the k-NN codec (native faiss/on_disk `.faiss` + quantized `.vec/.vemf/.vemq/.veq`); **byte-copies all other files**; rewrites `.si`. Asserts the non-vector files are byte-identical. |
| `SnapshotBlobVectorRewriter.java` | Edits the fs-repository **blobs** standalone (`FsBlobStore` + `ChecksumBlobStoreFormat`, no node): reuses non-vector `FileInfo` verbatim, writes new `__<uuid>` data blobs for the rewritten vector files with **recomputed checksums**, and rewrites the shard `snap-<uuid>.dat` + `index-<gen>` manifests. |

## Pipeline (what actually runs)

```
create faiss/on_disk index (derived_source on) + ingest
  -> snapshot  ->  srcrepo/snap1                       (the INPUT snapshot)
  -> restore snap1 -> 'work'                           (materialize the shard's Lucene files)
  -> copy work shard dir -> /tmp/srcSeg
  -> VectorOnlySegmentRewriter  /tmp/srcSeg -> /tmp/destSeg  (regenerate vectors only, +SHIFT)
  -> cp -r srcrepo destrepo
  -> SnapshotBlobVectorRewriter  destrepo <indexId> 0 <snapUuid> /tmp/destSeg <vecExts>
  -> register destrepo -> _restore snap1 -> 'upgraded'  (GREEN)
  -> verify: new vectors, KNN self-match, source untouched
```

Why the `restore -> 'work'` step: the snapshot stores Lucene files as opaque `__<uuid>` blobs keyed
by physical name in the manifest. Restoring once is the simplest way to materialize them as a normal
Lucene directory that `VectorOnlySegmentRewriter` can read. (A fully offline variant could reassemble
the files straight from the blobs using the manifest; not needed for the demo.)

## Run it manually (against a container)

Start a node with `path.repo=/tmp` (so both `srcrepo` and `destrepo` under `/tmp` are allowed):

```bash
docker run -d --name os-blob -p 19202:9200 \
  -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
  -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms1g -Xmx1g" \
  -e path.repo=/tmp  opensearchproject/opensearch:3.7.0
```

Stage the jars the tools need (from a local gradle cache) and compile inside the container:

```bash
# guava (k-NN treats it as compileOnly; provided at runtime in a real node)
docker cp guava-33.x-jre.jar  os-blob:/tmp/guava.jar
# mockito + byte-buddy + objenesis (only for the standalone KNNSettings ClusterService stub)
docker exec os-blob mkdir -p /tmp/mlib
docker cp mockito-core-*.jar os-blob:/tmp/mlib/ ; docker cp byte-buddy-*.jar os-blob:/tmp/mlib/
docker cp byte-buddy-agent-*.jar os-blob:/tmp/mlib/ ; docker cp objenesis-*.jar os-blob:/tmp/mlib/
docker cp VectorOnlySegmentRewriter.java   os-blob:/tmp/
docker cp SnapshotBlobVectorRewriter.java  os-blob:/tmp/

CP='/tmp:/tmp/guava.jar:/tmp/mlib/*:/usr/share/opensearch/lib/*:/usr/share/opensearch/plugins/opensearch-knn/*:/usr/share/opensearch/plugins/opensearch-knn/lib/*'
docker exec os-blob /usr/share/opensearch/jdk/bin/javac -cp "$CP" -d /tmp \
  /tmp/VectorOnlySegmentRewriter.java /tmp/SnapshotBlobVectorRewriter.java
```

Run each tool (note `--add-modules jdk.incubator.vector` and the native lib path):

```bash
LP=-Djava.library.path=/usr/share/opensearch/plugins/opensearch-knn/lib

# 1) regenerate only the vector files (replace-all, transform new[d] = (docId+1)+d+SHIFT)
docker exec os-blob /usr/share/opensearch/jdk/bin/java --add-modules jdk.incubator.vector \
  -cp "$CP" $LP VectorOnlySegmentRewriter /tmp/srcSeg /tmp/destSeg vec 100

# 2) splice into the copied repo's snapshot blobs
docker exec os-blob /usr/share/opensearch/jdk/bin/java --add-modules jdk.incubator.vector \
  -cp "$CP" $LP SnapshotBlobVectorRewriter \
  /tmp/destrepo <indexId> 0 <snapUuid> /tmp/destSeg .faiss,.vec,.vemf,.vemq,.veq,.vem,.vex
```

`<indexId>` = `ls /tmp/srcrepo/indices`; `<snapUuid>` = the `snap-<uuid>.dat` name at the repo root.

## Key learnings (non-obvious, hard-won)

1. **No translog problem in this flow.** A snapshot contains **no translog** — `_restore` builds the
   shard + translog fresh from the blobs. (An earlier attempt to splice files into an *already
   restored, live* shard failed on a `translog_uuid` mismatch; that was an artifact of editing a live
   shard, not of the rewrite. Rewriting at the blob level avoids it entirely.)

2. **Forcing the native faiss format standalone.** k-NN's `PerFieldKnnVectorsFormat` picks
   faiss-vs-Lucene from the **`MapperService`**, which doesn't exist outside a node — so the default
   selector falls back to plain Lucene HNSW (full precision, *not* on_disk). `VectorOnlySegmentRewriter`
   bypasses it by constructing `Faiss1040ScalarQuantizedKnnVectorsFormat()` directly; the faiss SQ
   writer derives quantization from the `FieldInfo` attributes (`sq_config`, `parameters`) the source
   segment already carries.

3. **KNNSettings needs a ClusterService.** The native write path calls `KNNSettings` (e.g. index
   thread qty) via a `ClusterService` singleton a node normally wires up. Standalone we set one with a
   `ClusterSettings` registered with k-NN's node-scoped settings (the demo uses a Mockito mock for
   `getClusterSettings()`), else `NativeIndexWriter.getParameters()` NPEs.

4. **Checksums.** Each rewritten file's `StoreFileMetadata` checksum is recomputed as
   `Long.toString(CodecUtil.retrieveChecksum(indexInput), Character.MAX_RADIX)` (mirrors
   `Store.digestToString`). Non-vector files reuse their existing `FileInfo` (blob + checksum) verbatim.

5. **Vectors are assigned by ordinal, not docId.** After a force-merge the ordinal order ≠ id order,
   so verify with **each doc's own stored vector as the query** (expect self-match, score 1.0) rather
   than an arithmetic guess about which doc a given vector belongs to.

## Scope / caveats (PoC)

- One `knn_vector` field; **all** vectors replaced (model-version upgrade). Replace-all means the tool
  never needs the old vector values — it enumerates docIDs and synthesizes new vectors.
- Relies on **derived source** keeping the vector out of stored fields (`.fdt`) so regenerating only
  the vector files updates `_source` via read-time injection. Verified: a faiss/on_disk segment with
  `index.knn.derived_source.enabled=true` (or the default) stamps the `derived_vector_fields=vec`
  segment attribute, confirming the vector is masked from `.fdt`. **Do not** enable *core*
  `index.derived_source.enabled` — on 3.7 that disables k-NN masking and leaves the vector in `.fdt`,
  which would make a vector-only rewrite leave `_source` stale.
- Single shard, single segment in the demo. Multi-segment / multi-shard generalizes (loop over
  segments / shards) but isn't wired in the PoC.
- The demo restores the **same snapshot name** from the copied repo (simplest provable round-trip);
  a production tool would register the rewritten content as a distinct snapshot id.
- `DISABLE_SECURITY_PLUGIN=true` and Mockito-on-classpath are demo conveniences, not for production.

## Adapting to a real upgrade

- Replace the transform in `VectorOnlySegmentRewriter.newVector(...)` with your model-v2 embedding
  (it currently synthesizes `(docId+1)+d+shift`; a real tool would map docId → new embedding).
- Point the tools at your real fs-repository path and the target index's `indexId` / shard / snapshot
  uuid.
- For multi-segment shards, run the segment rewriter per segment and pass the full rewritten file set
  to the blob rewriter.
