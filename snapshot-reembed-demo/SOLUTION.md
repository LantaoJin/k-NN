# Re-embedding based on an OpenSearch snapshot: how to skip unnecessary re-indexing

> **TL;DR** — To upgrade the embedding model of a `knn_vector` field in an OpenSearch index, you do
> **not** have to rebuild the whole index. If the field uses `engine: faiss`, `mode: on_disk` with
> **derived source** enabled, you can take a snapshot, regenerate **only the vector field's files**
> at the Lucene-segment level, splice them back into the snapshot's blobs, and restore the result.
> Every non-vector field (text, doc-values, points, norms, stored) is carried over **byte-for-byte**
> — never re-analyzed, never rebuilt. A proof-of-concept does this end-to-end against a real
> OpenSearch 3.7.0 + k-NN/FAISS node; see [`vector-only/`](vector-only/).

---

## 1. The problem

A common operational need: **upgrade the embedding model.** You have an index with one
`knn_vector` field plus many other fields (say 1 vector field + 49 immutable text/keyword/numeric
fields). A new model version produces better embeddings, so **every vector must be replaced** — but
the other 49 fields don't change at all.

The textbook answer is *"reindex into a new index."* That works, but it re-analyzes and rewrites
**all 50 fields for every document**, even though only the vector changed. For large indices with
expensive analyzed text, big doc-values, or points fields, the 49 untouched fields dominate the cost
— and you must still **possess every field's source value** to reindex it, which is often
impractical (the source of truth may live elsewhere, or `_source` may be partially derived).

This document describes a solution that pays **only** for the vector work: regenerate the vector
field, copy everything else verbatim, all driven from a **snapshot** (so it can run offline against
a backup, not against a live, hot index).

---

## 2. Background: how a faiss/on_disk vector field is stored

To understand why selective rewriting is possible, you need the on-disk layout. For
`engine: faiss`, `mode: on_disk` (the default high-compression mode), a single vector field in a
segment is **three separate artifacts**:

| Artifact | File(s) | Produced by | Contains |
|---|---|---|---|
| Native HNSW graph | `_<seg>_165_<field>.faiss` | k-NN FAISS (JNI) | the graph only (`IO_FLAG_SKIP_STORAGE`) |
| Quantized flat vectors | `_<seg>_..._<field>.vec/.vemf/.vemq/.veq` | Lucene quantized codec | 1-bit binary-quantized codes + correction factors |
| (logical) original floats | re-injected into `_source` | **derived source** | the full-precision vectors, NOT stored in `.fdt` |

Every other field lives in **its own, independent files**, written by non-k-NN codecs:

- stored fields → `.fdt` / `.fdx` / `.fdm`
- postings (analyzed text) → `.doc` / `.tim` / `.tip` / `.tmd` / `.psm`
- doc-values → `.dvd` / `.dvm`
- points → `.kdd` / `.kdi` / `.kdm`
- norms → `.nvd` / `.nvm`
- field metadata → `.fnm`; segment manifest → `.si`

**The crucial property: the vector field's files contain only vector data, and no other field's
files contain any vector data.** This separation is what makes a vector-only rewrite possible at all.

### Derived source — the linchpin

With **derived source** enabled (`index.knn.derived_source.enabled: true`), k-NN **masks the vector
out of the stored `_source`** at index time (it writes a placeholder byte in `.fdt`) and
**re-injects the vector into `_source` at read time** from the vector files (`KnnVectorsReader`).

This is what makes the rewrite *complete*: if you regenerate the vector files with new values, a
`GET _doc` / `_search` automatically reflects the new vectors — **without touching `.fdt` at all**.
You can verify a segment is in this state by reading its `.si` attribute `derived_vector_fields`
(it lists the masked fields; present ⇒ vector is not in `.fdt`).

> **Anti-pattern:** if you instead enable **core** derived source (`index.derived_source.enabled`,
> a separate OpenSearch-core feature), it *takes precedence on 3.7+ and disables* k-NN masking — the
> vector then lives in `.fdt`, and a vector-only rewrite would leave `_source` stale. Use the k-NN
> setting (or the default), not the core one.

### A snapshot is not a live shard

An OpenSearch fs-repository snapshot stores:

- the Lucene segment files, wrapped as opaque `__<uuid>` **data blobs**;
- **metadata blobs**: `BlobStoreIndexShardSnapshot` (`snap-<uuid>.dat`, the per-snapshot file list +
  checksums), `BlobStoreIndexShardSnapshots` (`index-<gen>`, all snapshots of the shard),
  `RepositoryData` (`index-N`), `index.latest`, `SnapshotInfo`, `IndexMetadata`.

A snapshot contains **no translog** and no live shard state. On `_restore`, OpenSearch reconstructs
the Lucene files from the blobs and builds a **fresh shard + translog from scratch**. This fact
removes an entire class of problems (see §5).

---

## 3. The solution: vector-only snapshot-blob rewrite

```
                 ┌─────────────────────────────────────────────────────────────┐
  source         │  snapshot (srcrepo/snap1)                                    │
  snapshot  ───▶ │   indices/<id>/0/  __<blobs…>  snap-<uuid>.dat  index-<gen>  │
                 └─────────────────────────────────────────────────────────────┘
                        │  (1) materialize the shard's Lucene files
                        ▼
                  segment dir  ──(2) VectorOnlySegmentRewriter──▶  rewritten segment
                  (.faiss/.vec* + 13 other files)     regenerate ONLY .faiss/.vec*  (copy others verbatim)
                        │
                        │  (3) SnapshotBlobVectorRewriter
                        ▼
                 ┌─────────────────────────────────────────────────────────────┐
  new      ◀──── │  destrepo (copy of srcrepo) with new vector data blobs +     │
  snapshot       │  rewritten snap-<uuid>.dat + index-<gen>                     │
                 └─────────────────────────────────────────────────────────────┘
                        │  (4) _restore  →  fresh shard + translog
                        ▼
                  upgraded index: NEW vectors, all other fields identical
```

Two small standalone tools (no running node needed for the rewrite itself; they compile against the
jars bundled in the OpenSearch image):

1. **`VectorOnlySegmentRewriter`** — opens the shard's Lucene segment directory and, for the one
   vector field, **regenerates only its files** through the k-NN codec (so FAISS re-quantizes and
   rebuilds the native index for the new vectors). It **byte-copies every other file** and rewrites
   `.si`. It asserts the non-vector files are byte-identical. Because all vectors are being
   replaced, it never reads the old values — it enumerates docIDs and synthesizes new vectors (in a
   real upgrade this is where the new model runs).

2. **`SnapshotBlobVectorRewriter`** — edits the fs-repository **blobs** standalone
   (`FsBlobStore` + `ChecksumBlobStoreFormat`): reuses each non-vector file's existing `FileInfo`
   verbatim (same blob, same checksum), writes new `__<uuid>` data blobs for the rewritten vector
   files with **recomputed length + checksum**, and rewrites the shard's `snap-<uuid>.dat` and
   `index-<gen>` manifests to point at the new file set.

Then a normal `_restore` of the rewritten snapshot yields an index with the new vectors.

A one-command demo runs all of this and verifies it: [`vector-only/run-demo.sh`](vector-only/run-demo.sh).

---

## 4. Why the PoC works

1. **Field files are physically separable.** Only the vector field's files (`.faiss`, `.vec*`)
   carry vector data; the other 49 fields are in independent files. Regenerating the former and
   copying the latter is therefore sound — proven by the rewriter's invariant check:
   *"13 non-vector files byte-identical."*

2. **Derived source makes `_source` follow the vector files.** The vector is masked out of `.fdt`
   and re-injected from the vector files at read time, so new vector files ⇒ new `_source` vectors,
   with zero stored-fields changes.

3. **The k-NN codec rebuilds the native FAISS index correctly.** You cannot byte-patch a quantized
   `.faiss` (its codes depend on field-global quantization parameters). Instead, the rewriter feeds
   new float vectors to the codec's writer, which **re-quantizes and rebuilds** the native index
   exactly as normal ingestion would.

4. **The snapshot blob format is writable standalone.** `FsBlobStore`, `ChecksumBlobStoreFormat`,
   `BlobStoreIndexShardSnapshot`/`FileInfo`/`StoreFileMetadata`, and `RepositoryData` are pure
   data-structure + blob-IO classes with no hard node dependency, so a standalone tool can produce a
   valid, restorable snapshot.

5. **`_restore` rebuilds the shard from scratch.** Because we operate at the snapshot level (not on a
   live shard), restore creates a fresh shard + translog from the rewritten blobs — there is nothing
   stale to reconcile.

### Verified result (real OpenSearch 3.7.0 + k-NN/FAISS, faiss/on_disk)

- Restore of the rewritten snapshot is **GREEN** (no errors).
- The restored index returns the **new** vectors; **4/4 KNN self-match** (each doc's own new vector
  returns itself with score 1.0).
- Querying an **old** vector scores ~`1e-5` → all old vectors are gone.
- The **non-vector segment files are byte-identical** to the source.
- The **source repository is untouched** (still restores the old vectors).

---

## 5. Key issues to address

These are the non-obvious problems that must be solved for the approach to be correct. Each was hit
and resolved while building the PoC.

### 5.1 You must regenerate the vector files, not patch them
`mode: on_disk` stores **binary-quantized** vectors whose codes are derived from field-global
quantization parameters. Editing bytes in place would corrupt them. **Resolution:** run the new
float vectors through the k-NN codec writer so it re-quantizes and rebuilds the `.faiss` + flat
files.

### 5.2 Forcing the native faiss format outside a node
k-NN's `PerFieldKnnVectorsFormat` decides faiss-vs-Lucene **from the `MapperService`**, which does
not exist outside a running node. With no mapper, the selector silently **falls back to plain
Lucene HNSW (full precision)** — *not* faiss/on_disk. **Resolution:** the rewriter constructs
`Faiss1040ScalarQuantizedKnnVectorsFormat` directly; the FAISS SQ writer derives quantization from
the `FieldInfo` attributes (`sq_config`, `parameters`) that the source segment already carries.

### 5.3 KNNSettings needs a ClusterService
The native write path reads `KNNSettings` (e.g. index thread count) via a `ClusterService`
singleton a node normally wires up; standalone it is null and throws. **Resolution:** initialize
`KNNSettings.state().setClusterService(...)` with a `ClusterSettings` registered with k-NN's
node-scoped settings (defaults are fine).

### 5.4 Checksums must be recomputed
Each file in `BlobStoreIndexShardSnapshot` carries a `StoreFileMetadata` with a Lucene CRC. The
rewritten vector files have new content. **Resolution:** recompute as
`Long.toString(CodecUtil.retrieveChecksum(in), Character.MAX_RADIX)` (mirrors `Store.digestToString`).
Non-vector files keep their existing `FileInfo`/checksum verbatim.

### 5.5 Derived source must actually be engaged
The whole approach relies on the vector being absent from `.fdt`. **Resolution:** require
`index.knn.derived_source.enabled` (or the default that engages it) and verify via the segment's
`derived_vector_fields` attribute. **Do not** enable core `index.derived_source.enabled`, which
disables k-NN masking on 3.7+.

### 5.6 The translog is a non-issue (when done at the blob level)
A snapshot has no translog; `_restore` builds one fresh. **Pitfall to avoid:** splicing the
rewritten files into an *already-restored, live* shard fails with a `translog_uuid` mismatch — that
is an artifact of editing a live shard, not of the rewrite. Operating on snapshot blobs avoids it
entirely.

### 5.7 Vectors are addressed by ordinal, not docId
After a force-merge, vector ordinal order ≠ document-id order. **Resolution for verification:** test
with **each document's own stored vector** as the KNN query (expect a self-match at score 1.0)
rather than guessing which doc a value belongs to.

### 5.8 Compound files
Small/merged segments pack files into a `.cfs`. **Resolution:** read members through the codec's
compound reader and write the destination as loose (non-compound) files; the byte-identical check
compares the logical members.

### 5.9 Scope assumptions (PoC)
Single `knn_vector` field, all vectors replaced; single shard / single segment in the demo
(multi-segment/-shard generalizes by looping). These are PoC limits, not fundamental ones.

---

## 6. Comparison: vector-only snapshot rewrite vs. building a new index

| Dimension | **Build a new index** (full reindex) | **Vector-only snapshot rewrite** |
|---|---|---|
| What gets rebuilt | **All 50 fields** for every doc (re-analysis, postings, doc-values, points, norms, stored) | **Only** the vector field's `.faiss`/`.vec*`; other files copied byte-for-byte |
| Vector field work | rebuilt by the codec (same as the rewrite) | rebuilt by the codec (same) |
| Non-vector field work | re-analyzed + rewritten (wasted) | **none** |
| Do you need the other fields' source values? | **Yes** — must re-supply all 50 fields per doc | **No** — only `(docId, newVector)`; other fields ride along on disk |
| Input | a live index (or its data) you can read all fields from | a **snapshot** (a backup); source index/cluster need not be hot |
| Runs offline? | needs ingestion into a live cluster | the rewrite is standalone; only `_restore`/verify needs a node |
| Cost driver | `O(docs × all-field indexing)` | `O(docs × vector quantize+graph)` + a cheap byte-copy of other files |
| Atomicity / rollback | new index, atomic alias swap (clean) | new snapshot; restore to a new index, swap (clean) |
| Correctness risk | low (ordinary indexing path) | moderate — depends on derived source + segment-format details (§5) |
| Tooling maturity | first-class (`_reindex`, ingest) | custom segment + blob tooling (PoC here) |
| Best fit | small docs, or you already hold all field sources, or you want the simplest path | large/expensive non-vector fields you must not rebuild; source-of-truth for other fields not at hand |

### How to read this comparison

- Both approaches do the **same** amount of vector work (the codec rebuilds the FAISS index either
  way). The difference is entirely about the **non-vector fields**.
- "Build a new index" is the right default when documents are cheap to reindex, when you already
  have every field's source, or when you simply want the most standard, lowest-risk path.
- "Vector-only snapshot rewrite" wins precisely in the motivating case — **1 vector field + many
  expensive, immutable non-vector fields** — where reindexing the rest is pure waste and you'd
  rather not (or can't) re-supply their source values. It also fits a backup-centric workflow: you
  upgrade a snapshot and restore it, rather than mutating a hot index.

> **Note on `_bulk`/`_reindex` granularity.** OpenSearch has no per-field reindex: `_bulk`,
> `_reindex`, and `_update_by_query` all re-index the **whole document** (Lucene's unit of indexing
> is the document). So *any* cluster-REST approach to changing one field rebuilds all of them. The
> only way to touch a single field's data is **below** the cluster, at the Lucene-segment /
> snapshot-blob level — which is exactly what this solution does. (This repo also includes a
> baseline "full reindex via `_bulk`" demo in [`run-demo.sh`](run-demo.sh) for contrast.)

---

## 7. Production hardening (beyond the PoC)

- **Multi-segment / multi-shard:** loop the segment rewriter over every segment, and the blob
  rewriter over every shard; aggregate the rewritten file set per shard.
- **Distinct snapshot id:** the PoC reuses the same snapshot name in a copied repo for the simplest
  provable round-trip; a real tool would register the rewritten content as a new `SnapshotId` and
  bump `RepositoryData`/`index.latest`.
- **The embedding step:** replace the synthetic transform with the real model — map each `docId`
  (or its `_source`/text) to its new vector.
- **Streaming at scale:** materialize/iterate per segment rather than holding everything in memory.
- **Validation gate:** run `CheckIndex` and a KNN self-match sample before publishing the new
  snapshot.
- **Security & deps:** the PoC disables the security plugin and puts Mockito on the classpath for
  the standalone `ClusterService` stub — both are demo conveniences. A production tool would use a
  real minimal `ClusterService` and keep security enabled for any cluster interaction.

---

## 8. Files in this directory

| Path | What it is |
|---|---|
| `SOLUTION.md` | This document |
| `README.md` | Overview of both approaches (A: full reindex, B: vector-only) |
| `vector-only/` | **Approach B** — the solution described here (tools + `run-demo.sh` + `README.md`) |
| `run-demo.sh`, `reembed_tool.py`, `dest-mapping.json` | **Approach A** — baseline full-reindex demo |

To see it run end-to-end: `./vector-only/run-demo.sh` (requires Docker; pulls
`opensearchproject/opensearch:3.7.0`).
