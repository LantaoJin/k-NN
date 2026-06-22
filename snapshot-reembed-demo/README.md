# Demo: re-embed a faiss/on_disk knn_vector field inside an OpenSearch snapshot

Upgrade the embedding model of a `knn_vector` field by **replacing all its vectors** — taking an
OpenSearch **snapshot** as input and producing a **new snapshot** that restores as the upgraded
index. Everything runs against a real **OpenSearch 3.7.0** node (k-NN + FAISS bundled) in Docker, on
a genuine `engine: faiss`, `mode: on_disk` index.

> 📖 For the full write-up — why it works, the key issues, and a comparison against building a whole
> new index — see **[`SOLUTION.md`](SOLUTION.md)**. This README is the hands-on/run guide.

## Two approaches in this directory

There are **two** demonstrated ways to do this. They reach the same outcome but differ in *what work
they do* — which matters a lot when the index has many non-vector fields.

| | **A. Full re-index** (`run-demo.sh` + `reembed_tool.py`) | **B. Vector-only blob rewrite** (`vector-only/`) |
|---|---|---|
| Mechanism | restore → `_bulk` re-index every doc into a fresh index → snapshot | rewrite **only the vector files** inside the snapshot blobs; restore |
| Re-indexes non-vector fields? | **Yes** — re-analyzes/rewrites postings, doc-values, points, norms, stored for every doc | **No** — those segment files are copied byte-for-byte; only `.faiss`/`.vec*` are regenerated |
| Needs a running node? | Yes (drives the cluster REST API) | No for the rewrite (standalone Java on the repo dir); a node only to *verify* via restore |
| Best when | quick to run; fine if the doc is cheap to re-index | the index has expensive/large non-vector fields you must not touch (the motivating case) |
| Status | passes end-to-end | passes end-to-end |

If your goal is *"change the embeddings without paying to reindex the other 49 fields,"* **B is the
one you want.** A is the simpler baseline and a useful contrast. See
[`vector-only/README.md`](vector-only/README.md) for B.

The rest of this file documents **approach A**.

## Run it (approach A)

```bash
./run-demo.sh          # start container -> build v1 index -> snapshot -> re-embed -> verify
./run-demo.sh --keep   # same, but leave the container running for manual curl poking
```

Requires: `docker`, `curl`, `python3`, `jq`. First run pulls `opensearchproject/opensearch:3.7.0`.

## At a glance

```
STEP 0  start OpenSearch 3.7.0 (k-NN + FAISS) in Docker
STEP 1  create source index 'products' (knn_vector 'vec', faiss/on_disk) + ingest 8 docs
STEP 2  register an fs snapshot repo and snapshot 'products'  -> snap-model-v1
STEP 3  reembed_tool.py:  restore snap-model-v1 -> replace ALL vectors (new = old + 100)
                          into a fresh faiss/on_disk index -> snapshot  -> snap-model-v2
STEP 4  restore snap-model-v2 and verify the new vectors + KNN search
RESULT  PASS
```

Expected tail:

```
restored vec(doc-3) = [103.0,103.0,103.0,103.0,103.0,103.0,103.0,103.0]
top hit = doc-3
PASS: snapshot re-embed produced a restorable faiss/on_disk index with the NEW vectors.
```

## Step-by-step (what each step actually does)

The script defines its config up front (`run-demo.sh` lines 25–34): `DIM=8`, `NUM_DOCS=8`,
the transform `SHIFT=100`, and the object names — repo `demo-repo`, v1 index `products`, v1
snapshot `snap-model-v1`, scratch `products-work`, rebuilt index `products-v2`, its snapshot
`snap-model-v2`, final restored index `products-v2-restored`. A `trap cleanup EXIT` force-removes
the container however the script ends (skipped with `--keep`).

### STEP 0 — Start a real OpenSearch node in Docker

```
docker run -d --name knn-reembed-demo -p 19200:9200 \
  -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
  -e path.repo=/tmp/snapshots  opensearchproject/opensearch:3.7.0
```

- `-p 19200:9200` maps the container's REST port to host `19200` (avoids clashing with a local 9200).
- `discovery.type=single-node` skips cluster bootstrap checks.
- `DISABLE_SECURITY_PLUGIN=true` allows plain `http://`, no TLS/auth (**demo only**).
- `path.repo=/tmp/snapshots` is **mandatory** — OpenSearch refuses to register an `fs` repository
  unless its location sits under a declared `path.repo`. The script then `chown`s that dir to the
  `opensearch` user so the node can write snapshot blobs.
- **Why Docker matters:** the container's own network namespace sidesteps a host JMX port collision
  that blocks the gradle integration test in this environment.
- The script polls `curl localhost:19200` until the API answers (~20s), then prints the
  `opensearch-knn 3.7.0.0` plugin line — that plugin provides the `knn_vector` type and the FAISS lib.

### STEP 1 — Create the source faiss/on_disk index and ingest docs

`PUT /products` with `dest-mapping.json`:

```json
"vec":   { "type":"knn_vector","dimension":8,
           "mode":"on_disk","method":{"name":"hnsw","engine":"faiss","space_type":"l2"} },
"label": { "type":"keyword" }
```

- `index.knn:true` enables the k-NN engine; `mode:on_disk` + `engine:faiss` is the hard
  production shape: on disk this becomes a **native binary-quantized `.faiss` HNSW graph** plus
  Lucene quantized flat files (`.vec`/`.veb`) — not plain floats.
- `label` is a normal non-vector field, included to show non-vector data survives the round-trip.

Then doc `i` is ingested as `{"vec":[i,…,i],"label":"doc-i"}` with `refresh=true` (immediately
searchable). k-NN's codec quantizes each vector and builds the FAISS graph. The script prints
`vec(doc-3) = [3.0,…]` as the "before" value.

### STEP 2 — Register an fs repo and snapshot v1

```
PUT /_snapshot/demo-repo            {"type":"fs","settings":{"location":"/tmp/snapshots"}}
PUT /_snapshot/demo-repo/snap-model-v1?wait_for_completion=true   {"indices":"products"}
```

Registers the filesystem repository (rooted at the `path.repo` dir), then snapshots `products`
into `snap-model-v1` (`include_global_state:false` keeps it to just this index). OpenSearch writes
the blob-store format — `snap-<uuid>.dat`, `index-N`, `index.latest`, and chunked `__<uuid>` data
blobs wrapping the Lucene segment files (including the `.faiss`). This snapshot is the **input** to
the tool: "the embeddings from model v1, sitting in a backup."

### STEP 3 — Run the re-embed tool (the actual feature)

`reembed_tool.py` performs three sub-steps (printed `[1/3]…[3/3]`):

1. **restore** — `POST /_snapshot/demo-repo/snap-model-v1/_restore` renaming `products` →
   `products-work`. Old vectors are queryable again.
2. **re-embed** —
   - create a **fresh** `products-v2` with the same faiss/on_disk mapping;
   - `match_all` scan `products-work` to read every doc's `_source`. **Key fact:** for on_disk,
     k-NN *derived source* keeps the original float vector in `_source`, so the tool can read the
     old vector back;
   - for each doc, `source["vec"] = reembed(old, shift)` → `old + 100` (**this line is the stand-in
     for "run the new embedding model"**; non-vector fields are left untouched);
   - `POST /_bulk?refresh=true` the modified docs into `products-v2`. We never touch `.faiss` bytes —
     handing new floats to normal indexing makes `products-v2`'s k-NN codec **re-quantize and rebuild
     the native FAISS index** from scratch. That is the only correct way to change on_disk vectors.
3. **snapshot** — `PUT /_snapshot/demo-repo/snap-model-v2 {"indices":"products-v2"}`. This new
   snapshot is the **deliverable**: a backup containing the model-v2 embeddings.

### STEP 4 — Restore the new snapshot and verify

Restores `snap-model-v2` → `products-v2-restored` ("deploy the upgraded index"), then checks:

1. **Value:** `GET .../_doc/3` → `vec(doc-3)` must be `[103.0,…]` (survived snapshot→restore).
2. **Positive KNN:** a `knn` query for the **new** vector `[103,…]` must return `doc-3` as top hit —
   proving the native FAISS index was genuinely rebuilt and is searchable, not just that `_source`
   changed.
3. **Negative KNN:** a `knn` query for the **old** vector `[3,…]` returns some re-embedded doc
   (`doc-0`), not an exact doc-3 match — confirming all old vectors are gone.
4. **RESULT:** asserts `top hit == doc-3` **and** `vec(doc-3) == [103.0,…]`; prints `PASS` or fails
   the script with a non-zero exit.

## Why it works this way (the important part)

For `engine: faiss`, `mode: on_disk` the vector field is stored as a **native binary-quantized
`.faiss` HNSW index** plus Lucene quantized flat files (`.vec`/`.veb`), all derived from the
float vectors and **field-global quantization parameters**. You cannot byte-patch them; the vectors
must be regenerated by k-NN's codec. Approach A does that by **re-indexing whole documents** through
a fresh index, so the codec re-quantizes and rebuilds the native FAISS index during normal indexing.

The cost of that simplicity: **A re-indexes every field of every document**, not just the vector.
For an index with many or expensive non-vector fields that's wasteful — they're rebuilt for no
reason. Approach B (`vector-only/`) regenerates *only* the vector files and copies the rest
byte-for-byte, so it pays nothing for the non-vector fields. Same codec-rebuild for the vectors,
much less work overall.

## Files

| File | Purpose |
|------|---------|
| `run-demo.sh` | Approach-A orchestrator: Docker lifecycle + snapshot → re-embed → restore → verify |
| `reembed_tool.py` | Approach-A tool (pure stdlib): restore → replace all vectors via `_bulk` → re-snapshot. PoC transform `new = old + shift`; swap in a real model call where marked |
| `dest-mapping.json` | The faiss/on_disk `knn_vector` mapping (shared by both approaches) |
| `vector-only/` | **Approach B** — vector-only snapshot-blob rewrite (see `vector-only/README.md`) |

## Adapting to a real upgrade

- Point `reembed_tool.py` at your cluster (`--url`) and a real fs repository (`--repo`).
- Replace `reembed_vector()` with your model-v2 call (e.g. embed the document text/`_source`).
- Match `dest-mapping.json` to your field (dimension, space_type, compression, extra fields).
- Restore `--dest-snapshot` wherever you want the upgraded index.

### Scope / caveats (PoC)
- One `knn_vector` field; **all** vectors replaced (model-version upgrade).
- Relies on **derived source** keeping the float vector in `_source` (default for on_disk).
  If `_source` excludes the vector, supply new vectors from your own source instead of reading
  the old ones.
- Single-shard, single-node, in-memory scan of all docs (fine for a demo; a production tool
  would scroll/slice and stream `_bulk`).
- Non-vector fields ride along verbatim through `_source` → `_bulk`.
- `DISABLE_SECURITY_PLUGIN=true` is for the demo only; a real cluster keeps security on (use
  `https://` + credentials in the tool).
