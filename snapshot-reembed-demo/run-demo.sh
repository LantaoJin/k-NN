#!/usr/bin/env bash
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end demo: re-embed (replace ALL vectors of) a faiss/on_disk knn_vector field that
# lives inside an OpenSearch SNAPSHOT, producing a NEW snapshot you can restore as the
# upgraded index. Runs a REAL OpenSearch 3.7.0 node in Docker (k-NN + FAISS bundled), so the
# native on_disk index is genuinely rebuilt by k-NN's codec.
#
#   ./run-demo.sh            # full demo: start container -> build -> snapshot -> re-embed -> verify
#   ./run-demo.sh --keep     # leave the container running afterwards (for poking via curl)
#
# Requires: docker, curl, python3, jq.

set -euo pipefail

IMAGE="opensearchproject/opensearch:3.7.0"
CONTAINER="knn-reembed-demo"
PORT=19200
B="http://localhost:${PORT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

DIM=8
NUM_DOCS=8
SHIFT=100
REPO="demo-repo"
SRC_INDEX="products"
SRC_SNAP="snap-model-v1"
WORK_INDEX="products-work"
DEST_INDEX="products-v2"
DEST_SNAP="snap-model-v2"
RESTORED="products-v2-restored"

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

cleanup() {
  if [[ "$KEEP" -eq 0 ]]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
say "STEP 0  start OpenSearch 3.7.0 (k-NN + FAISS) in Docker"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" \
  -p ${PORT}:9200 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "DISABLE_INSTALL_DEMO_CONFIG=true" \
  -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" \
  -e "path.repo=/tmp/snapshots" \
  "$IMAGE" >/dev/null
docker exec "$CONTAINER" sh -c 'mkdir -p /tmp/snapshots && chown opensearch:opensearch /tmp/snapshots' >/dev/null 2>&1 || true

printf 'waiting for cluster'
for i in $(seq 1 60); do
  if curl -s "$B" >/dev/null 2>&1; then echo " up."; break; fi
  printf '.'; sleep 3
done
curl -s "$B/_cat/plugins" | grep -i knn | sed 's/^/  plugin: /'

# ---------------------------------------------------------------------------
say "STEP 1  create the source faiss/on_disk index (model v1) and ingest docs"
curl -s -X PUT "$B/$SRC_INDEX" -H 'Content-Type: application/json' \
  --data-binary @"$HERE/dest-mapping.json" >/dev/null   # same mapping shape as dest
echo "created index '$SRC_INDEX' (knn_vector 'vec', engine=faiss, mode=on_disk)"
for i in $(seq 0 $((NUM_DOCS-1))); do
  v=$(python3 -c "print('['+','.join([str($i)]*$DIM)+']')")
  curl -s -X PUT "$B/$SRC_INDEX/_doc/$i?refresh=true" -H 'Content-Type: application/json' \
    -d "{\"vec\":$v,\"label\":\"doc-$i\"}" >/dev/null
done
echo "ingested $NUM_DOCS docs; original vec(doc-3) = $(curl -s "$B/$SRC_INDEX/_doc/3" | jq -c '._source.vec')"

say "STEP 2  register fs repo + snapshot the v1 index"
curl -s -X PUT "$B/_snapshot/$REPO" -H 'Content-Type: application/json' \
  -d '{"type":"fs","settings":{"location":"/tmp/snapshots"}}' | jq -c .
curl -s -X PUT "$B/_snapshot/$REPO/$SRC_SNAP?wait_for_completion=true" -H 'Content-Type: application/json' \
  -d "{\"indices\":\"$SRC_INDEX\",\"include_global_state\":false}" | jq -c '{snapshot:.snapshot.snapshot,state:.snapshot.state}'

say "STEP 3  run the re-embed tool: restore -> replace ALL vectors (new = old + $SHIFT) -> new snapshot"
python3 "$HERE/reembed_tool.py" \
  --url "$B" --repo "$REPO" \
  --src-snapshot "$SRC_SNAP" --src-index "$SRC_INDEX" \
  --work-index "$WORK_INDEX" --dest-index "$DEST_INDEX" --dest-snapshot "$DEST_SNAP" \
  --field vec --mapping-file "$HERE/dest-mapping.json" --shift "$SHIFT"

say "STEP 4  restore the NEW snapshot as the upgraded index and verify"
curl -s -X POST "$B/_snapshot/$REPO/$DEST_SNAP/_restore?wait_for_completion=true" -H 'Content-Type: application/json' \
  -d "{\"indices\":\"$DEST_INDEX\",\"rename_pattern\":\"(.+)\",\"rename_replacement\":\"${RESTORED}\"}" \
  | jq -c '{restored:.snapshot.indices, shards:.snapshot.shards}'
sleep 2
curl -s -X POST "$B/$RESTORED/_refresh" >/dev/null

echo
echo "verify: doc-3 vector in the RESTORED upgraded index (expect original+$SHIFT):"
RESTORED_VEC=$(curl -s "$B/$RESTORED/_doc/3" | jq -c '._source.vec')
echo "  restored vec(doc-3) = $RESTORED_VEC"

echo
echo "verify: KNN search on the RESTORED index near the NEW doc-3 vector (expect top hit = doc-3):"
QV=$(python3 -c "print('['+','.join([str(3+$SHIFT)]*$DIM)+']')")
TOP=$(curl -s -X POST "$B/$RESTORED/_search" -H 'Content-Type: application/json' \
  -d "{\"size\":1,\"query\":{\"knn\":{\"vec\":{\"vector\":$QV,\"k\":1}}}}" | jq -r '.hits.hits[0]._source.label')
echo "  top hit = $TOP"

echo
echo "verify: KNN search near the OLD doc-3 vector should NOT strongly match (old vectors are gone):"
OQV=$(python3 -c "print('['+','.join(['3']*$DIM)+']')")
OLDTOP=$(curl -s -X POST "$B/$RESTORED/_search" -H 'Content-Type: application/json' \
  -d "{\"size\":1,\"query\":{\"knn\":{\"vec\":{\"vector\":$OQV,\"k\":1}}}}" | jq -r '.hits.hits[0]._source.label')
echo "  top hit for OLD query = $OLDTOP (a re-embedded doc, not the original doc-3 match)"

say "RESULT"
if [[ "$TOP" == "doc-3" && "$RESTORED_VEC" == "[103.0,103.0,103.0,103.0,103.0,103.0,103.0,103.0]" ]]; then
  echo "PASS: snapshot re-embed produced a restorable faiss/on_disk index with the NEW vectors."
else
  echo "UNEXPECTED: restored_vec=$RESTORED_VEC top=$TOP"
  exit 1
fi

if [[ "$KEEP" -eq 1 ]]; then
  echo
  echo "(container '$CONTAINER' left running on $B ; remove with: docker rm -f $CONTAINER)"
fi
