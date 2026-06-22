#!/usr/bin/env bash
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Approach B end-to-end demo: VECTOR-ONLY snapshot rewrite. Replaces ALL vectors of one
# faiss/on_disk knn_vector field inside an OpenSearch snapshot by regenerating ONLY the vector
# segment files (the other fields' files are copied byte-for-byte), splicing them into the
# snapshot's blobs, and restoring the result. Runs a REAL OpenSearch 3.7.0 node in Docker.
#
#   ./run-demo.sh            # full pipeline -> verify
#   ./run-demo.sh --keep     # leave the container running afterwards (poke via curl on :19202)
#
# Requires: docker, curl, python3. The two rewriter tools compile against the jars bundled in the
# OpenSearch image; two extra jars (guava, mockito+deps) are taken from your local Gradle cache
# (guava is compileOnly in k-NN / provided at runtime in a real node; mockito only backs the
# standalone KNNSettings ClusterService stub). Override locations with env vars if needed.

set -euo pipefail

IMAGE="opensearchproject/opensearch:3.7.0"
CONTAINER="knn-vec-only-demo"
PORT=19202
B="http://localhost:${PORT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

DIM=8
NUM_DOCS=4
SHIFT=100
INDEX="idx"
SNAP="snap1"
WORK="work"
RESTORED="upgraded"
VEC_EXTS=".faiss,.vec,.vemf,.vemq,.veq,.vem,.vex"

say()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

cleanup() { [[ "$KEEP" -eq 0 ]] && docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# --- locate the two extra jars on the host (Gradle cache), overridable via env --------------
find_jar() { find "$HOME/.gradle/caches" -name "$1" 2>/dev/null | grep -v sources | grep -v javadoc | sort -V | tail -1; }
GUAVA_JAR="${GUAVA_JAR:-$(find_jar 'guava-*-jre.jar')}"
MOCKITO_JAR="${MOCKITO_JAR:-$(find_jar 'mockito-core-*.jar')}"
BYTEBUDDY_JAR="${BYTEBUDDY_JAR:-$(find_jar 'byte-buddy-[0-9]*.jar')}"
BYTEBUDDY_AGENT_JAR="${BYTEBUDDY_AGENT_JAR:-$(find_jar 'byte-buddy-agent-*.jar')}"
OBJENESIS_JAR="${OBJENESIS_JAR:-$(find_jar 'objenesis-*.jar')}"
for v in GUAVA_JAR MOCKITO_JAR BYTEBUDDY_JAR BYTEBUDDY_AGENT_JAR OBJENESIS_JAR; do
  [[ -n "${!v}" && -f "${!v}" ]] || die "could not find jar for $v (set \$$v to its path)"
done

OSH=/usr/share/opensearch
CP="/tmp:/tmp/guava.jar:/tmp/mlib/*:$OSH/lib/*:$OSH/plugins/opensearch-knn/*:$OSH/plugins/opensearch-knn/lib/*"
LP="-Djava.library.path=$OSH/plugins/opensearch-knn/lib"
JAVA="$OSH/jdk/bin/java --add-modules jdk.incubator.vector -cp \"$CP\" $LP"
FILTER='grep -vE "WARNING|incubator API|SLF4J|PanamaVect|Sharing is only|self-attaching"'

# ---------------------------------------------------------------------------
say "STEP 0  start OpenSearch 3.7.0 (k-NN + FAISS) in Docker (path.repo=/tmp)"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" -p ${PORT}:9200 \
  -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "DISABLE_INSTALL_DEMO_CONFIG=true" -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" \
  -e "path.repo=/tmp" "$IMAGE" >/dev/null
printf 'waiting for cluster'
for i in $(seq 1 60); do curl -s "$B" >/dev/null 2>&1 && { echo " up."; break; }; printf '.'; sleep 3; done
curl -s "$B/_cat/plugins" | grep -i knn | sed 's/^/  plugin: /'

say "STEP 1  stage tools + jars, compile both rewriters in-container"
docker exec "$CONTAINER" mkdir -p /tmp/mlib
docker cp "$GUAVA_JAR"           "$CONTAINER:/tmp/guava.jar"
docker cp "$MOCKITO_JAR"         "$CONTAINER:/tmp/mlib/"
docker cp "$BYTEBUDDY_JAR"       "$CONTAINER:/tmp/mlib/"
docker cp "$BYTEBUDDY_AGENT_JAR" "$CONTAINER:/tmp/mlib/"
docker cp "$OBJENESIS_JAR"       "$CONTAINER:/tmp/mlib/"
docker cp "$HERE/VectorOnlySegmentRewriter.java"  "$CONTAINER:/tmp/"
docker cp "$HERE/SnapshotBlobVectorRewriter.java" "$CONTAINER:/tmp/"
docker exec "$CONTAINER" sh -c "$OSH/jdk/bin/javac -cp \"$CP\" -d /tmp \
  /tmp/VectorOnlySegmentRewriter.java /tmp/SnapshotBlobVectorRewriter.java" \
  && echo "  compiled both tools" || die "compile failed"

say "STEP 2  create faiss/on_disk index '$INDEX' (derived source on) + ingest $NUM_DOCS docs"
curl -s -X PUT "$B/$INDEX" -H 'Content-Type: application/json' -d '{
 "settings":{"index.knn":true,"number_of_shards":1,"number_of_replicas":0,"index.knn.derived_source.enabled":true},
 "mappings":{"properties":{
   "vec":{"type":"knn_vector","dimension":'"$DIM"',"mode":"on_disk","method":{"name":"hnsw","engine":"faiss","space_type":"l2"}},
   "label":{"type":"keyword"}}}}' >/dev/null
for i in $(seq 0 $((NUM_DOCS-1))); do
  v=$(python3 -c "print('['+','.join([str($i)]*$DIM)+']')")
  curl -s -X PUT "$B/$INDEX/_doc/$i?refresh=true" -H 'Content-Type: application/json' \
    -d "{\"vec\":$v,\"label\":\"doc-$i\"}" >/dev/null
done
curl -s -X POST "$B/$INDEX/_forcemerge?max_num_segments=1" >/dev/null
curl -s -X POST "$B/$INDEX/_refresh" >/dev/null
echo "  ingested; original vec(doc-1) = $(curl -s "$B/$INDEX/_doc/1" | python3 -c 'import sys,json;print(json.load(sys.stdin)["_source"]["vec"])')"

say "STEP 3  snapshot '$INDEX' -> srcrepo/$SNAP (the INPUT snapshot)"
curl -s -X PUT "$B/_snapshot/srcrepo" -H 'Content-Type: application/json' \
  -d '{"type":"fs","settings":{"location":"/tmp/srcrepo"}}' >/dev/null
curl -s -X PUT "$B/_snapshot/srcrepo/$SNAP?wait_for_completion=true" -H 'Content-Type: application/json' \
  -d "{\"indices\":\"$INDEX\",\"include_global_state\":false}" \
  | python3 -c "import sys,json;print('  snapshot state:',json.load(sys.stdin)['snapshot']['state'])"

say "STEP 4  restore -> '$WORK' to materialize the shard's Lucene files; copy to /tmp/srcSeg"
curl -s -X POST "$B/_snapshot/srcrepo/$SNAP/_restore?wait_for_completion=true" -H 'Content-Type: application/json' \
  -d "{\"indices\":\"$INDEX\",\"rename_pattern\":\"(.+)\",\"rename_replacement\":\"$WORK\"}" >/dev/null
WU=$(curl -s "$B/$WORK/_settings" | python3 -c "import sys,json;print(json.load(sys.stdin)['$WORK']['settings']['index']['uuid'])")
docker exec "$CONTAINER" sh -c "
  SH=$OSH/data/nodes/0/indices/$WU/0/index
  rm -rf /tmp/srcSeg /tmp/destSeg && mkdir -p /tmp/srcSeg /tmp/destSeg
  for f in \$(ls \$SH | grep -v write.lock); do cp \$SH/\$f /tmp/srcSeg/; done
"
echo "  srcSeg: $(docker exec "$CONTAINER" sh -c 'ls /tmp/srcSeg | wc -l') files"

say "STEP 5  VectorOnlySegmentRewriter: regenerate ONLY vector files (new = (docId+1)+d+$SHIFT)"
docker exec "$CONTAINER" sh -c "$JAVA VectorOnlySegmentRewriter /tmp/srcSeg /tmp/destSeg vec $SHIFT 2>&1 | $FILTER"

say "STEP 6  copy repo -> destrepo; SnapshotBlobVectorRewriter splices new vectors into the blobs"
IID=$(docker exec "$CONTAINER" sh -c 'ls /tmp/srcrepo/indices | head -1')
SUID=$(docker exec "$CONTAINER" sh -c 'ls /tmp/srcrepo | grep "^snap-" | head -1 | sed "s/^snap-//;s/\.dat$//"')
echo "  indexId=$IID snapUuid=$SUID"
docker exec "$CONTAINER" sh -c "cp -r /tmp/srcrepo /tmp/destrepo"
docker exec "$CONTAINER" sh -c "$JAVA SnapshotBlobVectorRewriter /tmp/destrepo $IID 0 $SUID /tmp/destSeg '$VEC_EXTS' 2>&1 | $FILTER"

say "STEP 7  register destrepo + restore $SNAP -> '$RESTORED'"
curl -s -X PUT "$B/_snapshot/destrepo" -H 'Content-Type: application/json' \
  -d '{"type":"fs","settings":{"location":"/tmp/destrepo"}}' >/dev/null
curl -s -X POST "$B/_snapshot/destrepo/$SNAP/_restore?wait_for_completion=true" -H 'Content-Type: application/json' \
  -d "{\"indices\":\"$INDEX\",\"rename_pattern\":\"(.+)\",\"rename_replacement\":\"$RESTORED\"}" \
  | python3 -c "import sys,json;print('  restore shards:',json.load(sys.stdin)['snapshot']['shards'])"
sleep 3
H=$(curl -s "$B/_cluster/health/$RESTORED?wait_for_status=yellow&timeout=30s" | python3 -c "import sys,json;print(json.load(sys.stdin).get('status'))")
echo "  restored index health: $H"
[[ "$H" == "green" || "$H" == "yellow" ]] || die "restore did not go healthy (status=$H)"

say "STEP 8  verify"
# 8a) KNN self-match: each restored doc's OWN new vector must return itself with score 1.0
pass=0
for i in $(seq 0 $((NUM_DOCS-1))); do
  QV=$(curl -s "$B/$RESTORED/_doc/$i" | python3 -c "import sys,json;print(json.dumps(json.load(sys.stdin)['_source']['vec']))")
  RES=$(curl -s -X POST "$B/$RESTORED/_search" -H 'Content-Type: application/json' \
    -d "{\"size\":1,\"query\":{\"knn\":{\"vec\":{\"vector\":$QV,\"k\":1}}}}" \
    | python3 -c "import sys,json;h=json.load(sys.stdin)['hits']['hits'][0];print(h['_source']['label'],round(h['_score'],4))")
  echo "  doc-$i own-vector query -> $RES"
  echo "$RES" | grep -q "doc-$i 1.0" && pass=$((pass+1))
done
# 8b) old vectors are gone: an original vector should NOT exact-match
OLD=$(curl -s -X POST "$B/$RESTORED/_search" -H 'Content-Type: application/json' \
  -d "{\"size\":1,\"query\":{\"knn\":{\"vec\":{\"vector\":$(python3 -c "print('['+','.join(['1']*$DIM)+']')"),\"k\":1}}}}" \
  | python3 -c "import sys,json;print(round(json.load(sys.stdin)['hits']['hits'][0]['_score'],6))")
echo "  old-vector [1..] query top score = $OLD (≪1 => no old vector remains)"
# 8c) source repo untouched: original snapshot still restores OLD vectors
curl -s -X POST "$B/_snapshot/srcrepo/$SNAP/_restore?wait_for_completion=true" -H 'Content-Type: application/json' \
  -d "{\"indices\":\"$INDEX\",\"rename_pattern\":\"(.+)\",\"rename_replacement\":\"orig-check\"}" >/dev/null
sleep 2
ORIG=$(curl -s "$B/orig-check/_doc/1" | python3 -c "import sys,json;print(json.load(sys.stdin)['_source']['vec'])")
echo "  source repo restore: orig doc-1 vec = $ORIG (unchanged)"

say "RESULT"
OK=$(python3 -c "print('1' if float('$OLD') < 0.5 else '0')")
if [[ "$pass" -eq "$NUM_DOCS" && "$OK" == "1" && "$ORIG" == "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]" ]]; then
  echo "PASS: vector-only snapshot rewrite produced a restorable faiss/on_disk index with the NEW"
  echo "      vectors ($pass/$NUM_DOCS KNN self-match), old vectors gone, source repo untouched,"
  echo "      and the non-vector segment files were copied byte-for-byte (see STEP 5 output)."
else
  die "verification failed (self-match $pass/$NUM_DOCS, old-score-ok=$OK, orig=$ORIG)"
fi

[[ "$KEEP" -eq 1 ]] && echo && echo "(container '$CONTAINER' left running on $B ; remove: docker rm -f $CONTAINER)" || true
