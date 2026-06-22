#!/usr/bin/env python3
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Snapshot vector re-embed tool (PoC) for OpenSearch k-NN, faiss + on_disk.

Goal: upgrade the embedding model of a single knn_vector field by REPLACING ALL its
vectors, without re-ingesting the other fields by hand. For faiss/on_disk the vector
field is a native binary-quantized .faiss index + Lucene quantized flat files; you cannot
byte-patch it. The only correct way to change the vectors is to let k-NN's codec rebuild
the field. This tool therefore orchestrates, over the REST API:

  1. restore  the source snapshot to a working index;
  2. re-embed  read every doc's _source (derived-source keeps the original float vector in
               _source), apply a transform to the vector field, and bulk-index into a fresh
               index that has the SAME faiss/on_disk mapping -> the destination's k-NN codec
               re-quantizes and rebuilds the native .faiss index for the new vectors;
  3. snapshot  the rebuilt index into a new snapshot, ready to restore as the upgraded index.

The HNSW graph + quantization are rebuilt by k-NN during step 2's indexing, exactly as a
normal ingest would (no graph work is done here).

PoC transform: newVector = oldVector + SHIFT (full replacement, deterministic). Swap in a
real model call where indicated.

Pure stdlib (urllib + json) so it runs anywhere with python3.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


class OpenSearch:
    def __init__(self, base):
        self.base = base.rstrip("/")

    def _req(self, method, path, body=None, params=None):
        url = self.base + path
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        data = None
        headers = {"Content-Type": "application/json"}
        if body is not None:
            data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode("utf-8")
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8")
            raise SystemExit(f"{method} {path} -> HTTP {e.code}: {raw}")

    def restore(self, repo, snapshot, src_index, work_index):
        body = {
            "indices": src_index,
            "rename_pattern": "(.+)",
            "rename_replacement": work_index,
            "include_global_state": False,
        }
        self._req("POST", f"/_snapshot/{repo}/{snapshot}/_restore", body, {"wait_for_completion": "true"})

    def create_index(self, index, settings_and_mapping):
        self._req("PUT", f"/{index}", settings_and_mapping)

    def refresh(self, index):
        self._req("POST", f"/{index}/_refresh")

    def scan_all(self, index, size=10000):
        # PoC volumes are small; one large match_all is fine.
        _, resp = self._req("GET", f"/{index}/_search", {"query": {"match_all": {}}, "size": size})
        return [(h["_id"], h["_source"]) for h in resp["hits"]["hits"]]

    def bulk(self, lines):
        body = "\n".join(lines) + "\n"
        _, resp = self._req("POST", "/_bulk", body, {"refresh": "true"})
        if resp.get("errors"):
            raise SystemExit("bulk had errors: " + json.dumps(resp)[:2000])

    def snapshot(self, repo, snapshot, index):
        body = {"indices": index, "include_global_state": False}
        self._req("PUT", f"/_snapshot/{repo}/{snapshot}", body, {"wait_for_completion": "true"})


def reembed_vector(old, shift):
    # ---- PoC stand-in for "run the new embedding model" ----
    # Replace this with a real model call: new = model_v2(doc_text).
    return [float(x) + shift for x in old]


def main():
    ap = argparse.ArgumentParser(description="OpenSearch snapshot vector re-embed (faiss/on_disk PoC)")
    ap.add_argument("--url", default="http://localhost:9200")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--src-snapshot", required=True)
    ap.add_argument("--src-index", required=True)
    ap.add_argument("--work-index", required=True, help="restore target (scratch)")
    ap.add_argument("--dest-index", required=True, help="rebuilt index with new vectors")
    ap.add_argument("--dest-snapshot", required=True)
    ap.add_argument("--field", required=True, help="the knn_vector field to replace")
    ap.add_argument("--mapping-file", required=True, help="JSON settings+mappings for dest index")
    ap.add_argument("--shift", type=float, default=100.0, help="PoC transform: new = old + shift")
    args = ap.parse_args()

    os_ = OpenSearch(args.url)

    print(f"[1/3] restoring {args.src_snapshot} -> {args.work_index}")
    os_.restore(args.repo, args.src_snapshot, args.src_index, args.work_index)
    os_.refresh(args.work_index)

    print(f"[2/3] re-embedding field '{args.field}' (new = old + {args.shift}) -> {args.dest_index}")
    with open(args.mapping_file) as f:
        dest_mapping = json.load(f)
    os_.create_index(args.dest_index, dest_mapping)

    docs = os_.scan_all(args.work_index)
    lines = []
    for doc_id, source in docs:
        old = source.get(args.field)
        if old is None:
            raise SystemExit(f"doc {doc_id} has no '{args.field}' in _source (derived source disabled?)")
        source[args.field] = reembed_vector(old, args.shift)
        lines.append(json.dumps({"index": {"_index": args.dest_index, "_id": doc_id}}))
        lines.append(json.dumps(source))
    if lines:
        os_.bulk(lines)
    os_.refresh(args.dest_index)
    print(f"      re-embedded {len(docs)} docs")

    print(f"[3/3] snapshotting {args.dest_index} -> {args.dest_snapshot}")
    os_.snapshot(args.repo, args.dest_snapshot, args.dest_index)

    print(f"DONE. Restore '{args.dest_snapshot}' from repo '{args.repo}' to get the upgraded index.")


if __name__ == "__main__":
    main()
