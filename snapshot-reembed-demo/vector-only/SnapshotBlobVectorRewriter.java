/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.blobstore.BlobContainer;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.common.blobstore.fs.FsBlobStore;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.compress.CompressorRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.snapshots.blobstore.BlobStoreIndexShardSnapshot;
import org.opensearch.index.snapshots.blobstore.BlobStoreIndexShardSnapshot.FileInfo;
import org.opensearch.index.snapshots.blobstore.BlobStoreIndexShardSnapshots;
import org.opensearch.index.snapshots.blobstore.SnapshotFiles;
import org.opensearch.index.store.StoreFileMetadata;
import org.opensearch.repositories.blobstore.ChecksumBlobStoreFormat;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Edits an OpenSearch fs-repository SNAPSHOT at the blob level so a single index's shard gets new
 * vectors for ONE knn_vector field, WITHOUT a running node and WITHOUT touching the other fields'
 * data. Pairs with {@code VectorOnlySegmentRewriter}, which regenerates only the vector segment
 * files (byte-identical others); this tool splices those into the snapshot's blobs.
 *
 * <p>Flow (operates on a repo dir that is a COPY of the source — caller copies first):
 * <ol>
 *   <li>Read the shard's {@code BlobStoreIndexShardSnapshot} (snap-&lt;uuid&gt;.dat) — its FileInfo
 *       list maps each Lucene file (physicalName) to a {@code __&lt;blob&gt;} data blob + checksum.
 *   <li>For each REWRITTEN vector file (provided in a local dir): write its bytes as a NEW
 *       {@code __&lt;uuid&gt;} data blob and build a new FileInfo with recomputed length + checksum.
 *       For files the rewrite did NOT change, reuse the existing FileInfo verbatim (blob already
 *       present, byte-identical — the proven invariant).
 *   <li>Drop FileInfos for source vector files that no longer exist (e.g. compound .faissc replaced
 *       by loose faiss/.vec/...); add FileInfos for the new loose vector files.
 *   <li>Write a NEW shard {@code BlobStoreIndexShardSnapshot} (snap-&lt;newSnapUuid&gt;.dat) and a new
 *       {@code BlobStoreIndexShardSnapshots} (index-&lt;newGen&gt;) listing it alongside existing ones.
 * </ol>
 *
 * <p>This tool intentionally edits only the SHARD-level blobs. The caller is responsible for the
 * root-level registration (snap-&lt;uuid&gt;.dat SnapshotInfo, RepositoryData index-N, index.latest)
 * — for the demo we keep the existing snapshot identity so the original snapshot name restores the
 * NEW vectors (simplest provable round-trip). See the demo script.
 *
 * <p>Usage:
 * {@code SnapshotBlobVectorRewriter <repoDir> <indexId> <shardId> <snapUuid> <rewrittenSegDir> <vecExtCsv>}
 */
public final class SnapshotBlobVectorRewriter {

    // Shard-level metadata formats (same constants BlobStoreRepository uses), reconstructed here.
    private static final ChecksumBlobStoreFormat<BlobStoreIndexShardSnapshot> SNAP_FORMAT =
            new ChecksumBlobStoreFormat<>("snapshot", "snap-%s.dat", BlobStoreIndexShardSnapshot::fromXContent);
    private static final ChecksumBlobStoreFormat<BlobStoreIndexShardSnapshots> SNAPS_FORMAT =
            new ChecksumBlobStoreFormat<>("snapshots", "index-%s", BlobStoreIndexShardSnapshots::fromXContent);

    public static void main(String[] args) throws Exception {
        if (args.length < 6) {
            System.err.println("usage: SnapshotBlobVectorRewriter <repoDir> <indexId> <shardId> "
                    + "<snapUuid> <rewrittenSegDir> <vecExtCsv>");
            System.exit(2);
        }
        Path repoDir = Path.of(args[0]);
        String indexId = args[1];
        String shardId = args[2];
        String snapUuid = args[3];
        Path segDir = Path.of(args[4]);
        String[] vecExts = args[5].split(",");

        FsBlobStore store = new FsBlobStore(8192, repoDir, false);
        BlobContainer shard = store.blobContainer(
                BlobPath.cleanPath().add("indices").add(indexId).add(shardId));

        // 1) Read the existing shard snapshot manifest.
        BlobStoreIndexShardSnapshot snap = SNAP_FORMAT.read(shard, snapUuid, NamedXContentRegistry.EMPTY);
        System.out.println("read shard snapshot '" + snap.snapshot() + "' with " + snap.indexFiles().size() + " files");

        // 2) Partition existing FileInfos: keep non-vector ones; the vector ones are replaced.
        List<FileInfo> newFiles = new ArrayList<>();
        int keptOther = 0, droppedVec = 0;
        for (FileInfo fi : snap.indexFiles()) {
            if (isVectorFile(fi.physicalName(), vecExts)) {
                droppedVec++;
            } else {
                newFiles.add(fi); // reuse blob + checksum verbatim (proven byte-identical)
                keptOther++;
            }
        }

        // 3) Write the rewritten vector files from segDir as new data blobs + new FileInfos.
        int addedVec = 0;
        try (Directory segFs = FSDirectory.open(segDir)) {
            for (String fileName : Files.list(segDir).map(p -> p.getFileName().toString()).sorted().toList()) {
                if (!isVectorFile(fileName, vecExts)) {
                    continue;
                }
                long length = Files.size(segDir.resolve(fileName));
                String checksum;
                try (IndexInput in = segFs.openInput(fileName, IOContext.READONCE)) {
                    checksum = digestToString(CodecUtil.retrieveChecksum(in));
                }
                // writtenBy: reuse from any existing FileInfo's metadata (same Lucene version);
                // fall back to null is not allowed, so read it off the first kept file.
                org.apache.lucene.util.Version writtenBy = anyWrittenBy(snap.indexFiles());
                StoreFileMetadata md = new StoreFileMetadata(fileName, length, checksum, writtenBy, null);
                String blobName = "__" + UUID.randomUUID().toString();
                // single-part: partSize null -> one part named exactly blobName
                FileInfo fi = new FileInfo(blobName, md, null);
                // upload the bytes
                try (InputStream is = Files.newInputStream(segDir.resolve(fileName))) {
                    shard.writeBlob(blobName, is, length, true);
                }
                newFiles.add(fi);
                addedVec++;
            }
        }
        System.out.println("files: kept " + keptOther + " others, dropped " + droppedVec
                + " old vector, added " + addedVec + " new vector");

        // 4) Write a NEW shard snapshot manifest under the SAME snapshot uuid (overwrite), so the
        //    existing root metadata (snap-<uuid>.dat / RepositoryData) still points to it and a
        //    restore of the original snapshot name yields the new vectors.
        // indexVersion has no public getter and is never read by the restore path (only serialized
        // as metadata), so 0 is safe here.
        BlobStoreIndexShardSnapshot newSnap = new BlobStoreIndexShardSnapshot(
                snap.snapshot(), 0L, newFiles, snap.startTime(), snap.time(),
                addedVec, sumLength(newFiles));
        // delete the old snap-<uuid>.dat then write the new one (writeBlob failIfAlreadyExists=true)
        shard.deleteBlobsIgnoringIfNotExists(List.of(SNAP_FORMAT.blobName(snapUuid)));
        SNAP_FORMAT.write(newSnap, shard, snapUuid, CompressorRegistry.none());

        // 5) Rewrite the shard-level index-<gen> (BlobStoreIndexShardSnapshots) to reference the new
        //    file list for this snapshot. Find the existing generation, rebuild, write same gen.
        rewriteShardIndex(shard, snap.snapshot(), newFiles);

        System.out.println("DONE: spliced new vectors into snapshot '" + snap.snapshot() + "' (shard " + shardId + ")");
        store.close();
    }

    private static void rewriteShardIndex(BlobContainer shard, String snapshotName, List<FileInfo> newFiles)
            throws Exception {
        // Find index-<gen> blob.
        String genBlob = null;
        for (String name : shard.listBlobs().keySet()) {
            if (name.startsWith("index-")) { genBlob = name; break; }
        }
        if (genBlob == null) {
            System.out.println("  (no shard index-<gen> blob found; skipping snapshots-index rewrite)");
            return;
        }
        String gen = genBlob.substring("index-".length());
        BlobStoreIndexShardSnapshots existing = SNAPS_FORMAT.read(shard, gen, NamedXContentRegistry.EMPTY);
        List<SnapshotFiles> updated = new ArrayList<>();
        for (SnapshotFiles sf : existing.snapshots()) {
            if (sf.snapshot().equals(snapshotName)) {
                updated.add(new SnapshotFiles(snapshotName, newFiles, sf.shardStateIdentifier()));
            } else {
                updated.add(sf);
            }
        }
        shard.deleteBlobsIgnoringIfNotExists(List.of(genBlob));
        SNAPS_FORMAT.write(new BlobStoreIndexShardSnapshots(updated), shard, gen, CompressorRegistry.none());
        System.out.println("  rewrote shard index-" + gen + " (" + updated.size() + " snapshot(s))");
    }

    private static boolean isVectorFile(String f, String[] exts) {
        for (String e : exts) {
            String ext = e.trim();
            if (!ext.isEmpty() && f.endsWith(ext)) return true;
        }
        return false;
    }

    private static org.apache.lucene.util.Version anyWrittenBy(List<FileInfo> files) {
        for (FileInfo fi : files) {
            if (fi.metadata().writtenBy() != null) return fi.metadata().writtenBy();
        }
        return org.apache.lucene.util.Version.LATEST;
    }

    private static long sumLength(List<FileInfo> files) {
        long s = 0;
        for (FileInfo fi : files) s += fi.length();
        return s;
    }

    /** Mirror of Store.digestToString (base-36). */
    private static String digestToString(long digest) {
        return Long.toString(digest, Character.MAX_RADIX);
    }
}
