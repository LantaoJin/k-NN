/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.TrackingDirectoryWrapper;
import org.apache.lucene.util.InfoStream;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Standalone, VECTOR-ONLY segment rewriter for an OpenSearch k-NN faiss/on_disk index with
 * DERIVED SOURCE enabled.
 *
 * <p>Goal (the thing a full reindex gets wrong): replace ALL vectors of the one knn_vector field
 * for an embedding-model upgrade, WITHOUT reindexing the other fields. For derived-source segments
 * the vector is masked out of stored fields (.fdt) and re-injected at read time from the vector
 * files, so:
 *
 * <ul>
 *   <li>regenerate ONLY the vector field's files (.faiss + Lucene quantized .vec/.veb) from new
 *       vectors, via the segment's own k-NN codec (it re-quantizes + rebuilds the native index);
 *   <li>copy every OTHER segment file verbatim — stored fields (.fdt/.fdx, already vector-masked),
 *       postings (.doc/.tim/.tip), doc-values (.dvd/.dvm), points (.kdd/.kdi/.kdm), norms
 *       (.nvd/.nvm), .fnm — byte-for-byte;
 *   <li>rewrite the segment info (.si), preserving its attributes (incl. derived_vector_fields).
 * </ul>
 *
 * <p>Because we REPLACE all vectors, we never read the old values (which would be quantized
 * approximations anyway): we enumerate the docIDs that have a vector and synthesize new ones.
 *
 * <p>Usage: {@code VectorOnlySegmentRewriter <srcSegmentDir> <destDir> <vectorField> <shift>}
 * (PoC transform: new[d] = (docId+1) + d + shift, deterministic, independent of the old value).
 * The codec (KNN1040Codec) is resolved automatically from each segment's .si via Lucene's SPI;
 * the k-NN jar + native faiss lib must be on the classpath / java.library.path.
 */
public final class VectorOnlySegmentRewriter {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("usage: VectorOnlySegmentRewriter <srcDir> <destDir> <vectorField> <shift>");
            System.exit(2);
        }
        Path srcPath = Path.of(args[0]);
        Path destPath = Path.of(args[1]);
        String vectorField = args[2];
        float shift = Float.parseFloat(args[3]);

        // The native faiss SQ writer reads KNNSettings (e.g. index thread qty) via a ClusterService
        // singleton that a real node wires up. Standalone, we give KNNSettings a real ClusterService
        // backed by a ClusterSettings registered with k-NN's node-scoped settings (defaults are fine).
        initKNNSettings();

        try (Directory srcDir = FSDirectory.open(srcPath); Directory destDir = FSDirectory.open(destPath)) {
            SegmentInfos srcInfos = SegmentInfos.readLatestCommit(srcDir);
            System.out.println("source commit: " + srcInfos.size() + " segment(s)");

            SegmentInfos destInfos = new SegmentInfos(srcInfos.getIndexCreatedVersionMajor());
            destInfos.counter = srcInfos.counter;
            destInfos.version = srcInfos.version;
            destInfos.setUserData(srcInfos.getUserData(), false);

            Set<String> created = new HashSet<>();
            for (SegmentCommitInfo srcCommit : srcInfos) {
                SegmentCommitInfo destCommit = rewriteSegment(srcCommit, destDir, vectorField, shift);
                destInfos.add(destCommit);
                created.addAll(destCommit.files());
            }
            destDir.sync(created);
            destInfos.commit(destDir);
            System.out.println("wrote rewritten commit to " + destPath);
        }
    }

    /** Give KNNSettings a ClusterService backed by k-NN's node-scoped settings (defaults). */
    private static void initKNNSettings() {
        Set<Setting<?>> clusterSettingsSet = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        for (Setting<?> s : KNNSettings.state().getSettings()) {
            if (s.getProperties().contains(Setting.Property.NodeScope)) {
                clusterSettingsSet.add(s);
            }
        }
        ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, clusterSettingsSet);
        ClusterService clusterService = org.mockito.Mockito.mock(ClusterService.class);
        org.mockito.Mockito.when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
    }

    private static SegmentCommitInfo rewriteSegment(
            SegmentCommitInfo srcCommit, Directory destDir, String vectorField, float shift) throws Exception {
        SegmentInfo srcInfo = srcCommit.info;
        Codec codec = srcInfo.getCodec();
        Directory srcDir = srcInfo.dir;

        Directory readDir = srcDir;
        CompoundDirectory cfs = null;
        if (srcInfo.getUseCompoundFile()) {
            cfs = codec.compoundFormat().getCompoundReader(srcDir, srcInfo);
            readDir = cfs;
        }
        try {
            FieldInfos fieldInfos = codec.fieldInfosFormat().read(readDir, srcInfo, "", IOContext.READONCE);

            // Fresh non-compound dest segment with identical identity + attributes (preserves
            // the derived_vector_fields segment attribute so derived source keeps working).
            SegmentInfo destInfo = new SegmentInfo(
                    destDir, srcInfo.getVersion(), srcInfo.getMinVersion(), srcInfo.name, srcInfo.maxDoc(),
                    false, srcInfo.getHasBlocks(), codec, srcInfo.getDiagnostics(), srcInfo.getId(),
                    srcInfo.getAttributes(), srcInfo.getIndexSort());

            TrackingDirectoryWrapper trackingDir = new TrackingDirectoryWrapper(destDir);

            // 1) Regenerate ONLY the vector field's files with NEW vectors.
            writeNewVectors(codec, trackingDir, readDir, srcInfo, fieldInfos, vectorField, shift);

            // 2) Copy every other member file verbatim (NOT vector files, NOT .si/.fnm).
            List<String> verbatim = copyOtherFiles(readDir, trackingDir, srcInfo);

            // 3) Rewrite .fnm (vector writer stamped per-field format/suffix attrs onto fieldInfos).
            codec.fieldInfosFormat().write(trackingDir, destInfo, "", fieldInfos, IOContext.DEFAULT);

            // 4) Record files + write .si.
            destInfo.setFiles(new HashSet<>(trackingDir.getCreatedFiles()));
            codec.segmentInfoFormat().write(destDir, destInfo, IOContext.DEFAULT);

            // INVARIANT CHECK: every non-vector file we copied must be byte-identical in dest.
            assertByteIdentical(readDir, destDir, verbatim);
            System.out.println("  segment " + srcInfo.name + ": " + verbatim.size()
                    + " non-vector files byte-identical; vector files regenerated");

            return new SegmentCommitInfo(destInfo, 0, 0, -1L, -1L, -1L, srcCommit.getId());
        } finally {
            if (cfs != null) cfs.close();
        }
    }

    private static void writeNewVectors(
            Codec codec, TrackingDirectoryWrapper trackingDir, Directory readDir, SegmentInfo srcInfo,
            FieldInfos fieldInfos, String vectorField, float shift) throws Exception {
        FieldInfo fi = fieldInfos.fieldInfo(vectorField);
        if (fi == null || !fi.hasVectorValues()) {
            throw new IllegalArgumentException("not a vector field: " + vectorField);
        }
        if (fi.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new IllegalArgumentException("only FLOAT32 supported, got " + fi.getVectorEncoding());
        }
        int dim = fi.getVectorDimension();

        SegmentReadState readState = new SegmentReadState(readDir, srcInfo, fieldInfos, IOContext.READONCE);
        SegmentWriteState writeState = new SegmentWriteState(
                InfoStream.getDefault(), trackingDir, srcInfo, fieldInfos, null, IOContext.DEFAULT);

        // READ via the segment's own codec (resolves the native faiss reader from the .si).
        KnnVectorsReader reader = codec.knnVectorsFormat().fieldsReader(readState);

        // WRITE via a per-field format that forces the native faiss/on_disk SQ format for the vector
        // field. The codec's normal PerField selector needs a live MapperService to pick faiss and
        // otherwise falls back to plain Lucene HNSW. Constructing Faiss1040ScalarQuantizedKnnVectorsFormat
        // directly bypasses the mapper: it derives quantization from the FieldInfo attributes
        // (sq_config / parameters) that the source segment already carries -> byte-faithful on_disk.
        final KnnVectorsFormat faissOnDisk = new Faiss1040ScalarQuantizedKnnVectorsFormat();
        KnnVectorsFormat writeFormat = new PerFieldKnnVectorsFormat() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String f) {
                return faissOnDisk;
            }
        };
        KnnVectorsWriter writer = writeFormat.fieldsWriter(writeState);
        boolean ok = false;
        try {
            @SuppressWarnings("unchecked")
            KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) writer.addField(fi);
            // Enumerate docIDs that have a vector (we DON'T use the old values: replace-all).
            FloatVectorValues values = reader.getFloatVectorValues(vectorField);
            KnnVectorValues.DocIndexIterator it = values.iterator();
            for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                fw.addValue(doc, newVector(doc, dim, shift));
            }
            writer.flush(srcInfo.maxDoc(), null);
            writer.finish();
            ok = true;
        } finally {
            if (ok) {
                reader.close();
                writer.close();
            } else {
                try { reader.close(); } catch (Exception ignore) {}
                try { writer.close(); } catch (Exception ignore) {}
            }
        }
    }

    /** PoC "new embedding model": deterministic, independent of the old value. */
    private static float[] newVector(int docId, int dim, float shift) {
        float[] v = new float[dim];
        for (int d = 0; d < dim; d++) {
            v[d] = (docId + 1) + d + shift;
        }
        return v;
    }

    private static List<String> copyOtherFiles(
            Directory readDir, TrackingDirectoryWrapper trackingDir, SegmentInfo srcInfo) throws Exception {
        List<String> members = srcInfo.getUseCompoundFile()
                ? new ArrayList<>(Arrays.asList(readDir.listAll()))
                : new ArrayList<>(srcInfo.files());
        List<String> copied = new ArrayList<>();
        for (String name : members) {
            if (isVectorFile(name) || name.endsWith(".si") || name.endsWith(".fnm")
                    || name.endsWith(".cfs") || name.endsWith(".cfe")) {
                continue;
            }
            trackingDir.copyFrom(readDir, name, name, IOContext.DEFAULT);
            copied.add(name);
        }
        return copied;
    }

    private static boolean isVectorFile(String f) {
        return f.endsWith(".vec") || f.endsWith(".vem") || f.endsWith(".vemf") || f.endsWith(".veb")
                || f.endsWith(".veq") || f.endsWith(".vemq") || f.endsWith(".vex") || f.endsWith(".faiss")
                || f.contains(".faiss"); // .faissc compound variant
    }

    private static void assertByteIdentical(Directory a, Directory b, List<String> names) throws Exception {
        for (String n : names) {
            long la = a.fileLength(n), lb = b.fileLength(n);
            if (la != lb) {
                throw new AssertionError("length differs for " + n + ": " + la + " vs " + lb);
            }
            try (var ia = a.openInput(n, IOContext.READONCE); var ib = b.openInput(n, IOContext.READONCE)) {
                byte[] ba = new byte[(int) Math.min(la, 1 << 20)];
                byte[] bb = new byte[ba.length];
                long remaining = la;
                while (remaining > 0) {
                    int chunk = (int) Math.min(remaining, ba.length);
                    ia.readBytes(ba, 0, chunk);
                    ib.readBytes(bb, 0, chunk);
                    if (!Arrays.equals(ba, 0, chunk, bb, 0, chunk)) {
                        throw new AssertionError("bytes differ for " + n);
                    }
                    remaining -= chunk;
                }
            }
        }
    }
}
