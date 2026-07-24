package org.opensearch.knn.index.query;

public class KNNQueryResult {
    private final int id;
    private final float score;

    public KNNQueryResult(int id, float score) {
        this.id = id;
        this.score = score;
    }

    public int getId() { return id; }
    public float getScore() { return score; }
}
