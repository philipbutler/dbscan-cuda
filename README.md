# DBSCAN, Abstract algorithm
    DBSCAN clustering is very simple:
    - 2 points belong to the same cluster if their distance is less than `epsilon` (they're neighbors),
        and if one of them has at least `min_neighbors`. This point is a "core point"
    - If a point has no core points as neighbors, then it is "noise"
    - If a point has a neighboring core point, but less than "min_neighbors", it is an "edge point",
        meaning we don't consider its other neighbors as part of the cluster

    The distance metric can be anything we choose. Here we'll use Euclidean distance.

    For now, this will be non-deterministic.
    In the future, clusters that share a border point could become the same cluster.

    This is my implementation of https://en.wikipedia.org/wiki/DBSCAN - "Original query-based algorithm"