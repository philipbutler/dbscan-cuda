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

# PDSDBSCAN - Parallel Disjoint-Set DBSCAN
    I plan on 
    - implementing [PDSDBSCAN](https://ieeexplore.ieee.org/document/6468492) in CUDA C++, without any libraries as an exercise
    - updating it using NVIDIA's recommended libraries such as [CCCL](https://github.com/NVIDIA/cccl) where possible
    - running experiements, (different cluster sizes, dimensions, distance metrics, distributions, real world data) and including speed updates, and speed-of-light (SOL) analysis
    - creating animations for 2D and 3D cases, then using this to extrapolate to longer vectors

# Journal
7/22 - have to run to work at Whole Foods, F this