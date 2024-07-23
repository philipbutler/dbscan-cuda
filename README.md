Currently this is a C++ implementation of [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN), soon it will be a CUDA implementation of [PDSDBSCAN](https://ieeexplore.ieee.org/document/6468492).

## How to compile & run (after setting up a machine for CUDA development)
```
# Compile [dbscan.cu, utilities.cu, phil_math.cu] into the binary, dbscan-serial, using the CUDA compiler, nvcc.
nvcc -o dbscan-serial dbscan.cu utilities.cu phil_math.cu

# Run the executable
./dbscan-serial
```

## To do
- [ ] Finish PDSDBSCAN description in my own words
- [ ] Finish PDSDBCAN CUDA implementation
- [ ] Create animations for 2D and 3D cases, and use this as a vehicle to build intuition for others to extrapolate to longer vectors
- [ ] Add some of those^ & figures to this README, the animations as gifs would be great
- [ ] create visuals for how memory is used
- [ ] create a version using NVIDIA's recommended [CCCL](https://github.com/NVIDIA/cccl) library where possible
- [ ] run experiments, (different cluster sizes, dimensions, distance metrics, distributions, real world data - could start with data from PDSDBSCAN paper) and including speed updates, and speed-of-light (SOL) analysis. 

# DBSCAN, Abstract algorithm
DBSCAN clustering is very simple:
- 2 points belong to the same cluster if their distance is less than `epsilon` (they're neighbors),
    and if one of them has at least `min_neighbors`. This point is a "core point"
- If a point has no core points as neighbors, then it is "noise"
- If a point has a neighboring core point, but less than "min_neighbors", it's called an "edge point" or "boarder point",
    meaning we don't consider its other neighbors as part of the cluster

The distance metric can be anything we choose. Here we'll use Euclidean distance.

For now, this version will be non-deterministic.
In a future version (the parallel version), there should be a deterministic option.
This could be done with some additional condition (like the boarder point will go to the smaller cluster), or clusters that share a border point could become the same cluster.

# PDSDBSCAN - Parallel Disjoint-Set DBSCAN
[[PDSDBSCAN](https://ieeexplore.ieee.org/document/6468492) description in my own words]

# Journal
- 7/23 - Creating slides with the simplest example I can put together to illustrate DSDBSCAN, PDSDBSCAN-S, and PDSDBSCAN-D and corner cases, and I realized I need to write somewhere that, regarding determinism, there should definitely be an option how to handle it. After some thought, the default will be DBSCAN*, that is, sharde boarder point will be noise.
- 7/22 - just getting this WIP up publically, have to run to work at Whole Foods, someone hire me please
