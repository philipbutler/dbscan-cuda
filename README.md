Currently this is a C++ implementation of [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN), soon it will be a CUDA implementation of [PDSDBSCAN](https://ieeexplore.ieee.org/document/6468492).

## How to compile & run (after setting up a machine for CUDA development)
From the src directory,
```
# Compile [dbscan.cu, utilities.cu, phil_math.cu] into the binary, dbscan-serial, using the CUDA compiler, nvcc.
nvcc -o dbscan dbscan.cu utilities.cu

# Run the executable
./dbscan
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

For now, this serial version is non-deterministic.
In the parallel version, there will be a dererministic option (by making border points noise), so I'll make the serial one determinitic too, so that the outputs can be compared.

# PDSDBSCAN - Parallel Disjoint-Set DBSCAN
[[PDSDBSCAN](https://ieeexplore.ieee.org/document/6468492) description in my own words]
