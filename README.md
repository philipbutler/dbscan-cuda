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

# Journal
- 8/10 - Time moves way too fast, how has it been 8 days... well I worked 5 out of the last 7 days at Whole Foods, went to a baby shower, spent some time modifying our budget, ran >30 miles, and spent a _fair_ amount of time with my wife so I guess that explains it.
    - First thing on my mind is that compiling yields an error that a constant value is needed for an array, but `const int N = n;` is not doing the trick. First I'm going to see what it was previously. I think I just used `32`, but of course we want it to be any number of vectors. Yes, they all had 32.
    - I either need to figure out the best way to use a variable number of vectors, or don't and just use a specific number, which is what I'm so in the habit of not doing lol. It seems like the value is needed at compile-time, not runtime. Need to investigate.
- 8/2 - Need test set up, probably can't get the chance to touch anything today 
- 7/31 - Kind of left it looking closer to spaghetti code, started getting more disorganized in the interest of moving faster. Did get to the point that the `new_neighbors` queue is holding the correct elements after 1 iteration. It's kinda bed time so I think all I wanna do tn is make a commit with this functioning now (just did), and now go back and do some clean up, and then push that cleaner version. Yeah, cleaned it up. Where I'm leaving it is much like I left it, but I did test if it worked right away after removing the `break`, and it does not work. It doesn't work so bad that somehow even after sending the vectors to the GPU and back, without modifying them, changes the result. So I suppose that means some poor memory management is happening or something? Anyways that's not for tonight for me to find out. Shout out to Hulu for not working today so I got to have this nice little session!
- 7/30 - Writing the code as I make the diagrams so I can validate each. I'm realizing I need to do more manually regarding device memory management. I'm using [this](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory) from the CUDA programming guide as my primary resource. I've just "staged" the kernel by having the kernel threads write to device global memory, and copy it back to host memory, without using the `__managed__` keyword or any other magic that I'm certain exists.
    - omg i can't use `std::queue` from a `__device__` function so now I'm implementing a queue. good cpp learning opportunity lol
    - cuda is hard😂 this is kinda hard even before cuda. idk if I'm tired or confusing myself, ~~but I'm not sure why I used the same queue in the serial version when processing neighbors of neighbors.~~ (7/31 edit: I was tired, they always were separate queues. The first queue `neighbors` has candidates for neighbors to expand, the 2nd queue `new_neighbors` has neighbors of a neighbor, and may add one of those candidates to `neighbors`. `neighbors` is never meant to hold a point that represents a neighbor of the point currently being expanded, only points that will be expanded.) I have to wrap this up for the day, but I hate leaving it in such an inconsistent state.
    - To save the context on my mind, I need to:
    - [ ] ~~Test my CircularQueue implementation, (and change the name to that), unless I come across something better to use~~ not using this anymore
    - [ ] ~~Would be nice if there was 1 implementation, but currently I separated the host and device `find_neighbors()` implementations to support my queue for the device~~ not using this for device code anymore, but honestly I should probably do some necessary play with pointers to get this working how I expect
    - [x] Can handle any vector length / number of components now
    - [ ] in the future, write code to be more extensible (more vectors, ~~more components~~, flexible to use on different devices, etc)
- 7/28 - Just centralizing my notes here. Reviewing my slides, I think I should do away with the coordinate plane once starting to illustrate how the parallel version workss
- 7/25 - Plan for today is to actually start writing the parallel verison's code because there's some nuances I'm trying to sort out while making these slides/diagrams. Firstly, I think I need to realize that I first need a baseline parallel implementation that starts with some assumptions.
    - The first assumption is that all P points can fit in smem, and cannot fit in rmem.
    - Also, since theres N cores working with their own copy of the data, I need to decide if P/N points needs to fit in smem, or if there can be coordination s. t. only few copies are needed to exist in smem (each copy private to each thread). The simplest approach is to first assume that P/N points can fit in smem.
    - Actually let me first run a kernel at all that doesn't do anything except print the thread index into an output array.
- 7/24 - Nothing today. Had the first solid workout I've done in a while, did literal housekeeping, then had to go to work.
- 7/23 - Creating slides with the simplest example I can put together to illustrate DSDBSCAN, PDSDBSCAN-S, and PDSDBSCAN-D and corner cases, and I realized I need to write somewhere that, regarding determinism, there should definitely be an option how to handle it. After some thought, the default will be DBSCAN*, that is, sharde boarder point will be noise.
- 7/22 - just getting this WIP up publically, have to run to work at Whole Foods, someone hire me please
