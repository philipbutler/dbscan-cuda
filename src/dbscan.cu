#include <cstdlib>
#include <iostream>
#include <cmath>
#include <queue>

//#include "phil_math.h"
#include "utilities.h"

/* CircularQueue is a circular buffer, acts as a queue.

Assumes you only ever need `aCapacity` elements, can't store more.

This isn't working, and idk why, so I'm implementing it right in the kernel.

class CircularQueue {
    public:
        int * buffer;
        int start = 0;
        int end = 0;
        int capacity;
        int size = 0;

        __device__ CircularQueue(int aCapacity) {
            buffer = (int*) malloc(capacity * sizeof(int));
            capacity = aCapacity;
        }

        // Add an element to the next open space
        __device__ void push(int x) {
            buffer[(start + end) % capacity] = x;
            size++;
            end++;
            return;
        }

        // Return & remove the next element
        __device__ int pop() {
            int val = buffer[start];
            size--;
            start++;
            return val;
        }

        // Returns whether its empty
        __device__ bool isEmpty() {
            return (size == 0);
        }
};
*/

/* `euclidean_distance_3D` uses the `ed_3D` helper function just for readability.
    Could later compare to something like a 2D array or something else.
    
    Originally I had/have these in a header file, but based on [this](https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/#caveats)
    tells us that we can use the `-dc` flag, but potentially as a performance cost.
    This could be investigated later too.
*/
__host__ __device__ float ed_3D(int p1_x, int p1_y, int p1_z, int p2_x, int p2_y, int p2_z) {
    return sqrtf((p1_x - p2_x) * (p1_x - p2_x) +                        // I believe faster than pow(x, 2)
                     (p1_y - p2_y) * (p1_y - p2_y) +
                     (p1_z - p2_z) * (p1_z - p2_z));
}

// See `ed_3D`
__host__ __device__ float euclidean_distance_3D(int p1, int p2, int* vectors) {
    float val = ed_3D(vectors[p1 * 3], vectors[p1 * 3 + 1], vectors[p1 * 3 + 2],
                      vectors[p2 * 3], vectors[p2 * 3 + 1], vectors[p2 * 3 + 1]);
    //std::cout << "distance(" << p1 << ", " << p2 << "): " << val << "\n";
    return val;
}

__host__
void find_neighbors(int point_A, int* vectors, int N, float epsilon, int* output_cluster_IDs, std::queue<int> &neighbors) {
    for (int point_B = 0; point_B < N; point_B++) {
            if (point_A == point_B) continue;
            if (output_cluster_IDs[point_B] != -2) continue;   // previously processed
            if (euclidean_distance_3D(point_A, point_B, vectors) < epsilon) {   // same cluster
                neighbors.push(point_B);
            }
        }
    return;
}

/*
__device__
void find_neighbors(int point_A, int* vectors, int N, float epsilon, int* output_cluster_IDs, int* neighbors, int* end, int* a_size) {
    for (int point_B = 0; point_B < N; point_B++) {
            if (point_A == point_B) continue;
            if (output_cluster_IDs[point_B] != -2) continue;   // previously processed
            if (euclidean_distance_3D(point_A, point_B, vectors) < epsilon) {   // same cluster
                neighbors[*end] = point_B;
                *a_size++;
                *end++;
            }
        }
    return;
}
*/

// For use by `dbscan_kernel()`. I'm unsure why we'd want to use #define instead of passing `TILE_WIDTH` as a fuction argument,
// but I'm just following PMPP for now.
#define VECS_SIZE 96    // 32 vectors * 3 components

__global__
void dbscan_kernel(int min_neighbors, float epsilon, int* vectors, int vector_length, int N, int* d_roots, int* n1) {
    __shared__ int shared_vectors[VECS_SIZE];

    // This is the collection of "pointers" - root[i] is the root node of i
    int roots[32];
    for (int q = 0; q < 32; q++) {
        roots[q] = q;
    }

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Collaborative loading of vectors into smem
    for (int a = 0; a < 3; a++) {
        shared_vectors[i * vector_length + a] = vectors[i * vector_length + a];
    }
    __syncthreads();

    // Each thread will be responsible for `thread_range` = N / num_threads outputs in `roots`
    int thread_range = N / blockDim.x;
    for (int a = 0; a < thread_range; a++) {

        // Queue
        int neighbors[32]; int n_start = 0; int n_end = 0; int n_size = 0;

        int point_A = i * thread_range + a; // The ith thread will process `thread_range` consecutive vectors
        
        // Find neighbors
        for (int point_B = 0; point_B < N; point_B++) {
            if (point_A == point_B) continue;
            if (roots[point_B] != point_B) continue;   // previously processed (starts as self)
            if (euclidean_distance_3D(point_A, point_B, vectors) < epsilon) {   // same cluster
                neighbors[n_end] = point_B;
                n_size++;
                n_end++;
            }
        }

        while (n_size > 0) {
            int neighbor = neighbors[n_start];
            n_size--;
            if (roots[neighbor] != neighbor) continue;  // previously processed
            
            roots[neighbor] = point_A;                  // label point A as its root

            // Queue neighbors of neighbors
            int nneighbors[32]; int nn_start = 0; int nn_end = 0; int nn_size = 0;
            
            // Find neighbors
            for (int point_B = 0; point_B < N; point_B++) {
                if (neighbor == point_B) continue;
                if (roots[point_B] != point_B) continue;   // previously processed (starts as self)
                if (euclidean_distance_3D(neighbor, point_B, vectors) < epsilon) {   // same cluster
                    nneighbors[n_end] = point_B;
                    nn_size++;
                    nn_end++;
                }
            } 

            // test, delete
            for (int q = 0; q < 32; q++) {
                n1[q] = nneighbors[q];
            }
            break;

            if (nn_size >= min_neighbors) {                 // `neighbor` is a core point,
                while (nn_size > 0) {                       // so we can expand its neighbors
                    neighbors[n_end] = nneighbors[nn_start];
                    n_size++;   // n push
                    nn_start++; // nn pop
                }
            }
        }
        
        
        
    }

    if (i < N)
        roots[i] = i + 3;

    return;
}


void dbscan_serial(int min_neighbors, float epsilon, int* vectors, int vector_length, int N,
                    int* output_cluster_IDs) {
// Refer to README.md

    std::cout << "Running DBSCAN\tepsilon: " << epsilon << "\tmin_neighbors: " << min_neighbors << "\n\n";

    // Noise will be ID'ed as -1.
    // Clusters will be 0+
    int current_cluster_ID = 0;
    for (int point_A = 0; point_A < N; point_A++) {

        if (output_cluster_IDs[point_A] != -2) continue;    // previously processed

        std::queue<int> neighbors;

        find_neighbors(point_A, vectors, N, epsilon, output_cluster_IDs, neighbors);        // updates neighbors queue
        std::cout << "Point " << point_A << " Neighbor list:\n";
        showq(neighbors);
        std::cout << '\n';

        std::cout << "neighbors.size(): " << neighbors.size() << "\n";
        if (neighbors.size() < min_neighbors) {                         // label as noise
            output_cluster_IDs[point_A] = -1;
            continue;
        }

        // Process neighbors
        output_cluster_IDs[point_A] = current_cluster_ID;
        while (!neighbors.empty()) {
            int neighbor = neighbors.front();
            neighbors.pop();

            std::cout << "\ncurrent neighbor: " << neighbor << " cluster ID: " << output_cluster_IDs[neighbor] << "\n"; 
            if (output_cluster_IDs[neighbor] != -2) continue;   // previously processed
            
            output_cluster_IDs[neighbor] = current_cluster_ID; // label as this cluster

            // Queue neighbors of neighbors
            std::queue<int> new_neighbors;
            find_neighbors(neighbor, vectors, N, epsilon, output_cluster_IDs, new_neighbors);   // updates neighbors queue
            if (new_neighbors.size() >= min_neighbors) {                                        // `neighbor` is a core point,
                while (!new_neighbors.empty()) {                                                // so we can expand its neighbors
                    neighbors.push(new_neighbors.front());
                    new_neighbors.pop();
                }
            }
        }

        current_cluster_ID += 1;

        std::cout << "Cluster IDs:\n";
        show_numbered(output_cluster_IDs, N);
    }

}

int main() {

    // `N` vectors with `vector_length` components, totaling `vectors_size` bytes
    int N = 32;
    int vector_length = 3;
    size_t vectors_size = sizeof(int) * N * vector_length;
    int vectors[vectors_size];

    // DBSCAN parameters
    int min_neighbors = 3;
    float epsilon = 50;
    
    // Populate them with random integers within [0, 99], seed for reproducibility
    int upper_bound = 100;
    srand(1);                       // set seed
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < vector_length; j++) {
            vectors[i * 3 + j] = rand() % upper_bound;
        }
    }

    // Print vectors
    std::cout << "Vectors:\n";
    show(vectors, vector_length, N);

    // Points will be labeled with the cluster ID (N points, first value is point ID, second is cluster ID)
    // undefined    -> -2
    // noise        -> -1
    // cluster      ->  0+
    int output_cluster_IDs[N];
    size_t cluster_IDs_size = N * sizeof(int);
    std::fill_n(output_cluster_IDs, N, -2); // fill array with -2

    /* // Serial code

    std::cout << "Cluster IDs:\n";
    show_numbered(output_cluster_IDs, N);

    dbscan_serial(min_neighbors, epsilon, vectors, vector_length, N, output_cluster_IDs);

    std::cout << "Cluster IDs:\n";
    show_numbered(output_cluster_IDs, N);*/

    // Allocate memory on the device
    int* d_vectors;
    int* d_cluster_IDs;
    cudaMalloc(&d_vectors, sizeof(vectors));
    cudaMalloc(&d_cluster_IDs, cluster_IDs_size);

    // Copy from host memory to device memory
    cudaMemcpy(d_vectors, vectors, vectors_size, cudaMemcpyHostToDevice);

    // test - remove
    int* h_n1 = (int*) malloc(32 * sizeof(int));
    int* d_n1;
    cudaMalloc(&d_n1, 32 * sizeof(int));

    // Invoke kernel
    // test todo: remove n1
    dbscan_kernel<<<1, 1>>>(min_neighbors, epsilon, d_vectors, vector_length, N, d_cluster_IDs, d_n1);
    cudaDeviceSynchronize();

    // Copy result from device memory to host memory
    cudaMemcpy(output_cluster_IDs, d_cluster_IDs, cluster_IDs_size, cudaMemcpyDeviceToHost);

    // test, delete
    cudaMemcpy(h_n1, d_n1, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_n1);
    std::cout << "first neighbors:\n";
    show_numbered(h_n1, 32);

    // For testing, load smem vectors (through gmem) into a new array
    int test_vectors[vectors_size];
    cudaMemcpy(test_vectors, d_vectors, vectors_size, cudaMemcpyDeviceToHost);
    std::cout << "Test Vectors (sent to GPU & back):\n";
    show(test_vectors, vector_length, N);

    /* For testing, load first neighbors queue*/

    // Free device memory
    cudaFree(d_vectors);
    cudaFree(d_cluster_IDs);

    std::cout << "Clusters:\n";
    show_numbered(output_cluster_IDs, N);
    return 0;
}
