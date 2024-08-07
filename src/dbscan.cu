#include <cstdlib>
#include <iostream>
#include <cmath>
#include <queue>

#include "utilities.h"

__host__ __device__ float euclidean_distance(int p1, int p2, int dim, int* vectors) {
    float acc = 0;
    for (int d = 0; d < dim; d++)
        acc += (vectors[p1 * d + d] - vectors[p2 * d + d]) * (vectors[p1 * d + d] - vectors[p2 * d + d]);

    return sqrtf(acc);
}

__host__
void find_neighbors(int point_A, int* vectors, int N, float epsilon, int* output_cluster_IDs, std::queue<int> &neighbors) {
    for (int point_B = 0; point_B < N; point_B++) {
            if (point_A == point_B) continue;
            if (output_cluster_IDs[point_B] != -2) continue;                                           // previously processed
            if (euclidean_distance(point_A, point_B, 3, vectors) < epsilon) neighbors.push(point_B);   // same cluster
        }
    return;
}

// For use by `dbscan_kernel()`. I'm unsure why we'd want to use #define instead of passing `TILE_WIDTH` as a fuction argument,
// but I'm just following PMPP for now.
#define VECS_SIZE 96    // 32 vectors * 3 components

/*
    @param test_output This is used to send data from device to host for inspection.
*/
__global__
void dbscan_kernel(int min_neighbors, float epsilon, int* vectors, const int vector_length, int n, int* d_roots, int* test_output) {
    __shared__ int shared_vectors[VECS_SIZE];

    // This is the collection of "pointers" - root[i] is the root node of i
    int roots[n];
    for (int q = 0; q < n; q++)
        roots[q] = q;
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Collaborative loading of vectors into smem
    for (int a = 0; a < vector_length; a++) 
        shared_vectors[i * vector_length + a] = vectors[i * vector_length + a];
    
    __syncthreads();

    // Each thread will be responsible for `thread_range` = N / num_threads outputs in `roots`
    int thread_range = n / blockDim.x;
    for (int a = 0; a < thread_range; a++) {

        // Queue
        int neighbors[n]; int n_start = 0; int n_end = 0; int n_size = 0;

        int point_A = i * thread_range + a; // The ith thread will process `thread_range` consecutive vectors
        
        // Find neighbors
        for (int point_B = 0; point_B < n; point_B++) {
            if (point_A == point_B) continue;
            if (roots[point_B] != point_B) continue;   // previously processed (starts as self)
            if (euclidean_distance(point_A, point_B, vector_length, vectors) < epsilon) {   // same cluster
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
            int nneighbors[n]; int nn_start = 0; int nn_end = 0; int nn_size = 0;
            
            // Find neighbors
            for (int point_B = 0; point_B < n; point_B++) {
                if (neighbor == point_B) continue;
                if (roots[point_B] != point_B) continue;   // previously processed (starts as self)
                if (euclidean_distance(neighbor, point_B, vector_length, vectors) < epsilon) {   // same cluster
                    nneighbors[n_end] = point_B;
                    nn_size++;
                    nn_end++;
                }
            } 

            // Send to host memory for inspection
            for (int q = 0; q < n; q++) 
                test_output[q] = nneighbors[q];
            
            break;

            // Removing this break will cause things to go haywire,
            // like even when sending the vectors to the GPU and back,
            // under the output "Test Vectors..." they're completely changed,
            // and even have negative numbers.

            if (nn_size >= min_neighbors) {                 // `neighbor` is a core point,
                while (nn_size > 0) {                       // so we can expand its neighbors
                    neighbors[n_end] = nneighbors[nn_start];
                    n_size++;   // neighbors push
                    if (n_size > n){
                        // something's wrong, a visited point shouldn't be able to be added,
                        // so it should never exceed 32, and start and end will never pass each other,
                        // and idk the best way to throw an error or print something in cuda
                    }
                    n_end++;
                    if (n_end == n)
                        n_end = 0;
                    nn_start++; // new neighbors pop
                }
            }
        }
    }

    if (i < n) 
        roots[i] = i + 3;  // just some change to make sure we're modifying them

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

        if (output_cluster_IDs[point_A] != -2) // previously processed
            continue;    

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
            vectors[i * vector_length + j] = rand() % upper_bound;
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

    // Serial code

    std::cout << "Cluster IDs:\n";
    show_numbered(output_cluster_IDs, N);

    dbscan_serial(min_neighbors, epsilon, vectors, vector_length, N, output_cluster_IDs);

    std::cout << "Cluster IDs:\n";
    show_numbered(output_cluster_IDs, N);

    // CUDA MODE

    // Allocate memory on host
    // Note `roots` will sort of serve as cluster IDs
    int h_roots[N];
    size_t roots_size = N * sizeof(int);

    // Allocate memory on the device
    int* d_vectors;
    int* d_roots;
    cudaMalloc(&d_vectors, sizeof(vectors));
    cudaMalloc(&d_roots, cluster_IDs_size);

    // Copy from host memory to device memory
    cudaMemcpy(d_vectors, vectors, vectors_size, cudaMemcpyHostToDevice);

    // Start Test (A) - Meaning relevant snippets are (A)
    int* host_test_ouput = (int*) malloc(N * sizeof(int));
    int* device_test_output;
    cudaMalloc(&device_test_output, N * sizeof(int));
    // End Test (A)

    // Invoke kernel
    dbscan_kernel<<<1, 1>>>(min_neighbors, epsilon, d_vectors, vector_length, N, d_roots, device_test_output);
    cudaDeviceSynchronize();

    // Copy result from device memory to host memory
    cudaMemcpy(h_roots, d_roots, roots_size, cudaMemcpyDeviceToHost);

    // Start Test (A) - load first neighbors queue
    cudaMemcpy(host_test_ouput, device_test_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_test_output);
    std::cout << "Test Device Output:\n";
    show_numbered(host_test_ouput, N);
    // End Test

    // Start Test (B) - load smem vectors (through gmem) into a new array
    int test_vectors[vectors_size];
    cudaMemcpy(test_vectors, d_vectors, vectors_size, cudaMemcpyDeviceToHost);
    std::cout << "Test Vectors (sent to GPU & back):\n";
    show(test_vectors, vector_length, N);
    // End Test

    // Free device memory
    cudaFree(d_vectors);
    cudaFree(d_roots);

    std::cout << "Roots:\n";
    show_numbered(h_roots, N);
    return 0;
}
