#include <cstdlib>
#include <iostream>
#include <cmath>
#include <queue>

#include "phil_math.h"
#include "utilities.h"

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

    // `N` vectors with `vector_length` components
    int N = 32;
    int vector_length = 3;
    int vectors[N * vector_length];

    // DBSCAN parameters
    int min_neighbors = 3;
    float epsilon = 30;
    
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
    std::fill_n(output_cluster_IDs, N, -2); // fill array with -2

    std::cout << "Cluster IDs:\n";
    show_numbered(output_cluster_IDs, N);

    dbscan_serial(min_neighbors, epsilon, vectors, vector_length, N, output_cluster_IDs);

    std::cout << "Cluster IDs:\n";
    show_numbered(output_cluster_IDs, N);

    //add<<<32, 1024>>>(vector_a, vector_b, vector_c, length);
    //cudaDeviceSynchronize();
    return 0;
}
