#include "phil_math.h"


__host__ __device__
float ed_3D(int p1_x, int p1_y, int p1_z, int p2_x, int p2_y, int p2_z) {
    return sqrtf((p1_x - p2_x) * (p1_x - p2_x) +                        // I believe faster than pow(x, 2)
                     (p1_y - p2_y) * (p1_y - p2_y) +
                     (p1_z - p2_z) * (p1_z - p2_z));
}

__host__ __device__
float euclidean_distance_3D(int p1, int p2, int* vectors) {
    float val = ed_3D(vectors[p1 * 3], vectors[p1 * 3 + 1], vectors[p1 * 3 + 2],
                      vectors[p2 * 3], vectors[p2 * 3 + 1], vectors[p2 * 3 + 1]);
    //std::cout << "distance(" << p1 << ", " << p2 << "): " << val << "\n";
    return val;
}

/*
// rewrite to handle any vector length
// float precision
float ed_2D(int p1_x, int p1_y, int p2_x, int p2_y) {
    return std::sqrt((p1_x - p2_x) * (p1_y - p2_y) + 
                     (p1_x - p2_x) * (p1_y - p2_y));
}

// int precision: bit shifting will take square root & round down. just experimenting.
int ed_2D_int(int p1_x, int p1_y, int p2_x, int p2_y) {
    return ((p1_x - p2_x) * (p1_y - p2_y) + (p2_x - p2_x) * (p2_y - p2_y)) >> 1;
}

float euclidean_distance_2D(int p1, int p2, int* vectors) {
    return ed_2D(vectors[p1 * 2], vectors[p1 * 2 + 1], vectors[p2 * 2], vectors[p2 * 2 + 1]);
}

// to do, and probably replace 2D and 3D versions
float euclidean_distance_ND(int p1, int p2, int* vectors, int dim) {
    return 0.0; // To do
}
*/
