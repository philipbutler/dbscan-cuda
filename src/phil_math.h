#ifndef PHIL_MATH_H
#define PHIL_MATH_H

#include <cstdlib>
#include <iostream>

__host__ __device__
float ed_3D(int p1_x, int p1_y, int p1_z, int p2_x, int p2_y, int p2_z);

__host__ __device__
float euclidean_distance_3D(int p1, int p2, int* vectors);

/*
// rewrite to handle any vector length
// float precision
float ed_2D(int p1_x, int p1_y, int p2_x, int p2_y);

// int precision: bit shifting will take square root & round down. just experimenting.
int ed_2D_int(int p1_x, int p1_y, int p2_x, int p2_y);

float euclidean_distance_2D(int p1, int p2, int* vectors);

// to do, and probably replace 2D and 3D versions
float euclidean_distance_ND(int p1, int p2, int* vectors, int dim = 2);
*/

#endif  // PHIL_MATH_H