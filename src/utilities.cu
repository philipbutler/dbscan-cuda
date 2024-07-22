#include "utilities.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <queue>

void show(int* vectors, int vector_length, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < vector_length; j++) {
            std::cout << vectors[i * vector_length + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << '\n';
}

// Print the queue (makes a copy)
void showq(std::queue<int> gq) {
    std::queue<int> g = gq;
    while (!g.empty()) {
        std::cout << '\t' << g.front();
        g.pop();
    }
    std::cout << '\n';
}

void show_numbered(int* vectors, int N) {
    for (int i = 0; i < N; i++) {
        std::cout << i << " " << vectors[i] << "\n";
    }
    std::cout << '\n';
}