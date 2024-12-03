#include "matmul.h"
#include "softmax.h"
#include <cstdlib>
#include <iostream>

#define L 2
#define D 3

int main() {
  // Host matrices
  float h_Q[L * D] = {1, 2, 3, 4, 5, 6};    // 2x3 matrix
  float h_K[D * L] = {7, 8, 9, 10, 11, 12}; // 3x2 matrix
  float h_output[L * L] = {0};
  scaled_matmul_cpp(h_Q, h_K, h_output, L, D, L, 1);

  std::cout << "Result matrix (scaled):" << std::endl;
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      std::cout << h_output[i * L + j] << " ";
    }
    std::cout << std::endl;
  }
  softmax(h_output, L, L);
  std::cout << "Result matrix (softmaxed):" << std::endl;
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      std::cout << h_output[i * L + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
