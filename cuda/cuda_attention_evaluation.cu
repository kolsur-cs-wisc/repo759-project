#include "attention.cuh"
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <random>

using namespace std;
using chrono::duration;
using chrono::duration_cast;
using chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
  std::random_device entropy_source;
  std::mt19937_64 generator(entropy_source());
  std::uniform_real_distribution<float> uniform_dist(-3.0, 3.0);

  unsigned int B = atoll(argv[1]);
  unsigned int T = atoll(argv[2]);
  unsigned int C = atoll(argv[3]);
  unsigned int NH = atoll(argv[4]);

  float *Q = new float[B * T * C];
  float *K = new float[B * T * C];
  float *V = new float[B * T * C];

  for (int i = 0; i < B * T * C; i++) {
    Q[i] = uniform_dist(generator);
    K[i] = uniform_dist(generator);
    V[i] = uniform_dist(generator);
  }
  int C_per_H = C / NH;
  float *O = new float[B * T * C_per_H * NH];

  float *d_Q, *d_K, *d_V, *d_O;
  cudaMalloc(&d_Q, B * T * C * sizeof(float));
  cudaMalloc(&d_K, B * T * C * sizeof(float));
  cudaMalloc(&d_V, B * T * C * sizeof(float));
  cudaMalloc(&d_O, B * T * C * sizeof(float));

  cudaMemcpy(d_Q, Q, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, K, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

  float time = attention_forward_batched(d_Q, d_K, d_V, d_O, B, T, C, NH);

  std::cout << time << std::endl;

  delete[] Q;
  delete[] K;
  delete[] V;
  delete[] O;

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);

  return 0;
}
