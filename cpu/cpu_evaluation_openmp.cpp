#include "attention.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>

using namespace std;
using chrono::duration;
using chrono::duration_cast;
using chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
  std::random_device entropy_source;
  std::mt19937_64 generator(entropy_source());
  std::uniform_real_distribution<float> uniform_dist(-3.0, 3.0);

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, milli> duration_milli;

  unsigned int B = atoll(argv[1]);
  unsigned int T = atoll(argv[2]);
  unsigned int C = atoll(argv[3]);
  unsigned int NH = atoll(argv[4]);
  const int t = atoi(argv[5]);

  omp_set_num_threads(t);

  float *Q = new float[B * T * C];
  float *K = new float[B * T * C];
  float *V = new float[B * T * C];

  for (int i = 0; i < B * T * C; i++) {
    Q[i] = uniform_dist(generator);
    K[i] = uniform_dist(generator);
    V[i] = uniform_dist(generator);
  }

  float *O = new float[B * T * C];

  start = high_resolution_clock::now();
  attention_forward_cpp(Q, K, V, O, B, T, C, NH);
  end = high_resolution_clock::now();

  duration_milli = duration_cast<duration<double, milli>>(end - start);
  cout << duration_milli.count() << endl;

  delete[] Q;
  delete[] K;
  delete[] V;
  delete[] O;

  return 0;
}