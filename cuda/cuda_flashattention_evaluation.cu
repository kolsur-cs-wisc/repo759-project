#include "flash_attention.cu"
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <random>

using namespace std;
using chrono::duration;
using chrono::duration_cast;
using chrono::high_resolution_clock;

int main(int argc, char *argv[])
{
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

   for (int i = 0; i < B * T * C; i++)
   {
      Q[i] = uniform_dist(generator);
      K[i] = uniform_dist(generator);
      V[i] = uniform_dist(generator);
   }
   int C_per_H = C/NH;
   float *O = new float[B * T * C_per_H * NH];
   const float factor = 1 / std::sqrt(C_per_H);

   float time = flash_attention(Q, K, V, O, B, T, C_per_H, NH, factor);

   std::cout << "Time for kernel call " << time << "ms" << std::endl;

   std::cout << std::endl;
   delete[] Q;
   delete[] K;
   delete[] V;
   delete[] O;

   return 0;
}
