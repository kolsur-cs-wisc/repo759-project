#include "attention.cuh"
#include <cuda.h>

float attention_forward(const float *Q, const float *K, const float *V,
                        float *output, const int L, const int D) {
  dim3 blockDim(16, 16);
  dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
               (L + blockDim.y - 1) / blockDim.y);

  float *attn;
  cudaMalloc(&attn, L * L * sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  scaled_matmul_transposed<<<gridDim, blockDim>>>(Q, K, attn, L, L, D,
                                                  1 / sqrtf(D));
  softmax<<<1, L>>>(attn, L, L);

  dim3 gridDim2((L + blockDim.x - 1) / blockDim.x,
                (D + blockDim.y - 1) / blockDim.y);
  scaled_matmul<<<gridDim2, blockDim>>>(attn, V, output, L, L, D, 1);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaFree(attn);

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  return millis;
}

float attention_forward_batched(const float *Q, const float *K, const float *V,
                                float *output, unsigned int B, unsigned int T,
                                unsigned int C, unsigned int NH) {
  float *attention_scores;
  unsigned int threads_per_block = 256;
  unsigned int total_elements_scores = B * T * T * NH;
  unsigned int softmax_rows = B * NH * T;
  unsigned int total_elements_output = B * T * C;

  cudaMalloc(&attention_scores, B * T * T * NH * sizeof(float));
  int dk = C / NH;
  float factor = 1 / std::sqrt(dk);

  unsigned int num_blocks_1 =
      ((total_elements_scores + threads_per_block - 1) / threads_per_block);
  unsigned int num_blocks_2 =
      ((softmax_rows + threads_per_block - 1) / threads_per_block);
  unsigned int num_blocks_3 =
      ((total_elements_output + threads_per_block - 1) / threads_per_block);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  scaled_batched_matmul_transposed<<<num_blocks_1, threads_per_block>>>(
      Q, K, attention_scores, B, T, C, NH, factor);

  softmax_batched<<<num_blocks_2, threads_per_block>>>(attention_scores, B, T, NH);

  scaled_batched_matmul<<<num_blocks_3, threads_per_block>>>(
      attention_scores, V, output, B, T, C, NH, 1.0);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaFree(attention_scores);

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  return millis;
}