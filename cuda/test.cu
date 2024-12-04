#include "attention.cu"
// #include "matmul.cuh"
// #include "softmax.cu"
#include <cuda.h>
#include <iostream>

#define L 2
#define D 2

int main() {
  const unsigned int B = 1;  // Batch size
  const unsigned int T = 2;  // Sequence length
  const unsigned int C = 6;  // Feature dimension
  const unsigned int NH = 2; // Number of heads
  const float factor = 1.0f;

  // Hardcoded input values for Q and K
  // Shape: B x NH x T x C = 2 x 2 x 2 x 6
  float h_q[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };

  float h_k[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };

  float h_v[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };

  float h_output1[B * T * T * NH];
  float h_output2[B * T * C];

  // Device matrices
  float *d_Q, *d_K, *d_V, *d_output, *d_final;
  cudaMalloc(&d_Q, B * T * C * sizeof(float));
  cudaMalloc(&d_K, B * T * C * sizeof(float));
  cudaMalloc(&d_V, B * T * C * sizeof(float));
  cudaMalloc(&d_output, B * T * T * NH * sizeof(float));
  cudaMalloc(&d_final, B * T * C * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_Q, h_q, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_k, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_v, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

  attention_forward_batched(d_Q, d_K, d_V, d_final, B, T, C, NH);
  // scaled_batched_matmul_transposed<<<1, B * T * T * NH>>>(d_Q, d_K, d_output,
  // B,
  //                                                         T, C, NH, 1.0);
  // softmax_batched<<<1, B * NH * T>>>(d_output, B, T, NH);
  // scaled_batched_matmul<<<1, B * T * C>>>(d_output, d_V, d_final, B, T, C,
  // NH,
  //                                         1.0);

  // cudaMemcpy(h_output2, d_output, B * T * T * NH * sizeof(float),
  //            cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output2, d_final, B * T * C * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "Result matrix:" << std::endl;
  // for (int i = 0; i < L; ++i)
  // {
  //     for (int j = 0; j < D; ++j)
  //     {
  //         std::cout << h_output[i * D + j] << " ";
  //     }
  //     std::cout << std::endl;
  // }

  // for (int j = 0; j < B * T * T * NH; ++j)
  // {
  //     std::cout << h_output1[j] << " ";
  // }
  // std::cout << std::endl;

  for (int j = 0; j < B * T * C; ++j) {
    std::cout << h_output2[j] << " ";
  }

  // Free device memory
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_output);
  cudaFree(d_final);

  return 0;
}
