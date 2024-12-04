// #include "attention.cu"
// #include "matmul.cuh"
#include "flash_attention.cu"
#include <cuda.h>
#include <iostream>

#define L 2
#define D 2
#define Br 1

int main()
{
  const unsigned int B = 1;  // Batch size
  const unsigned int T = 2;  // Sequence length
  const unsigned int C = 3;  // Feature dimension
  const unsigned int NH = 2; // Number of heads
  const float factor = 1.0f;

  // Shape: B x NH x T x C= 2 x 2 x 2 x 6
  float h_q_nonpermuted[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };

  float h_k_nonpermuted[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };

  float h_v_nonpermuted[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };
  float *h_output = (float *)malloc(B * NH * T * C * sizeof(float));
  float time = flash_attention(h_q_nonpermuted, h_k_nonpermuted, h_v_nonpermuted, h_output, B, T, C, NH, factor);
  std::cout << "Time for kernel call " << time << "ms" << std::endl;

  std::cout << "Result matrix:" << std::endl;

  for (int j = 0; j < B * NH * T * C; ++j)
  {
    std::cout << h_output[j] << " ";
  }
  std::cout << std::endl;

  return 0;
}
