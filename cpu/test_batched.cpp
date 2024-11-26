// #include "attention.cu"
#include "attention.h"
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

  float h_output_1[] = {14, 32, 32, 77, 194, 266, 266, 365};

  float h_output1[B * T * T * NH];
  float h_output2[B * T * C];

  attention_forward_cpp(h_q, h_k, h_v, h_output2, B, T, C, NH);
  // scaled_batched_matmul_transposed_cpp(h_q, h_k, h_output1, B, T, C, NH,
  //                                      factor);
  // softmax_batched(h_output1, B, NH, T);
  // scaled_matmul_batched_cpp(h_output1, h_v, h_output2, B, T, C, NH, factor);

  std::cout << "Result matrix 2:" << std::endl;
  for (int j = 0; j < B * T * C; ++j) {
    std::cout << h_output2[j] << " ";
  }

  return 0;
}
