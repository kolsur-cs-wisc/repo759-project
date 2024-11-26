#include "attention.h"
#include <cmath>
#include <iostream>

void attention_forward_cpp(const float *Q, const float *K, const float *V,
                           float *output, unsigned int B, unsigned int T,
                           unsigned int C, unsigned int NH) {
  float *attention_scores = new float[B * T * T * NH];
  int dk = C / NH;
  float factor = 1 / std::sqrt(dk);

  scaled_batched_matmul_transposed_cpp(Q, K, attention_scores, B, T, C, NH,
                                       factor);
  softmax_batched(attention_scores, B, NH, T);
  scaled_matmul_batched_cpp(attention_scores, V, output, B, T, C, NH, factor);
}