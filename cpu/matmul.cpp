#include "matmul.h"
#include <cstdlib>
#include <iostream>

void scaled_matmul_cpp(const float *m1, const float *m2, float *out,
                       unsigned int r1, unsigned int c1, unsigned int c2,
                       float factor) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < r1; i++) {
    for (int j = 0; j < c2; j++) {
      float val = 0.0f;
      for (int k = 0; k < c1; k++) {
        val += m1[i * c1 + k] * m2[j + k * c2];
      }
      out[i * c2 + j] = val * factor;1
      
    }
  }
}