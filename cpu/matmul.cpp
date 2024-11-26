#include "matmul.h"
#include <cstdlib>
#include <iostream>
// #include <omp.h>

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
      out[i * c2 + j] = val * factor;
    }
  }
}

void scaled_matmul_transposed_cpp(const float *m1, const float *m2, float *out,
                                  unsigned int r1, unsigned int r2,
                                  unsigned int c1, float factor) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < r1; i++) {
    for (int j = 0; j < r2; j++) {
      float value = 0.0;
      for (int k = 0; k < c1; k++) {
        value += m1[i * c1 + k] * m2[j * c1 + k];
      }
      out[i * r2 + j] = value * factor;
    }
  }
}

void scaled_matmul_batched_cpp(const float *a, const float *v, float *out,
                               unsigned int B, unsigned int T, unsigned int C,
                               unsigned int NH, float factor) {
  int hs = C / NH;
#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < T; j++) {
      for (int k = 0; k < C; k++) {
        int index = i * T * C + j * C + k;
        float val = 0.0f;
        const float *a_s = a + i * NH * T * T + (k / hs) * T * T + j * T;
        const float *v_s = v + i * T * C;
        for (int z = 0; z < T; z++) {
          val += a[i] * v[i * C + k];
        }
        out[index] = val * factor;
      }
    }
  }
}

void scaled_batched_matmul_transposed_cpp(const float *q, const float *k,
                                          float *out, unsigned int B,
                                          unsigned int T, unsigned int C,
                                          unsigned int NH, float factor) {
  // Q: B T C (NH * CH)
  // K: B T C  (NH * CH)
  // out: B NH T T

  int CH = C / NH;
#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++) {
    for (int nh = 0; nh < NH; nh++) {
      for (int t = 0; t < T; t++) // row
      {
        const float *q_head = q + b * T * C + t * C + nh * CH;
        const float *k_head = k + b * T * C + t * C + nh * CH;
        float *out_head = out + b * NH * T * T + nh * T * T + t * T;
        for (int col = 0; col < T; col++) // col
        {
          float val = 0.0f;
          for (int k = 0; k < T; k++) {
            val += q_head[t + k] * k_head[k + col * T + k];
          }
          out_head[col] = val * factor;
        }
      }
    }
  }
}
