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
#pragma omp parallel for
  for (int index = 0; index < B * T * C; index++) {
    int batch = index / (T * C);
    int rem = index % (T * C);
    int row = rem / C;
    int col = rem % C;
    int hs = C / NH;
    float val = 0.0f;
    const float *a_s = a + batch * NH * T * T + (col / hs) * T * T + row * T;
    const float *v_s = v + batch * T * C;
    for (int z = 0; z < T; z++) {
      val += a_s[z] * v_s[z * C + col];
    }
    out[index] = val * factor;
  }
}

void scaled_batched_matmul_transposed_cpp(const float *q, const float *k,
                                          float *out, unsigned int B,
                                          unsigned int T, unsigned int C,
                                          unsigned int NH, float factor) {
  // Q: B T C (NH * CH)
  // K: B T C  (NH * CH)
  // out: B NH T T
#pragma omp parallel for
  for (int index = 0; index < B * T * T * NH; index++) {
    int batch = index / (NH * T * T);
    int rem1 = index % (NH * T * T);
    int head = rem1 / (T * T);
    int rem2 = rem1 % (T * T);
    int row = rem2 / T;
    int col = rem2 % T;
    int hs = C / NH;
    float val = 0.0;

    const float *q_start = q + batch * T * C + head * hs + row * NH * hs;
    const float *k_start = k + batch * T * C + head * hs + col * NH * hs;

    for (int c = 0; c < hs; c++) {
      val += q_start[c] * k_start[c];
    }

    out[index] = val * factor;
  }
}
