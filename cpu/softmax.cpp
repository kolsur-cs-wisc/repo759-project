#include "softmax.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

void softmax(float *m, unsigned int R, unsigned int C) {
#pragma omp parallel for
  for (int i = 0; i < R; i++) {
    float maxVal = -INFINITY;
    for (int j = 0; j < C; j++) {
      maxVal = std::max(maxVal, m[i * C + j]);
    }
    float denominator = 0.0f;
    for (int j = 0; j < C; j++) {
      m[i * C + j] = exp(m[i * C + j] - maxVal);
      denominator += m[i * C + j];
    }
    for (int j = 0; j < C; j++) {
      m[i * C + j] /= denominator;
    }
  }
}

void softmax_batched(float *m, unsigned int B, unsigned int NH,
                     unsigned int T) {

#pragma omp parallel for collapse(3)
  for (int i = 0; i < B; i++) {
    for (int k = 0; k < NH; k++) {
      for (int z = 0; z < T; z++) {
        float *row = m + i * NH * T + k * T + z;

        float max = -INFINITY;
        for (int j = 0; j < T; j++) {
          if (max < row[j]) {
            max = row[j];
          }
        }

        float denom = 0.0;
        for (int j = 0; j < T; j++) {
          row[j] = expf(row[j] - max);
          denom += row[j];
        }

        for (int j = 0; j < T; j++) {
          row[j] /= denom;
        }
      }
    }
  }
}