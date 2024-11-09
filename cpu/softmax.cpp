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