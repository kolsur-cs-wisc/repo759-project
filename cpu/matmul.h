#ifndef MATMUL_H
#define MATMUL_H

#include <cstddef>
#include <vector>

void scaled_matmul_cpp(const float *m1, const float *m2, float *out,
                   unsigned int r1, unsigned int c1, unsigned int c2,
                   float factor);

void scaled_matmul_transposed_cpp(const float *m1, const float *m2, float *out,
                              unsigned int r1, unsigned int r2, unsigned int c1,
                              float factor);
#endif