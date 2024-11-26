#ifndef MATMUL_H
#define MATMUL_H

#include <cstddef>
#include <vector>

void scaled_matmul_cpp(const float *m1, const float *m2, float *out,
                       unsigned int r1, unsigned int c1, unsigned int c2,
                       float factor);

void scaled_matmul_transposed_cpp(const float *m1, const float *m2, float *out,
                                  unsigned int r1, unsigned int r2,
                                  unsigned int c1, float factor);

void scaled_matmul_batched_cpp(const float *a, const float *v, float *out,
                               unsigned int B, unsigned int T, unsigned int C,
                               unsigned int NH, float factor);

void scaled_batched_matmul_transposed_cpp(const float *q, const float *k,
                                          float *out, unsigned int B,
                                          unsigned int T, unsigned int C,
                                          unsigned int NH, float factor);
#endif