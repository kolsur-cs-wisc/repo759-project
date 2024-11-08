#ifndef MATMUL_CUH
#define MATMUL_CUH

__global__ void scaled_matmul(const float *m1, const float *m2, float* out, unsigned int r1, unsigned int c1, unsigned int c2, float factor);
__global__ void scaled_matmul_transposed(const float *m1, const float *m2, float* out, unsigned int r1, unsigned int r2, unsigned int c1, float factor);

#endif