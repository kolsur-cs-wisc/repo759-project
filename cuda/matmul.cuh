#ifndef MATMUL_CUH
#define MATMUL_CUH

__global__ void matmul(const float *m1, const float *m2, float* out, unsigned int r1, unsigned int c1, unsigned int c2);

#endif