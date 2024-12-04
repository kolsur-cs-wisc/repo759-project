#ifndef ATTENTION_CUH
#define ATTENTION_CUH

__global__ void scaled_matmul(const float *m1, const float *m2, float *out,
                              unsigned int r1, unsigned int c1, unsigned int c2,
                              float factor);

__global__ void scaled_matmul_transposed(const float *m1, const float *m2,
                                         float *out, unsigned int r1,
                                         unsigned int r2, unsigned int c1,
                                         float factor);

__global__ void scaled_batched_matmul(const float *a, const float *v,
                                      float *out, unsigned int B,
                                      unsigned int T, unsigned int C,
                                      unsigned int NH, float factor);
__global__ void scaled_batched_matmul_transposed(const float *q, const float *k,
                                                 float *out, unsigned int B,
                                                 unsigned int T, unsigned int C,
                                                 unsigned int NH, float factor);

__global__ void softmax(float *m, unsigned int R, unsigned int C);

__global__ void softmax_batched(float *a, unsigned int B, unsigned int T,
                                unsigned int NH);

float attention_forward(const float *Q, const float *K, const float *V,
                        float *output, const int L, const int D);

float attention_forward_batched(const float *Q, const float *K, const float *V,
                                float *output, unsigned int B, unsigned int T,
                                unsigned int C, unsigned int NH);

#endif