#include "matmul.cuh"
#include "softmax.cu"

void attention_forward(const float* Q, const float* K, const float* V, float* output, const int L, const int D){
    dim3 blockDim(16, 16);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x, (L + blockDim.y - 1) / blockDim.y);

    float* attn;
    cudaMalloc(&attn, L * L * sizeof(float));

    scaled_matmul_transposed<<<gridDim, blockDim>>>(Q, K, attn, L, L, D, 1/sqrtf(D));
    softmax<<<1, L>>>(attn, L, L);

    dim3 gridDim2((L + blockDim.x - 1) / blockDim.x, (D + blockDim.y - 1) / blockDim.y);
    scaled_matmul<<<gridDim2, blockDim>>>(attn, V, output, L, L, D, 1);

    cudaFree(attn);
}