#include "matmul.cuh"

__global__ void scaled_matmul(const float *m1, const float *m2, float* out, unsigned int r1, unsigned int c1, unsigned int c2, float factor){
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if(r < r1 && c < c2) {
        float value = 0.0;
        for(int i = 0; i < c1; i++){
            value += m1[r * c1 + i] * m2[i * c2 + c];
        }
        out[r * c2 + c] = value * factor;
    }
}

__global__ void scaled_matmul_transposed(const float *m1, const float *m2, float* out, unsigned int r1, unsigned int r2, unsigned int c1, float factor){
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if(r < r1 && c < r2) {
        float value = 0.0;
        for(int i = 0; i < c1; i++){
            value += m1[r * c1 + i] * m2[c * c1 + i];
        }
        out[r * r2 + c] = value * factor;
    }
}