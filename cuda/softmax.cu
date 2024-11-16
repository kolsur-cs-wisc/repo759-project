#include <math.h>

__global__ void softmax(float *m, unsigned int R, unsigned int C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < R) {
        float* row = m + i * C;

        float max = -INFINITY;
        for(int j = 0; j < C; j++){
            if(max < row[j]){
                max = row[j];
            }
        }

        float denom = 0.0;
        for(int j = 0; j < C; j++){
            row[j] = expf(row[j] - max);
            denom += row[j];
        }

        for(int j = 0; j < C; j++){
            row[j] /= denom;
        }
    }
}

__global__ void softmax_batched(float *a, unsigned int B, unsigned int T, unsigned int NH){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < B * NH * T) {
        float* row = a + i * T;

        float max = -INFINITY;
        for(int j = 0; j < T; j++){
            if(max < row[j]){
                max = row[j];
            }
        }

        float denom = 0.0;
        for(int j = 0; j < T; j++){
            row[j] = expf(row[j] - max);
            denom += row[j];
        }

        for(int j = 0; j < T; j++){
            row[j] /= denom;
        }
    }
}