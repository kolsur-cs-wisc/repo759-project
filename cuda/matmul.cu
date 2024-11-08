#include "matmul.cuh"
#include <iostream>

#define ROWS_A 2
#define COLS_A 3
#define ROWS_B 3
#define COLS_B 2
#define ROWS_C ROWS_A
#define COLS_C COLS_B

__global__ void matmul(const float *m1, const float *m2, float* out, unsigned int r1, unsigned int c1, unsigned int c2){
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if(r < r1 && c < c2) {
        float value = 0.0;
        for(int i = 0; i < c1; i++){
            value += m1[r * c1 + i] * m2[i * c2 + c];
        }
        out[r * c2 + c] = value;
    }
}

__global__ void matmul_transposed(const float *m1, const float *m2, float* out, unsigned int r1, unsigned int r2, unsigned int c1){
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if(r < r1 && c < r2) {
        float value = 0.0;
        for(int i = 0; i < c1; i++){
            value += m1[r * c1 + i] * m2[c * c1 + i];
        }
        out[r * r2 + c] = value;
    }
}

int main() {
// Host matrices
float h_A[ROWS_A * COLS_A] = {1, 2, 3, 4, 5, 6};  // 2x3 matrix
float h_B[ROWS_B * COLS_B] = {7, 8, 9, 10, 11, 12};  // 3x2 matrix
float h_C[ROWS_C * COLS_C];  // 2x2 matrix for result

// Device matrices
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, ROWS_A * COLS_A * sizeof(float));
cudaMalloc(&d_B, ROWS_B * COLS_B * sizeof(float));
cudaMalloc(&d_C, ROWS_C * COLS_C * sizeof(float));

// Copy data to device
cudaMemcpy(d_A, h_A, ROWS_A * COLS_A * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, ROWS_B * COLS_B * sizeof(float), cudaMemcpyHostToDevice);

// Define block and grid dimensions
dim3 blockDim(16, 16);
dim3 gridDim((ROWS_C + blockDim.x - 1) / blockDim.x, (COLS_C + blockDim.y - 1) / blockDim.y);

// Launch kernel
matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, ROWS_A, COLS_A, COLS_B);

// Copy result back to host
cudaMemcpy(h_C, d_C, ROWS_C * COLS_C * sizeof(float), cudaMemcpyDeviceToHost);

// Print the result
std::cout << "Result matrix C (2x2):" << std::endl;
for (int i = 0; i < ROWS_C; ++i) {
    for (int j = 0; j < COLS_C; ++j) {
        std::cout << h_C[i * COLS_C + j] << " ";
    }
    std::cout << std::endl;
}

// Free device memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

return 0;

}
