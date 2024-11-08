#include "attention.cu"
#include <iostream>

#define L 3
#define D 2

int main() {
    // Host matrices
    float h_Q[L * D] = {1, 2, 3, 4, 5, 6};  // 2x3 matrix
    float h_K[L * D] = {7, 8, 9, 10, 11, 12};  // 2x3 matrix
    float h_V[L * D] = {1, 2, 3, 4, 5, 6};  // 2x3 matrix
    float h_output[L * D];
    
    // Device matrices
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, L * D * sizeof(float));
    cudaMalloc(&d_K, L * D * sizeof(float));
    cudaMalloc(&d_V, L * D * sizeof(float));
    cudaMalloc(&d_output, L * D * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_Q, h_Q, L * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, L * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, L * D * sizeof(float), cudaMemcpyHostToDevice);

    attention_forward(d_Q, d_K, d_V, d_output, L, D);

    cudaMemcpy(h_output, d_output, L * D * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Result matrix:" << std::endl;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < D; ++j) {
            std::cout << h_output[i * D + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    
    return 0;
    
    }
    