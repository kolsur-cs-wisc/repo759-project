#include "attention.cu"
#include <iostream>

#define L 2
#define D 2

int main()
{
    const unsigned int B = 1;  // Batch size
    const unsigned int T = 2;  // Sequence length
    const unsigned int C = 2;  // Feature dimension
    const unsigned int NH = 1; // Number of heads

    // Host matrices
    // float h_Q[L * D] = {1, 2, 3, 4, 5, 6};    // 2x3 matrix
    // float h_K[L * D] = {7, 8, 9, 10, 11, 12}; // 2x3 matrix
    // float h_V[L * D] = {1, 2, 3, 4, 5, 6};    // 2x3 matrix
    float h_output[B * T * T * NH];

    float h_q[] = {
        1.0f, 2.0f, // Q[0][0]
        3.0f, 4.0f  // Q[0][1]
    };

    float h_k[] = {
        5.0f, 6.0f, // K[0][0]
        7.0f, 8.0f  // K[0][1]
    };

    // Device matrices
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, B * T * C * sizeof(float));
    cudaMalloc(&d_K, B * T * C * sizeof(float));
    // cudaMalloc(&d_V, L * D * sizeof(float));
    cudaMalloc(&d_output, B * T * T * NH * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_Q, h_q, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_k, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_V, h_V, L * D * sizeof(float), cudaMemcpyHostToDevice);

    // attention_forward(d_Q, d_K, d_V, d_output, L, D);
    scaled_batched_matmul<<<1, B * T * T * NH>>>(d_Q, d_K, d_output, B, T, C, NH, 1.0);

    cudaMemcpy(h_output, d_output, B * T * T * NH * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix:" << std::endl;
    // for (int i = 0; i < L; ++i)
    // {
    //     for (int j = 0; j < D; ++j)
    //     {
    //         std::cout << h_output[i * D + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    for (int j = 0; j < B * T * T * NH; ++j)
    {
        std::cout << h_output[j] << " ";
    }

    // Free device memory
    cudaFree(d_q);
    cudaFree(d_k);
    // cudaFree(d_V);
    cudaFree(d_output);

    return 0;
}
