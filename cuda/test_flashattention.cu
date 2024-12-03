// #include "attention.cu"
// #include "matmul.cuh"
#include "flash_attention.cu"
#include <cuda.h>
#include <iostream>

#define L 2
#define D 2
#define Br 1

// Helper function to calculate 1D index for a 4D tensor
inline size_t getIndex(size_t b, size_t t, size_t c, size_t nh, size_t T,
                       size_t C, size_t NH) {
  return b * T * C * NH + t * C * NH + nh * C + c;
}

inline size_t getIndexOutput(size_t b, size_t t, size_t c, size_t nh, size_t T,
                             size_t C, size_t NH) {
  // B *NH *T *C
  return b * T * C * NH + nh * T * C + t * C + c;
}

// Function to permute dimensions of the tensor
void permuteTensor(const float *input, float *output, size_t B, size_t T,
                   size_t C, size_t NH) {
  // Permute from B*T*NH*C to B*NH*T*C
  for (size_t b = 0; b < B; ++b) {
    for (size_t nh = 0; nh < NH; ++nh) {
      for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < C; ++c) {
          // Compute indices for input and output
          size_t inputIdx = getIndex(b, t, c, nh, T, C, NH);
          size_t outputIdx = b * NH * T * C + nh * T * C + t * C + c;

          // Assign value to permuted output
          // std::cout << "Input Index = " << inputIdx << " : " << outputIdx <<
          // std::endl;
          output[outputIdx] = input[inputIdx];
        }
      }
    }
  }
}

// Function to permute dimensions of the tensor
void permuteOutput(const float *input, float *output, size_t B, size_t T,
                   size_t C, size_t NH) {
  // Permute from B*NH*T*C to B*T*NH*C
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t nh = 0; nh < NH; ++nh) {
        for (size_t c = 0; c < C; ++c) {
          // Compute indices for input and output
          size_t inputIdx = getIndexOutput(b, t, c, nh, T, C, NH);
          size_t outputIdx = b * NH * T * C + t * NH * C + nh * C + c;

          // Assign value to permuted output
          // std::cout << "Input Index = " << inputIdx << " : " << outputIdx <<
          // std::endl;
          output[outputIdx] = input[inputIdx];
        }
      }
    }
  }
}
int main() {
  const unsigned int B = 1;  // Batch size
  const unsigned int T = 2;  // Sequence length
  const unsigned int C = 3;  // Feature dimension
  const unsigned int NH = 2; // Number of heads
  const float factor = 1.0f;

  // Shape: B x NH x T x C= 2 x 2 x 2 x 6
  float h_q_nonpermuted[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };

  float h_k_nonpermuted[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };

  float h_v_nonpermuted[] = {
      // Batch 0, Head 0
      1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12,
      // Batch 0, Head 1
      // 13, 14, 15, 19, 20, 21, 16, 17, 18, 22, 23, 24,
  };
  float *h_q = (float *)malloc(B * NH * T * C * sizeof(float));
  float *h_k = (float *)malloc(B * NH * T * C * sizeof(float));
  float *h_v = (float *)malloc(B * NH * T * C * sizeof(float));

  permuteTensor(h_q_nonpermuted, h_q, B, T, C, NH);
  permuteTensor(h_k_nonpermuted, h_k, B, T, C, NH);
  permuteTensor(h_v_nonpermuted, h_v, B, T, C, NH);

  float h_output_flash[B * NH * T * C];

  float h_m[] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};

  // Device matrices
  float *d_Q, *d_K, *d_V, *d_flash, *d_l, *d_m;
  cudaMalloc(&d_Q, B * NH * T * C * sizeof(float));
  cudaMalloc(&d_K, B * NH * T * C * sizeof(float));
  cudaMalloc(&d_V, B * NH * T * C * sizeof(float));
  cudaMalloc(&d_flash, B * NH * T * C * sizeof(float));
  cudaMalloc(&d_l, B * NH * T * sizeof(float));
  cudaMalloc(&d_m, B * NH * T * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_Q, h_q, B * NH * T * C * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_k, B * NH * T * C * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_v, B * NH * T * C * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, h_m, B * NH * T * sizeof(float), cudaMemcpyHostToDevice);
  dim3 grid_dim(B, NH);
  // flash_attention_kernel(const float *Q, const float *K, const float *V,
  // float *O, unsigned int N, unsigned int d, float scaling_factor, float *l,
  // float *m, int NH);
  size_t sram_size = Br * C * 3 * sizeof(float) + Br * Br * sizeof(float);
  flash_attention_kernel<<<grid_dim, Br, sram_size>>>(d_Q, d_K, d_V, d_flash, T,
                                                      C, factor, d_l, d_m, NH);
  cudaMemcpy(h_output_flash, d_flash, B * NH * T * C * sizeof(float),
             cudaMemcpyDeviceToHost);

  float *h_output = (float *)malloc(B * NH * T * C * sizeof(float));
  permuteOutput(h_output_flash, h_output, B, T, C, NH);
  std::cout << "Result matrix:" << std::endl;
  // for (int i = 0; i < L; ++i)
  // {
  //     for (int j = 0; j < D; ++j)
  //     {
  //         std::cout << h_output[i * D + j] << " ";
  //     }
  //     std::cout << std::endl;
  // }

  for (int j = 0; j < B * NH * T * C; ++j) {
    std::cout << h_output[j] << " ";
  }
  std::cout << std::endl;

  // Free device memory
  // cudaFree(d_Q);
  // cudaFree(d_K);
  // cudaFree(d_V);
  // cudaFree(d_output);
  // cudaFree(d_final);

  return 0;
}
