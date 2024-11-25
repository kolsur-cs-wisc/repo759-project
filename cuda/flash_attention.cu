#define Br 1
#include <cuda.h>
#include <stdio.h>

__global__ void flash_attention_kernel(const float *Q, const float *K, const float *V, float *O, unsigned int N, unsigned int d, float scaling_factor, float *l, float *m, int NH)
{
   int B = Br;
   int index = threadIdx.x;
   int batch = blockIdx.x;
   int head = blockIdx.y;
   if (index >= N)
      return;
   int qkv_offset = batch * NH * N * d + head * N * d;
   int lm_offset = batch * NH * N + head * N;
   int T = (N + B - 1) / B;
   extern __shared__ float sram[];
   float *sram_k = sram;
   float *sram_v = sram + B * d;
   float *sram_q = sram_v + B * d;
   float *sram_s = sram_q + B * d;
   for (int j = 0; j < T; j++)
   {
      // load Kj and Vj
      for (int i = 0; i < d; i++)
      {
         sram_k[index * d + i] = K[qkv_offset + j * B * d + index * d + i];
         sram_v[index * d + i] = V[qkv_offset + j * B * d + index * d + i];
      }
      __syncthreads();
      for (int i = 0; i < T; i++)
      {
         // load Qi
         for (int c = 0; c < d; c++)
         {
            sram_q[index * d + c] = Q[qkv_offset + i * B * d + index * d + c];
         }
         float l_i = l[lm_offset + B * i + index]; // load li [...B elements] -> get the value corresposing to the thread.
         float m_i = m[lm_offset + B * i + index];
         float m_ij = -INFINITY;

         for (int k = 0; k < B; k++) // for rows of K and index row of Q [each thread works on diff rows of Qi]
         {
            float val = 0.0f;
            for (int c = 0; c < d; c++)
            {
               val += sram_q[index * d + c] * sram_k[k * d + c]; // filling the S[index][k]
            }
            val = val * scaling_factor;
            sram_s[index * B + k] = val;
            m_ij = max(val, m_ij);
         }
         float l_ij = 0.0f;
         for (int c = 0; c < B; c++)
         {
            sram_s[index * B + c] = __expf(sram_s[index * B + c] - m_ij);
            l_ij += sram_s[index * B + c];
         }
         float m_i_new = max(m_i, m_ij);
         float l_i_new = __expf(m_i - m_i_new) * l_i + __expf(m_ij - m_i_new) * l_ij;

         for (int k = 0; k < d; k++) // traversing over diff columns of V to get one row of S * V [different threads calculate diff rows of O]
         {
            float pv = 0.0f;
            for (int c = 0; c < B; c++) // go through elements in the row of Q
            {
               pv += sram_s[index * B + c] * sram_v[k + c * d];
            }
            O[qkv_offset + i * B * d + index * d + k] = (1 / l_i_new) * (l_i * __expf(m_i - m_i_new) * O[qkv_offset + i * B * d + index * d + k] + __expf(m_ij - m_i_new) * pv);
         }
         l[lm_offset + B * i + index] = l_i_new;
         m[lm_offset + B * i + index] = m_i_new;
      }
      __syncthreads();
   }
}