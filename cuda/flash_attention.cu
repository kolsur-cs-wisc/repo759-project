#define B 32

__global__ void flash_attention_kernel(const float *Q, const float *K, const float *V, float *O, unsigned int N, unsigned int d, float scaling_factor, float *l, float *m, int NH)
{
   int index = threadIdx.x;
   int batch = blockIdx.x;
   int head = blockIdx.y;
   int qkv_offset = batch * NH * N * d + head * N * d;
   int lm_offset = batch * NH * N + head * N;
   int T = N / B;
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
         for (int c = 0; c < d; c++)
         {
            sram_q[index * d + c] = Q[qkv_offset + i * B * d + index * d + c];
         }
         float* l_i = l + lm_offset;
         float* m_i = m+lm_offset;
         // for()
      }
   }
}