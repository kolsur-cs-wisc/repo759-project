#define B 32
__global__ void flash_attention_kernel(const float *Q, const float *K, const float *V, float *O, unsigned int N, unsigned int d, float *l, float *m)
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   float *Kj = K + x * B * d;
   float *Vj = V + x * B * d;
   float *Qi = Q + y * B * d;
   float *Oi = O + y * B * d;
   float *li = l + y * B;
   float *mi = m + y * B;

   extern __shared__ float sram[];
   float *sram_k = sram;
   float *sram_v = sram + B * d;
   float *sram_q = sram_v + B * d;
   float *sram_o = sram_q + B * d;
   float *sram_l = sram_o + B * d;
   float *sram_m = sram_l + B;
   for (int i = 0; i < B * d; i++)
   {
      sram_k[i] = Kj[i];
      sram_v[i] = Vj[i];
      sram_q[i] = Qi[i];
      sram_o[i] = Oi[i];
   }
   for (int i = 0; i < B; i++)
   {
      sram_l[i] = li[i];
      sram_m[i] = mi[i];
   }
}