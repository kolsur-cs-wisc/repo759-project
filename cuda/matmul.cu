#include "matmul.cuh"

__global__ void scaled_matmul(const float *m1, const float *m2, float *out, unsigned int r1, unsigned int c1, unsigned int c2, float factor)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < r1 && c < c2)
    {
        float value = 0.0;
        for (int i = 0; i < c1; i++)
        {
            value += m1[r * c1 + i] * m2[i * c2 + c];
        }
        out[r * c2 + c] = value * factor;
    }
}

__global__ void scaled_matmul_transposed(const float *m1, const float *m2, float *out, unsigned int r1, unsigned int r2, unsigned int c1, float factor)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < r1 && c < r2)
    {
        float value = 0.0;
        for (int i = 0; i < c1; i++)
        {
            value += m1[r * c1 + i] * m2[c * c1 + i];
        }
        out[r * r2 + c] = value * factor;
    }
}

__global__ void scaled_batched_matmul(const float *q, const float *k, float *out, unsigned int B, unsigned int T, unsigned int C, unsigned int NH, float factor)
{
    // sz of output is B*NH*T*T
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= B * NH * T * T)
        return;
    int batch = index / (NH * T * T);
    int rem1 = index % (NH * T * T);
    int head = rem1 / (T * T);
    int rem2 = rem1 % (T * T);
    int row = rem2 / T;
    int col = rem2 % T;
    int hs = C / NH;
    float val = 0.0;
    const float *q_start = q + batch * T * C + head * hs + row * NH * hs;
    const float *k_start = k + batch * T * C + head * hs + col * NH * hs;
    for (int c = 0; c < hs; c++)
    {
        val += q_start[c] * k_start[c];
    }
    out[index] = val;
}