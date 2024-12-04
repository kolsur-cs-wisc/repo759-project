#define Br 1
#include <cuda.h>
#include <stdio.h>

__global__ void flash_attention_kernel(const float *Q, const float *K,
													const float *V, float *O, unsigned int N,
													unsigned int d, float scaling_factor,
													float *l, float *m, int NH)
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
			float l_i =
				 l[lm_offset + B * i + index]; // load li [...B elements] -> get the
														 // value corresposing to the thread.
			float m_i = m[lm_offset + B * i + index];
			float m_ij = -INFINITY;

			for (int k = 0; k < B; k++) // for rows of K and index row of Q [each
												 // thread works on diff rows of Qi]
			{
				float val = 0.0f;
				for (int c = 0; c < d; c++)
				{
					val += sram_q[index * d + c] *
							 sram_k[k * d + c]; // filling the S[index][k]
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
			float l_i_new =
				 __expf(m_i - m_i_new) * l_i + __expf(m_ij - m_i_new) * l_ij;

			for (int k = 0; k < d;
				  k++) // traversing over diff columns of V to get one row of S * V
						 // [different threads calculate diff rows of O]
			{
				float pv = 0.0f;
				for (int c = 0; c < B; c++) // go through elements in the row of Q
				{
					pv += sram_s[index * B + c] * sram_v[k + c * d];
				}
				O[qkv_offset + i * B * d + index * d + k] =
					 (1 / l_i_new) * (l_i * __expf(m_i - m_i_new) *
												 O[qkv_offset + i * B * d + index * d + k] +
											__expf(m_ij - m_i_new) * pv);
			}
			l[lm_offset + B * i + index] = l_i_new;
			m[lm_offset + B * i + index] = m_i_new;
		}
		__syncthreads();
	}
}

// Helper function to calculate 1D index for a 4D tensor
inline size_t getIndex(size_t b, size_t t, size_t c, size_t nh, size_t T,
							  size_t C, size_t NH)
{
	return b * T * C * NH + t * C * NH + nh * C + c;
}

inline size_t getIndexOutput(size_t b, size_t t, size_t c, size_t nh, size_t T,
									  size_t C, size_t NH)
{
	// B *NH *T *C
	return b * T * C * NH + nh * T * C + t * C + c;
}

// Function to permute dimensions of the tensor
void permuteTensor(const float *input, float *output, size_t B, size_t T,
						 size_t C, size_t NH)
{
	// Permute from B*T*NH*C to B*NH*T*C
	for (size_t b = 0; b < B; ++b)
	{
		for (size_t nh = 0; nh < NH; ++nh)
		{
			for (size_t t = 0; t < T; ++t)
			{
				for (size_t c = 0; c < C; ++c)
				{
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
						 size_t C, size_t NH)
{
	// Permute from B*NH*T*C to B*T*NH*C
	for (size_t b = 0; b < B; ++b)
	{
		for (size_t t = 0; t < T; ++t)
		{
			for (size_t nh = 0; nh < NH; ++nh)
			{
				for (size_t c = 0; c < C; ++c)
				{
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

float flash_attention(const float *h_q_nonpermuted, const float *h_k_nonpermuted,
							const float *h_v_nonpermuted, float *h_output, unsigned int B, unsigned int T, unsigned int C, int NH, float scaling_factor)
{
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
	
	size_t sram_size = Br * C * 3 * sizeof(float) + Br * Br * sizeof(float);
	// Create events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record start event
	cudaEventRecord(start, 0);

	flash_attention_kernel<<<grid_dim, Br, sram_size>>>(d_Q, d_K, d_V, d_flash, T,
																		 C, scaling_factor, d_l, d_m, NH);

	// Record stop event
	cudaEventRecord(stop, 0);

	// Synchronize
	cudaEventSynchronize(stop);

	// Calculate elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaMemcpy(h_output_flash, d_flash, B * NH * T * C * sizeof(float),
				  cudaMemcpyDeviceToHost);
	permuteOutput(h_output_flash, h_output, B, T, C, NH);
	// Clean up events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_Q);
	cudaFree(d_K);
	cudaFree(d_V);
	cudaFree(d_flash);
	cudaFree(d_l);
	cudaFree(d_m);
	return milliseconds;
}