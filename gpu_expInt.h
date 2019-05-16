#ifndef _GPU_EXP_INT_H_

#ifdef __cplusplus

extern void GPUexponentialIntegralFloat(float *results, int block_size_X, int block_size_Y);

#endif

#define _GPU_EXP_INT_H_

__device__ float calcExp_simple(int n, float x, int maxIters);
__device__ float calcExp_shared(float *consts, int n, float x);

__global__ void calcExpIntegral_glob(float *res_glob, int n, int numSamples, int a, float division, int maxIters);
__global__ void calcExpIntegral_shared(float *res_glob, int n, int numSamples, int a, float division, int maxIters);

#endif
