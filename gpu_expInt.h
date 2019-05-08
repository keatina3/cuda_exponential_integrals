#ifndef _GPU_EXP_INT_H_

#ifdef __cplusplus

extern void GPUexponentialIntegralFloat(float *results, int block_size_X, int block_size_Y);

#endif

#define _GPU_EXP_INT_H_

__device__ float calcExp_slow(int n, int x, int maxIters);

__global__ void calcExpIntegralGrid(float *res_glob, int n, int numSamples, int a, float division, int maxIters);

#endif
