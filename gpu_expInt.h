#ifndef _GPU_EXP_INT_H_

#ifdef __cplusplus

extern void GPUexponentialIntegralFloat(float *results, int block_size_X, int block_size_Y);
extern void GPUexponentialIntegralFloat_mpi(int argc, char **argv, float *results, int block_size_X, int block_size_Y);

#endif

#define _GPU_EXP_INT_H_

__device__ float calcExp_simple(int n, float x, int maxIters);
__device__ float calcExp_shared(float *consts, int n, float x);
__device__ float calcExp_dynamic(float *consts, float *dynam_glob, int n, float x);


__global__ void calcExpIntegral_glob(float *res_glob, int n0, int n, 
                        int numSamples, int a, float division, int maxIters);
__global__ void calcExpIntegral_simple(float *res_glob, int n0, int n, 
                        int numSamples, int a, float division, int maxIters);
__global__ void calcExpIntegral_dynamic(float *res_glob, float *dynam_glob, int n, 
                        int numSamples, int a, float division, int maxIters);
__global__ void calcExpIntegral_mpi(float *res_glob, int n0, int n, 
                        int numSamples, int a, float division, int maxIters);


__global__ void calc_series1_dynamic(float *h_glob, int iter, int n, float b, float c, float d);
__global__ void calc_series2_dynamic(float *del_sum, float ans, int iter, int n, int x);

#endif
