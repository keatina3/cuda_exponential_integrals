#ifndef _GPU_EXP_DOUBLE_H_

#ifdef __cplusplus

extern void GPUexponentialIntegralDouble(double *results, int block_size_X, int block_size_Y, Tau *tau);
extern void GPUexponentialIntegralDouble_mpi(int argc, char **argv, double *results, int block_size_X, int block_size_Y, Tau *tau);

#endif

#define _GPU_EXP_DOUBLE_H_

__device__ double calcExp_simple_double(int n, double x, int maxIters);
__device__ double calcExp_shared_double(double *consts, int n, double x);
__device__ double calcExp_dynamic_double(double *consts, double *dynam_glob, int n, double x);


__global__ void calcExpIntegral_glob_double(double *res_glob, int n, 
                        int numSamples, int a, double division, int maxIters);
__global__ void calcExpIntegral_simple_double(double *res_glob, int n, 
                        int numSamples, int a, double division, int maxIters);
__global__ void calcExpIntegral_dynamic_double(double *res_glob, double *dynam_glob, int n, 
                        int numSamples, int a, double division, int maxIters);
__global__ void calcExpIntegral_portion_double(double *res_glob, int n0, int n, 
                        int numSamples, int a, double division, int maxIters);


__global__ void calc_series1_dynamic_double(double *h_glob, int iter, int n, double b, double c, double d);
__global__ void calc_series2_dynamic_double(double *del_sum, double ans, int iter, int n, int x);

#endif
