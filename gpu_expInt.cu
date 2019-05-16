#include <stdio.h>
#include <vector>
#include "utils.h"
#include "gpu_expInt.h"

__device__ float calcExp_simple(int n, float x, int maxIters){
    float eulerConstant=0.5772156649015329;
    float epsilon=1.E-30;
    float bigfloat = 3.40282E38;
    float a,b,c,d,del,fact,h,psi,ans=0.0;
    int i,ii;

    //if( n<0.0 || x<0.0 || (fabsf(x)<epsilon && ((n==0) || (n==1)) ) ) {
    //    std::cout << "Bad arguments were passed to the exponentialIntegral function call" << std::endl;
    //    exit(1);
    //}
    if(n==0){
        ans=expf(-x)/x;
    } else {
        if(x>1.0){
            b=(float)n+x;
            c=bigfloat;
            d=1.0/b;
            h=d;
            for(i=1;i<=maxIters;i++){
                a=(float)(-i)*(n-1+i);
                b+=2.0;
                d=1.0/(a*d+b);
                c=b+a/c;
                del=c*d;
                h*=del;
                if(fabsf(del-1.0)<=epsilon){
                    ans=h*expf(-x);
                    return ans;
                }
            }
            ans=h*expf(-x);
            return ans;
        } else { // Evaluate series
            ans=( (n-1) !=0 ? 1.0/(float)(n-1) : -logf(x)-eulerConstant); // First term
            fact=1.0;
            for(i=1;i<=maxIters;i++){
                fact *= -x/(float)i;
                if(i != (n-1)){
                    del = -fact/(float)(i-n+1);
                } else {
                    psi = -eulerConstant;
                    for(ii=1;ii<=(n-1);ii++){
                        psi += 1.0/(float)ii;
                    }
                    del=fact*(-logf(x)+psi);
                }
                ans+=del;
                if(fabsf(del)<fabsf(ans)*epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

__device__ float calcExp_shared(float *consts, int n, float x){
    float a,b,c,d,del,fact,h,psi,ans=0.0;
    int i,ii;

    //if( n<0.0 || x<0.0 || (fabsf(x)<consts[1] && ((n==0) || (n==1)) ) ) {
    //    std::cout << "Bad arguments were passed to the exponentialIntegral function call" << std::endl;
    //    exit(1);
    //}
    if(n==0){
        ans=expf(-x)/x;
    } else {
        if(x>1.0){
            b=(float)n+x;
            c=consts[2];
            d=1.0/b;
            h=d;
            for(i=1;i<=consts[3];i++){
                a=(float)(-i)*(n-1+i);
                b+=2.0;
                d=1.0/(a*d+b);
                c=b+a/c;
                del=c*d;
                h*=del;
                if(fabsf(del-1.0)<=consts[1]){
                    ans=h*expf(-x);
                    return ans;
                }
            }
            ans=h*expf(-x);
            return ans;
        } else { // Evaluate series
            ans=( (n-1) !=0 ? 1.0/(float)(n-1) : -logf(x)-consts[0]); // First term
            fact=1.0;
            for(i=1;i<=consts[3];i++){
                fact *= -x/(float)i;
                if(i != (n-1)){
                    del = -fact/(float)(i-n+1);
                } else {
                    psi = -consts[0];
                    for(ii=1;ii<=(n-1);ii++){
                        psi += 1.0/(float)ii;
                    }
                    del=fact*(-logf(x)+psi);
                }
                ans+=del;
                if(fabsf(del)<fabsf(ans)*consts[1]) return ans;
            }
            return ans;
        }
    }
    return ans;
}

__global__ void calcExpIntegral_shared(float *res_glob, int n, int numSamples, int a, float division, int maxIters){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    extern __shared__ float consts[];
    
    consts[0] = 0.5772156649015329;
    consts[1] = 1.E-30;
    consts[2] = 3.40282E38;
    consts[3] = maxIters;

    float x = a + (idy+1)*division;
    
    if(idx<n && idy < numSamples){
        res_glob[idy + idx*numSamples] = calcExp_shared(consts, idx+1, x); 
    }
}

__global__ void calcExpIntegral_glob(float *res_glob, int n, int numSamples, int a, float division, int maxIters){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    float x = a + (idy+1)*division;
    
    if(idx<n && idy<numSamples){
        res_glob[idy + idx*numSamples] = calcExp_simple(idx+1, x, maxIters);
    }
}

extern void GPUexponentialIntegralFloat(float *results, int block_size_X, int block_size_Y){
    float *res_glob;
    //size_t results_glob_s;
    //int pitch;
    float division = (b-a)/numSamples;
    
    printf("size of n,numsamples = %d,%d\n",n,numSamples);
    cudaMalloc( (void**)&res_glob, n*numSamples*sizeof(float));
    
    dim3 dimBlock(block_size_X, block_size_Y);
    dim3 dimGrid((n/dimBlock.x)+(!(n%dimBlock.x)?0:1),
                (numSamples/dimBlock.y)+(!(numSamples%dimBlock.y)?0:1));
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    //calcExpIntegral_glob<<<dimGrid, dimBlock>>>(res_glob, n, numSamples, a, division, maxIters);
    calcExpIntegral_shared<<<dimGrid, dimBlock, 4*sizeof(float)>>>(res_glob, n, numSamples, a, division, maxIters);
    
    cudaMemcpy(results, res_glob, n*numSamples*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(res_glob);
}
