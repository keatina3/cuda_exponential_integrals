#include <cstdio>
#include <vector>
#include "../utils.h"
#include "gpu_expInt_double.h"

__device__ double calcExp_simple_double(int n, double x, int maxIters){
    double eulerConstant=0.5772156649015329;
    double epsilon=1.E-30;
    double bigdouble = 3.40282E38;
    double a,b,c,d,del,fact,h,psi,ans=0.0;
    int i,ii;

    if(n==0){
        ans=expf(-x)/x;
    } else {
        if(x>1.0){
            b=(double)n+x;
            c=bigdouble;
            d=1.0/b;
            h=d;
            for(i=1;i<=maxIters;i++){
                a=(double)(-i)*(n-1+i);
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
            ans=( (n-1) !=0 ? 1.0/(double)(n-1) : -logf(x)-eulerConstant); // First term
            fact=1.0;
            for(i=1;i<=maxIters;i++){
                fact *= -x/(double)i;
                if(i != (n-1)){
                    del = -fact/(double)(i-n+1);
                } else {
                    psi = -eulerConstant;
                    for(ii=1;ii<=(n-1);ii++){
                        psi += 1.0/(double)ii;
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

__device__ double calcExp_shared_double(double *consts, int n, double x){
    double a,b,c,d,del,fact,h,psi,ans=0.0;
    int i,ii;

    if(n==0){
        ans=expf(-x)/x;
    } else {
        if(x>1.0){
            b=(double)n+x;
            c=consts[2];
            d=1.0/b;
            h=d;
            for(i=1;i<=consts[3];i++){
                a=(double)(-i)*(n-1+i);
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
            ans=( (n-1) !=0 ? 1.0/(double)(n-1) : -logf(x)-consts[0]); // First term
            fact=1.0;
            for(i=1;i<=consts[3];i++){
                fact *= -x/(double)i;
                if(i != (n-1)){
                    del = -fact/(double)(i-n+1);
                } else {
                    psi = -consts[0];
                    for(ii=1;ii<=(n-1);ii++){
                        psi += 1.0/(double)ii;
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

__global__ void calcExpIntegral_simple_double(double *res_glob, int n, int numSamples, int a, double division, int maxIters){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    double x = a + (idy+1)*division;
    
    if(idx<n && idy<numSamples){
        res_glob[idy + idx*numSamples] = calcExp_simple_double(idx+1, x, maxIters);
    }
}

__global__ void calcExpIntegral_shared_double(double *res_glob, int n, int numSamples, int a, double division, int maxIters){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    extern __shared__ double consts[];
    
    consts[0] = 0.5772156649015329;
    consts[1] = 1.E-30;
    consts[2] = 3.40282E38;
    consts[3] = maxIters;

    double x = a + (idy+1)*division;
    
    if(idx<n && idy < numSamples){
        res_glob[idy + idx*numSamples] = calcExp_shared_double(consts, idx+1, x); 
    }
}

extern void GPUexponentialIntegralDouble(double *results, int block_size_X, int block_size_Y, Tau *tau){
    double *res_glob, *dynam_glob;
    cudaStream_t stream[numStreams];
    int tmp;
    double division = (b-a)/numSamples;
    cudaEvent_t start, finish;
    
    cudaEventCreate(&start); 
    cudaEventCreate(&finish); 
    cudaEventRecord(start);

    cudaMalloc( (void**)&res_glob, n*numSamples*sizeof(double));
    cudaMalloc( (void**)&dynam_glob, n*numSamples*sizeof(double));

    findBestDevice();

    if(numStreams){
        tmp = n - (numStreams-1)*n/numStreams;
        for(int i=0;i<numStreams;i++)
            cudaStreamCreate(&stream[i]);
    }
    
    dim3 dimBlock(block_size_X, block_size_Y);
    dim3 dimGrid((n/dimBlock.x)+(!(n%dimBlock.x)?0:1),
                (numSamples/dimBlock.y)+(!(numSamples%dimBlock.y)?0:1));

    /////////////////////// APPLYING KERNEL ///////////////////////////////
    
    if(shared){
        if(cache) cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        calcExpIntegral_shared_double<<<dimGrid, dimBlock, 4*sizeof(double)>>>
                            (res_glob, n, numSamples, a, division, maxIters);
    } else if(dynamic) {
        calcExpIntegral_dynamic_double<<<dimGrid,dimBlock, 4*sizeof(double)>>>
                            (res_glob, dynam_glob, n, numSamples, a, division, maxIters);        
    } else if(numStreams) {
        dim3 dimGrid((tmp/dimBlock.x)+(!(tmp%dimBlock.x)?0:1),
                    (numSamples/dimBlock.y)+(!(numSamples%dimBlock.y)?0:1));
                    
        for(int i=0;i<numStreams;i++){
            int val = i == (numStreams - 1) ? tmp : n/numStreams;
            calcExpIntegral_portion_double<<<dimGrid,dimBlock,4*sizeof(double),stream[i]>>>
                        (&res_glob[numSamples*i*n/numStreams], i*n/numStreams, 
                         val, numSamples, a, division, maxIters);
        }
        
    } else {
        calcExpIntegral_simple_double<<<dimGrid, dimBlock>>>
                            (res_glob, n, numSamples, a, division, maxIters);
    }
    ////////////////////////////////////////////////////////////////////

    ///////////////// TRANSFERRING RESULTS TO HOST /////////////////////
    
    if(numStreams){
        for(int i=0; i<numStreams;i++){
            int val = i == (numStreams - 1) ? tmp : n/numStreams;
            cudaMemcpyAsync(&results[numSamples*i*n/numStreams],
                    &res_glob[numSamples*i*n/numStreams], numSamples*val*sizeof(double), 
                    cudaMemcpyDeviceToHost, stream[i]);
        }
        
        for(int i=0;i<numStreams;i++)
            cudaStreamDestroy(stream[i]);

    } else {
        cudaMemcpy(results, res_glob, n*numSamples*sizeof(double), cudaMemcpyDeviceToHost);
    }
    //////////////////////////////////////////////////////////////////////
    
    cudaEventRecord(finish);
    cudaEventSynchronize(finish);
    cudaEventElapsedTime((float *)&tau->double_CPU, start, finish);

    cudaFree(res_glob);
}
