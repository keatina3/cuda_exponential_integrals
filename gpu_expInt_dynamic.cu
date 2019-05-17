#include <stdio.h>
#include <vector>
#include <mpi.h>
#include "utils.h"
#include "gpu_expInt.h"

__device__ float calcExp_dynamic(float *consts, float *dynam_glob, int n, float x){
    float b,c,d,h,ans=0.0;
    float tmp;
    int i, arr_len=512;
    dim3 dimBlock(arr_len);
    dim3 dimGrid((arr_len/dimBlock.x)+(!(arr_len%dimBlock.x)?0:1));

    // NEED TO MAKE CONSTS CONSTANT MEMORY //

    if(n==0){
        ans=expf(-x)/x;
    } else {
        if(x>1.0){
            b=(float)n+x;
            c=consts[2];
            d=1.0/b;
            h=d; 
            for(i=1; i<consts[3]; i+=arr_len){
                calc_series1_dynamic<<<dimGrid,dimBlock,arr_len*sizeof(float)>>>
                                (dynam_glob,i,n,b,c,d);
                tmp = *dynam_glob;
                h *= tmp;
                if(fabs(tmp-1.0)<consts[1])
                    break;
            }
            ans=h*expf(-x);
            return ans;
        } else { // Evaluate series
            ans=( (n-1) !=0 ? 1.0/(float)(n-1) : -logf(x)-consts[0]); // First term
            for(i=1; i<=consts[3]; i+=arr_len){
                calc_series2_dynamic<<<dimGrid,dimBlock,arr_len*sizeof(float)>>>
                                (dynam_glob,ans,i,n,x);
                tmp = *dynam_glob;
                ans += tmp;
                if(fabsf(tmp)<consts[1])
                    break;
            }
            return ans;
        }
    }
    return ans;
}

__global__ void calcExpIntegral_dynamic(float *res_glob, float *dynam_glob, int n, int numSamples, int a, float division, int maxIters){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    extern __shared__ float consts[];
    
    consts[0] = 0.5772156649015329;
    consts[1] = 1.E-30;
    consts[2] = 3.40282E38;
    consts[3] = maxIters;

    float x = a + (idy+1)*division;

    if(idx<n && idy < numSamples){
        res_glob[idy + idx*numSamples] = calcExp_dynamic(consts, 
                                            &dynam_glob[idy + idx*numSamples], n, x);
    }
}

__global__ void calc_series1_dynamic(float *h_glob, int iter, int n, float b, float c, float d){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    float a,del,h,epsilon;
    int i,disp;
    extern __shared__ float del_vals[];
    
    disp = iter + idx;
    epsilon = 1.0E-30;

    if(idx<blockDim.x){
        a = (float)(-disp)*(n-1+1);
        b += disp*2.0;
        d = 1.0/(a*d+b);
        c = b+a/c;
        del=c*d;
        del_vals[idx] = del;
        __syncthreads();
        
        if(fabsf(del_vals[0] - 1.0) < epsilon){
            h = 1.0;
            *h_glob = h;
            return;
        }

        i = blockDim.x;     // won't work for non powers of 2 //
        for( ; i>1;i>>=1){
            if(idx<(i/2)){
                del_vals[idx] *= del_vals[idx+(i/2)];
            }
            __syncthreads();
        }
        if(idx==0){
            h = d*del_vals[0];
            *h_glob = h;
        }
    }
}

__global__ void calc_series2_dynamic(float *del_sum, float ans, int iter, int n, int x){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i;
    float fact, epsilon, psi, eulerConstant;
    extern __shared__ float del_vals[];
    
    epsilon = 1.0E-30;
    eulerConstant = 0.5772156649015329;
    
    if(idx<blockDim.x){
        fact = powf(-x/(float)iter, iter);
        if(iter != (n-1)){
            del_vals[idx] = -fact/(float)(iter-n+1);
        } else {
            psi = -eulerConstant;
            for(i=1;i<(n-1);i++){
                psi += 1.0/(float)i;
            }
            del_vals[idx] = fact*(-logf(x)+psi);
        }
        __syncthreads();

        if(fabsf(del_vals[0]<fabsf(ans)*epsilon)){
                *del_sum = 0.0;
                return;
        }
        
        i = blockDim.x;     // won't work for non powers of 2 //
        for( ; i>1;i>>=1){
            if(idx<(i/2)){
                del_vals[idx] += del_vals[idx+(i/2)];
            }
            __syncthreads();
        }
        if(idx==0){
            *del_sum = del_vals[0];
        }
    }
}
