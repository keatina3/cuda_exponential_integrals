#include <iostream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
#include "utils.h"
#include "ser_expInt.h"

extern void GPUexponentialIntegralFloat(float *results, int block_size_X, int block_size_Y, Tau *tau);
extern void GPUexponentialIntegralFloat_mpi(int argc, char **argv, float *results, int block_size_X, int block_size_Y, Tau *tau);
extern void GPUexponentialIntegralDouble(double *results, int block_size_X, int block_size_Y, Tau *tau);
extern void GPUexponentialIntegralDouble_mpi(int argc, char **argv, double *results, int block_size_X, int block_size_Y, Tau *tau);

int main(int argc, char **argv){
    struct timeval expoStart, expoEnd;
    double timeTotalCpu=0.0;
    double x, division;
    float SSE;
    Tau tau;

    std::vector<std::vector<float> > resultsFloatCpu;
    std::vector<std::vector<double> > resultsDoubleCpu;
    float *resultsFloatGPU;
    double *resultsDoubleGPU;
    
    parseArguments(argc, argv);
   
    if (verbose) {
        std::cout << "n=" << n << std::endl;
        std::cout << "numSamples=" << numSamples << std::endl;
        std::cout << "a=" << a << std::endl;
        std::cout << "b=" << b << std::endl;
        std::cout << "timing=" << timing << std::endl;
        std::cout << "verbose=" << verbose << std::endl;
    }

    // Sanity checks
    if (a>=b) {
        std::cout << "Incorrect interval ("<<a<<","<<b<<") has been stated!" << std::endl;
        return 0;
    }
    if (n<=0) {
        std::cout << "Incorrect orders ("<<n<<") have been stated!" << std::endl;
        return 0;
    }
    if (numSamples<=0) {
        std::cout << "Incorrect number of samples ("<<numSamples<<
                                ") have been stated!" << std::endl;
        return 0;
    }

    try {
        resultsFloatCpu.resize(n,std::vector<float>(numSamples));
    } catch (std::bad_alloc const&) {
        std::cout << "resultsFloatCpu memory allocation fail!" << std::endl;  exit(1);
    }
    try {
        resultsDoubleCpu.resize(n,std::vector<double>(numSamples));
    } catch (std::bad_alloc const&) {
        std::cout << "resultsDoubleCpu memory allocation fail!" << std::endl;  exit(1);
    }
    try {
        resultsFloatGPU = (float*)calloc(n*numSamples,sizeof(float));
    } catch (std::bad_alloc const&) {
        std::cout << "resultsFloatGPU memory allocation fail!" << std::endl;  exit(1);
    }
    try {
        resultsDoubleGPU = (double*)calloc(n*numSamples,sizeof(double));
    } catch (std::bad_alloc const&) {
        std::cout << "resultsFloatGPU memory allocation fail!" << std::endl;  exit(1);
    }

    division=(b-a)/((double)(numSamples));

    if(cpu){
        gettimeofday(&expoStart, NULL);
        for(unsigned int ui=1;ui<=n;ui++){
            for(unsigned int uj=1;uj<=numSamples;uj++){
                x=a+uj*division;
                resultsFloatCpu[ui-1][uj-1]=exponentialIntegralFloat(ui,x);
                resultsDoubleCpu[ui-1][uj-1]=exponentialIntegralDouble(ui,x);
            }
        }
        gettimeofday(&expoEnd, NULL);
        timeTotalCpu=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) 
                            - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
    }
    
    if(gpu){
        if(!mpi){
            GPUexponentialIntegralFloat(resultsFloatGPU,block_size_X,block_size_Y,&tau);
            GPUexponentialIntegralDouble(resultsDoubleGPU,block_size_X,block_size_Y,&tau);
        } else {
            GPUexponentialIntegralFloat_mpi(argc, argv, resultsFloatGPU, block_size_Y, block_size_X,&tau);
            GPUexponentialIntegralDouble_mpi(argc, argv, resultsDoubleGPU, block_size_Y, block_size_X,&tau);
        }
    }
    
    if(timing){
        if(cpu){
            std::cout << "calculating the exponentials on the cpu took: " << 
                    timeTotalCpu<< " seconds" << std::endl;
        }
        if(gpu){
            std::cout << "calculating the exponentials on the gpu took: " <<
                    tau.float_GPU + tau.double_GPU << std::endl;
    
        }
    }

    if(verbose){
        if(cpu){
            //outputResultsCpu(resultsFloatCpu,resultsDoubleCpu);
            //outputResults(resultsFloatGPU,resultsFloatCpu);
            SSE = sse(resultsFloatGPU, resultsFloatCpu);
            std::cout << "Float SSE = " << SSE << std::endl; 
        }
    }

    if(cpu){
        write_times("cpu_times.csv", n, numSamples, shared, numStreams, nprocs, cache, block_size_X, tau, 0);  
    }
    if(gpu && !mpi){
        write_times("gpu_times.csv", n, numSamples, shared, numStreams, nprocs, cache, block_size_X, tau, 1);
    }
    if(gpu && mpi){
        write_times("gpu_times_mpi.csv", n, numSamples, shared, numStreams, nprocs, cache, block_size_X, tau, 1);
    }

    free(resultsFloatGPU); 
    free(resultsDoubleGPU);
    
    return 0;
}
