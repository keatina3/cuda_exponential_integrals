#include <mpi.h>
#include "utils.h"
#include "gpu_expInt.h"

__global__ void calcExpIntegral_mpi(float *res_glob, int n0, int n, int numSamples, int a, float division, int maxIters){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    extern __shared__ float consts[];

    consts[0] = 0.5772156649015329;
    consts[1] = 1.E-30;
    consts[2] = 3.40282E38;
    consts[3] = maxIters;

    float x = a + (idy+1)*division;

    if(idx < n && idy < numSamples){
        res_glob[idy + idx*numSamples] = calcExp_shared(consts, n0+idx+1, x);
    }
}

extern void GPUexponentialIntegralFloat_mpi(int argc, char **argv, float *results, int block_size_X, int block_size_Y){
    int myid, nprocs, mycard, num_devices;
    int n_loc, s, e;
    float *res_loc, *res_gpu;
    float division;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Request req[nprocs];
    
    division = (b-a)/numSamples;
    decomp1d(n, nprocs, myid, &s, &e);
    
    n_loc = e-s+1;
 
    num_devices = findBestDevice();

	if (num_devices>1) {
		mycard = myid%num_devices;
		printf("This is process %d, numberOfDevices = %d cardForThisProcess=%d\n", myid, num_devices,mycard);		
		cudaSetDevice(mycard);
	}
	
    res_loc = (float *)malloc(n_loc*numSamples*sizeof(float)); 
	cudaMalloc((void **) &res_gpu, n_loc*numSamples*sizeof(float));
	
	dim3 dimBlock(block_size_X, block_size_Y);
	dim3 dimGrid((n_loc/dimBlock.x) + (!(n_loc%dimBlock.x)?0:1), 
	                    (numSamples/dimBlock.x) + (!(numSamples%dimBlock.x)?0:1));
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    calcExpIntegral_mpi<<<dimGrid,dimBlock,4*sizeof(float)>>>
                    (res_gpu, s, n_loc, numSamples, a, division, maxIters); 
    
    cudaMemcpy(res_loc, res_gpu, n_loc*numSamples*sizeof(float), cudaMemcpyDeviceToHost);
    
    MPI_Isend(res_loc, numSamples*(e-s+1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Wait(&req[0], MPI_STATUS_IGNORE);  
    
    if(myid==0){
        for(int i=0;i<nprocs;i++){
            decomp1d(n, nprocs, i, &s, &e);
            MPI_Irecv(&results[s*numSamples], numSamples*(e-s+1), MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req[i]);
        }
        MPI_Waitall(nprocs, req, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(results, n*numSamples, MPI_FLOAT, 0, MPI_COMM_WORLD);    // couldn't fix make file allow //
                                                              // MPI in C++ files so had to   //
                                                        //send back to all procs to verify results

    cudaFree(res_gpu);
    free(res_loc);

    MPI_Finalize();
}
