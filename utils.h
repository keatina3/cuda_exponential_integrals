#ifndef _UTILS_H_
#define _UTILS_H_

extern bool verbose,timing,cpu,gpu,mpi,shared,dynamic;
extern int block_size_X, block_size_Y, numStreams;
extern unsigned int maxIters, n, numSamples;
extern double a,b;     // The interval that we are going to use
extern const double eulerConstant;
extern const double epsilon;
extern const float bigfloat;
extern const double bigDouble;

int parseArguments(int argc, char **argv);
void printUsage();
int findBestDevice();
int decomp1d(int n, int p, int myid, int *s, int *e);

#include "utils.tcc"

template <typename T> T sse(T* gpu_vals, std::vector<std::vector<T> > &cpu_vals);
template <typename T> void outputResults(T* gpu_vals, std::vector<std::vector<T> > &cpu_vals);
template <typename T> void printGrid(T* vals);

#endif
