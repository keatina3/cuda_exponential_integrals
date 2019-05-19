#ifndef _UTILS_H_
#define _UTILS_H_

extern bool verbose,timing,cpu,gpu,mpi,shared,dynamic,cache,doubles;
extern int block_size_X, block_size_Y, numStreams, nprocs;
extern unsigned int maxIters, n, numSamples;
extern double a,b;
extern const double eulerConstant;
extern const double epsilon;
extern const float bigfloat;
extern const double bigDouble;

typedef struct Tau {
    float float_CPU;
    double double_CPU;
    float float_GPU;
    double double_GPU;
} tau, *tau_ptr;

int parseArguments(int argc, char **argv);
void printUsage();
int findBestDevice();
int decomp1d(int n, int p, int myid, int *s, int *e);
int is_empty(FILE* file);
void write_times(char* fname, int n, int numSamples, bool shared, int streams, int nprocs, bool cache, int block_size, Tau tau, bool GPU);

#include "utils.tcc"

#endif
