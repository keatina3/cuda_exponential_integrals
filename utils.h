#ifndef _UTILS_H_
#define _UTILS_H_

extern bool verbose,timing,cpu;
extern unsigned int maxIters, n, numSamples;
extern double a,b;     // The interval that we are going to use

int parseArguments(int argc, char **argv);
void printUsage();

#include "utils.tcc"

template <typename T> T sse(T* gpu_vals, std::vector<std::vector<T> > &cpu_vals);
template <typename T> void outputResults(T* gpu_vals, std::vector<std::vector<T> > &cpu_vals);


#endif
