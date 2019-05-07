#ifndef _UTILS_H_
#define _UTILS_H_

extern bool verbose,timing,cpu;
extern unsigned int maxIterations, n,numberOfSamples;
extern double a,b;     // The interval that we are going to use

int parseArguments(int argc, char **argv);
void printUsage();

#endif
