#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "utils.h"

bool cpu=true, verbose=false, timing=false;
unsigned int maxIters = 2E09, n=10, numSamples=10;
double a=0.0, b=10.0;

int parseArguments (int argc, char **argv) {
    int c; 

    while ((c = getopt (argc, argv, "chi:n:m:a:b:tv")) != -1) {
        switch(c) {
            case 'c':
                cpu=false; break;   //Skip the CPU test
            case 'h':
                printUsage(); exit(0); break;
            case 'i':
                maxIters = atoi(optarg); break;
            case 'n':
                n = atoi(optarg); break;
            case 'm':
                numSamples = atoi(optarg); break;
            case 'a':
                a = atof(optarg); break;
            case 'b':
                b = atof(optarg); break;
            case 't':
                timing = true; break;
            case 'v':
                verbose = true; break;
            default:
                fprintf(stderr, "Invalid option given\n");
                printUsage();
                return -1;
        }
    }
    return 0;
}

void printUsage () {
    printf("exponentialIntegral program\n");
    printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
    printf("This program will calculate a number of exponential integrals\n");
    printf("usage:\n");
    printf("exponentialIntegral.out [options]\n");
    printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
    printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
    printf("      -c           : will skip the CPU test\n");
    printf("      -g           : will skip the GPU test\n");
    printf("      -h           : will show this usage\n");
    printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
    printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
    printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
    printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
    printf("      -v           : will activate the verbose mode  (default: no)\n");
    printf("     \n");
}
