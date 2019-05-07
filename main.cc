///// Created by Jose Mauricio Refojo - 2014-04-02      Last changed: 2017-04-05
//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <sys/time.h>
#include "utils.h"
#include "ser_expInt.h"

int main(int argc, char **argv){
    unsigned int ui,uj;
    struct timeval expoStart, expoEnd;

    parseArguments(argc, argv);

    if (verbose) {
        std::cout << "n=" << n << std::endl;
        std::cout << "numberOfSamples=" << numberOfSamples << std::endl;
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
    if (numberOfSamples<=0) {
        std::cout << "Incorrect number of samples ("<<numberOfSamples<<
                                ") have been stated!" << std::endl;
        return 0;
    }

    std::vector< std::vector<float>> resultsFloatCpu;
    std::vector< std::vector<double>> resultsDoubleCpu;

    double timeTotalCpu=0.0;

    try {
        resultsFloatCpu.resize(n,std::vector<float>(numberOfSamples));
    } catch (std::bad_alloc const&) {
        std::cout << "resultsFloatCpu memory allocation fail!" << std::endl;  exit(1);
    }
    try {
        resultsDoubleCpu.resize(n,std::vector< double >(numberOfSamples));
    } catch (std::bad_alloc const&) {
        std::cout << "resultsDoubleCpu memory allocation fail!" << std::endl; exit(1);
    }

    double x,division=(b-a)/((double)(numberOfSamples));

    if (cpu) {
        gettimeofday(&expoStart, NULL);
        for (ui=1;ui<=n;ui++) {
            for (uj=1;uj<=numberOfSamples;uj++) {
                x=a+uj*division;
                resultsFloatCpu[ui-1][uj-1]=exponentialIntegralFloat (ui,x);
                resultsDoubleCpu[ui-1][uj-1]=exponentialIntegralDouble (ui,x);
            }
        }
        gettimeofday(&expoEnd, NULL);
        timeTotalCpu=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) 
                            - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
    }

    if (timing) {
        if (cpu) {
            printf ("calculating the exponentials on the cpu took: %f seconds\n",timeTotalCpu);
        }
    }

    if (verbose) {
        if (cpu) {
            outputResultsCpu (resultsFloatCpu,resultsDoubleCpu);
        }
    }
    
    return 0;
}
