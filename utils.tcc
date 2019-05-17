#include <vector>
#include <math.h>
#include <iostream>

template <typename T>
T sse(T* gpu_vals, std::vector<std::vector<T> > &cpu_vals){
    T sse = 0.0;
    for(unsigned int i=0;i<n;i++)
        for(unsigned int j=0;j<numSamples;j++)
            sse += (gpu_vals[j+i*numSamples] - cpu_vals[i][j]) *
                             (gpu_vals[j+i*numSamples] - cpu_vals[i][j]);

    return sse;
}


template <typename T>
void outputResults(T* gpu_vals, std::vector<std::vector<T> > &cpu_vals){
    T x, division=(b-a)/((T)(numSamples));
    
    for(unsigned int i=0;i<n;i++){
        for(unsigned int j=0;j<numSamples;j++){
            x = a + (j+1)*division;
            if(fabs(cpu_vals[i][j] - gpu_vals[j+i*numSamples]) > 1.E-5){
                std::cout << "CPU==>  (" << 
                        i+1 << "," << x <<")=" << cpu_vals[i][j] << " ,";
                std::cout << "GPU==>  (" << 
                        i+1 << "," << x <<")=" << gpu_vals[j+i*numSamples] << std::endl;
            }
        }
    }
}

template <typename T>
void printGrid(T* vals){
    for(unsigned int i=0;i<n;i++){
        for(unsigned int j=0;j<numSamples;j++){
            std::cout << vals[j + i*numSamples] << " ";;
        }
        std::cout << std::endl;
    }
}
