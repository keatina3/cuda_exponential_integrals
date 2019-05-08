#include <vector>
#include <iostream>

template <typename T>
T sse(T* gpu_vals, std::vector<std::vector<T> > &cpu_vals){
    T sse = 0.0;
    for(unsigned int i=0;i<n;i++)
        for(unsigned int j=0;j<numSamples;j++)
            sse += (gpu_vals[j+i*numSamples] - cpu_vals[i][numSamples]) *
                             (gpu_vals[j+i*numSamples] - cpu_vals[i][numSamples]);

    return sse;
}


template <typename T>
void outputResults(T* gpu_vals, std::vector<std::vector<T> > &cpu_vals){
    T x, division=(b-a)/((T)(numSamples));
    
    for(unsigned int i=0;i<n;i++){
        for(unsigned int j=0;j<numSamples;j++){
            x = a + (j+1)*division;
            std::cout << "CPU==>  (" << 
                    i << "," << x <<")=" << cpu_vals[i][j] << " ,";
            std::cout << "GPU++>  (" << 
                    i << "," << x <<")=" << gpu_vals[j+i*numSamples] << std::endl;
        }
    }
}
