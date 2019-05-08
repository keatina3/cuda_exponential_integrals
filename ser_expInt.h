#ifndef _SERIAL_EXP_INT_H_
#define _SERIAL_EXP_INT_H_

float   exponentialIntegralFloat(const int n,const float x);
double  exponentialIntegralDouble(const int n,const double x);
void    outputResultsCpu(const std::vector<std::vector<float> > &resultsFloatCpu, 
                const std::vector<std::vector<double> > &resultsDoubleCpu);

#endif
