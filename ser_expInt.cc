#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include "ser_expInt.h"
#include "utils.h"

float exponentialIntegralFloat(const int n, const float x){
    static const float eulerConstant=0.5772156649015329;
    float epsilon=1.E-30;
    float bigfloat=std::numeric_limits<float>::max();
    int i,ii,nm1=n-1;
    float a,b,c,d,del,fact,h,psi,ans=0.0;

    if(n<0.0 || x<0.0 || (x==0.0 && ((n==0) || (n==1)) ) ) {
        std::cout << "Bad arguments were passed to the exponentialIntegral function call" << std::endl;
        exit(1);
    }
    if(n==0){
        ans=exp(-x)/x;
    } else {
        if(x>1.0){
            b=x+n;
            c=bigfloat;
            d=1.0/b;
            h=d;
            for(i=1;i<=(int)maxIters;i++){
                a=-i*(nm1+i);
                b+=2.0;
                d=1.0/(a*d+b);
                c=b+a/c;
                del=c*d;
                h*=del;
                if(fabs(del-1.0)<=epsilon){
                    ans=h*exp(-x);
                    return ans;
                }
            }
            ans=h*exp(-x);
            return ans;
        } else { // Evaluate series
            ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant); // First term
            fact=1.0;
            for(i=1;i<=(int)maxIters;i++){
                fact*=-x/i;
                if(i != nm1){
                    del = -fact/(i-nm1);
                } else {
                    psi = -eulerConstant;
                    for(ii=1;ii<=nm1;ii++){
                        psi += 1.0/ii;
                    }
                    del=fact*(-log(x)+psi);
                }
                ans+=del;
                if(fabs(del)<fabs(ans)*epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

double exponentialIntegralDouble(const int n, const double x) {
    static const double eulerConstant=0.5772156649015329;
    double epsilon=1.E-30;
    double bigDouble=std::numeric_limits<double>::max();
    int i,ii,nm1=n-1;
    double a,b,c,d,del,fact,h,psi,ans=0.0;


    if(n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
        std::cout << "Bad arguments were passed to the exponentialIntegral function call" << std::endl;
        exit(1);
    }
    if(n==0){
        ans=exp(-x)/x;
    } else {
        if(x>1.0){
            b=x+n;
            c=bigDouble;
            d=1.0/b;
            h=d;
            for(i=1;i<=(int)maxIters;i++){
                a=-i*(nm1+i);
                b+=2.0;
                d=1.0/(a*d+b);
                c=b+a/c;
                del=c*d;
                h*=del;
                if (fabs(del-1.0)<=epsilon) {
                    ans=h*exp(-x);
                    return ans;
                }
            }
            //std::cout << "Continued fraction failed in exponentialIntegral" << std::endl;
            return ans;
        } else { // Evaluate series
            ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant); // First term
            fact=1.0;
            for(i=1;i<=(int)maxIters;i++){
                fact*=-x/i;
                if(i != nm1){
                    del = -fact/(i-nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii=1;ii<=nm1;ii++) {
                        psi += 1.0/ii;
                    }
                    del=fact*(-log(x)+psi);
                }
                ans+=del;
                if(fabs(del)<fabs(ans)*epsilon) return ans;
            }
            //std::cout << "Series failed in exponentialIntegral" << std::endl;
            return ans;
        }
    }
    return ans;
}

void outputResultsCpu(const std::vector<std::vector<float> > &resultsFloatCpu, 
            const std::vector<std::vector<double> > &resultsDoubleCpu){
    unsigned int ui,uj;
    double x,division=(b-a)/((double)(numSamples));

    for(ui=1;ui<=n;ui++){
        for(uj=1;uj<=numSamples;uj++){
            x=a+uj*division;
            std::cout << "CPU==> exponentialIntegralDouble (" << 
                    ui << "," << x <<")=" << resultsDoubleCpu[ui-1][uj-1] << " ,";
            std::cout << "exponentialIntegralFloat  (" << 
                    ui << "," << x <<")=" << resultsFloatCpu[ui-1][uj-1] << std::endl;
        }
    }
}
