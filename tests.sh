#!/bin/bash

rm -f *.csv

for i in 5000 8192 16384 2000; do
	echo "Testing MxN: $i x $i, block_size: 16..."
    echo "/*--------------------------------------------*/"
    ./exponentialIntegral.out -n $i -m $i -g > a.out        
    for j in 16 32 64 128 256 1024; do
		echo "Testing MxN: $i x $i, block_size: $j..."
		echo "/*--------------------------------------------*/"
        ./exponentialIntegral.out -n $i -m $i -x $j -y $j -S -c > a.out
        ./exponentialIntegral.out -n $i -m $i -x $j -y $j -l -c > a.out
        for k in 2 3 4; do
            mpirun -n $k ./exponentialIntegral.out -n $i -m $i -x $j -y $j -l -c -p > a.out
            ./exponentialIntegral.out -n $i -m $i -x $j -y $j -l -c -s $k > a.out
        done
    done
done
