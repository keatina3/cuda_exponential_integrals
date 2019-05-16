#compilers
CC		= g++
NVCC 	= nvcc


#flags
CXXFLAGS	= -g -pedantic -W -Wall
#CXXFLAGS	= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
NVCCFLAGS = -g -G --use_fast_math
#NVCCFLAGS = --use_fast_math


#objects
SOURCES	= main.cc ser_expInt.cc utils.cc
OBJECTS	= main.o ser_expInt.o utils.o
CU_OBJECTS = gpu_expInt.o
CU_SOURCES = gpu_expInt.cu


TARGET	= exponentialIntegral.out


all: cpp_objs cu_objs
	$(NVCC) $(OBJECTS) $(CU_OBJECTS) -o $(TARGET)

cu_objs: $(CU_SOURCES)
	$(NVCC) $(NVCCFLAGS) -c $^

cpp_objs: $(SOURCES)
	$(CC) $(CXXFLAGS) -c $^

.PHONY: clean test

test: all

clean:
	$(RM) $(OBJECTS) $(CU_OBJECTS) $(TARGET)
