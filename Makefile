#compilers
CC		= g++
NVCC 	= nvcc


#flags
CFLAGS	= -g -pedantic -W -Wall
#CFLAGS	= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
NVCCFLAGS = -g -G --use_fast_math


#objects
OBJECTS	= main.o ser_expInt.o utils.o
CU_OBJECTS = gpu_expInt.o
CU_SOURCES = gpu_expInt.cu


TARGET	= exponentialIntegral.out


all: $(OBJECTS) cu_objs
	$(NVCC) $(OBJECTS) $(CU_OBJECTS) -o $(TARGET)

cu_objs: $(CU_SOURCES)
	$(NVCC) $^ -c $(NVCCFLAGS)


.PHONY: clean test

test: all

clean:
	$(RM) $(OBJECTS) $(TARGET)
