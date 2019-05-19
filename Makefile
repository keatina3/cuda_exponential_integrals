#compilers
CC		= mpic++
NVCC 	= nvcc -Xptxas -v


#flags
CXXFLAGS	= -g -W -Wall
#CXXFLAGS	= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
NVCCFLAGS 	= -g -G -Xptxas -v --use_fast_math -arch=sm_35 --relocatable-device-code true
#NVCCFLAGS 	= -Xptxas -v --use_fast_math -arch=sm_35 --relocatable-device-code true
LDFLAGS		= -lmpi

#objects
SOURCES	= *.cc
OBJECTS	= *.o
#CU_OBJECTS = gpu_expInt.o gpu_expInt_dynamic.o gpu_expInt_mpi.o cuda_utils.o
CU_SOURCES = *.cu doubles/*.cu


#includes
MPI_INCLUDES	= /usr/include/openmpi
MPI_LIBS		= /usr/lib/openmpi
INCPATH			= /usr/include/

TARGET	= exponentialIntegral.out


all: cpp_objs cu_objs
	$(NVCC) $(OBJECTS) $(CU_OBJECTS) -o $(TARGET) $(NVCCFLAGS) -L$(MPI_LIBS) $(LDFLAGS) -I$(INCPATH)

cu_objs: $(CU_SOURCES)
	$(NVCC) $(NVCCFLAGS) -c $^ -I$(MPI_INCLUDES) -I$(INCPATH)

cpp_objs: $(SOURCES)
	$(CC) $(CXXFLAGS) -c $^ -I$(MPI_INCLUDES)

.PHONY: clean test mpi-test streams-test dynamic-test

test: all
	./exponentialIntegral.out -n 100 -m 100 -v

mpi-test:
	mpirun -n 4 ./exponentialIntegral.out -n 100 -m 100 -p

streams-test:

dynamic-test:
	
clean:
	$(RM) $(OBJECTS) $(CU_OBJECTS) $(TARGET)
