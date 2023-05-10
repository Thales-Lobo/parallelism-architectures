#####################################################################################
## CUDA + OpenMP Matrix product lab
## Makefile by S. Vialle, November 2021
#####################################################################################

CPUCC = g++
GPUCC = /usr/local/cuda/bin/nvcc

#CUDA_TARGET_FLAGS = -arch=sm_61      #GTX 1080 on Cameron cluster
#CUDA_TARGET_FLAGS = -arch=sm_75      #RTX 2080-Ti on Tx cluster

CXXFLAGS = #-DDP
CXXFLAGS += -I/usr/local/cuda/include/ -I/usr/include/x86_64-linux-gnu/
CC_CXXFLAGS = -Ofast -fopenmp 
CUDA_CXXFLAGS = -O3 $(CUDA_TARGET_FLAGS)

CC_LDFLAGS =  -fopenmp -L/usr/local/x86_64-linux-gnu
CUDA_LDFLAGS = -L/usr/local/cuda/lib64/ 

CC_LIBS = -lopenblas64 
CUDA_LIBS = -lcudart -lcuda -lcublas

CC_SOURCES =  main.cc init.cc  
CUDA_SOURCES = gpu.cu 
CC_OBJECTS = $(CC_SOURCES:%.cc=%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=%.o)

EXECNAME = MatrixProduct


all:
	$(CPUCC) -c $(CC_SOURCES) $(CXXFLAGS) $(CC_CXXFLAGS)
	$(GPUCC) -c $(CUDA_SOURCES) $(CXXFLAGS) $(CUDA_CXXFLAGS)
	$(CPUCC) -o $(EXECNAME) $(CC_LDFLAGS) $(CUDA_LDFLAGS) $(CC_OBJECTS) $(CUDA_OBJECTS) $(CUDA_LIBS) $(CC_LIBS)


clean:
	rm -f *.o $(EXECNAME) *.linkinfo *~ *.bak .depend


#Regles automatiques pour les objets
#%.o:  %.cc
#	$(CPUCC)  -c  $(CXXFLAGS) $(CC_CXXFLAGS) $<
#
#%.o:  %.cu
#	$(GPUCC)  -c  $(CXXFLAGS) $(CUDA_CXXFLAGS) $<

