# Files and dependencies
DEPS = util.h 
CUDEPS = gpu_util.cuh
OBJS = util.o gpu_util.o workload.o 

# Compiler settings
CC=gcc
NVCC=/usr/local/cuda-12.4/bin/nvcc 

# Compiler flags
CFLAGS=-O3 -std=c11
NVCCFLAGS=-O3 -arch=sm_50 -std=c++14 -Xcompiler -fopenmp -D_GLIBCXX_PARALLEL -I/usr/local/cuda-12.4/include -G

# Output executable name
EXE=workload

# Main target: make workload
workload: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(EXE) $^ -lpthread -g

# Compile CUDA source files
workload.o: workload.cu $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -g

# Compile C source files
util.o: util.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@ -g

# Compile other CUDA source files as needed
gpu_util.o: gpu_util.cu $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -g

# Clean up compiled files
clean:
	rm -f $(EXE) $(OBJS)

