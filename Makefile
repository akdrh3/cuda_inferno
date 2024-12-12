# Compiler and flags
CC = gcc
CFLAGS = -c -std=c99 -Wall -Wextra -fopenmp

NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -c -rdc=true -Xcompiler -fopenmp

# Files
DEPS = util.h
CUDEPS = gpu_util.cuh
OBJS = util.o gpu_util.o

# Output
MERGESORT_OUTPUT = thrust_merge
BASELINE_OUTPUT = baseline

# CUDA Libraries
CUDA_LIBS = -lcudart -lstdc++ -lgomp

# Rules
all: $(MERGESORT_OUTPUT) $(BASELINE_OUTPUT)

# Rule for C files
%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -o $@ $<

# Rule for CUDA (.cu) files
%.o: %.cu $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

# Rule to create thrust_merge binary
$(MERGESORT_OUTPUT): thrust_merge.o $(OBJS)
	$(NVCC) -o $@ thrust_merge.o $(OBJS) $(CUDA_LIBS) -Xcompiler -fopenmp

# Rule to create baseline binary
$(BASELINE_OUTPUT): baseline.o $(OBJS)
	$(NVCC) -o $@ baseline.o $(OBJS) $(CUDA_LIBS) -Xcompiler -fopenmp

# Clean rule to remove object files and binaries
clean:
	rm -f *.o $(MERGESORT_OUTPUT) $(BASELINE_OUTPUT)

.PHONY: all clean
