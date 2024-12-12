# Compiler and flags
CC = gcc
CFLAGS = -c -std=c99 -Wall -Wextra -I -fopenmp
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -c -rdc=true -Xcompiler -fopenmp -fgomp

# Files
DEPS = util.h
CUDEPS = gpu_util.cuh
OBJS = util.o gpu_util.o

# Output
MERGESORT_OUTPUT = thrust_merge

# Rules
all: $(MERGESORT_OUTPUT)

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

thrust_merge: thrust_merge.o $(OBJS)
	$(NVCC) -o thrust_merge thrust_merge.o util.o gpu_util.o -lcudart -lstdc++

clean:
	rm -f *.o $(MERGESORT_OUTPUT)

.PHONY: clean
