# Compiler and flags
CC = gcc
CFLAGS = -c -std=c99 -Wall -Wextra -I.
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -c -rdc=true

# Files
C_FILES = util.c
CUDA_FILES = gpu_util.cu mergesort.cu
DEPS = util.h
CUDEPS = gpu_util.h

# Objects
C_OBJS = util.o
CUDA_OBJS = gpu_util.o
OBJS = util.o gpu_util.o
# Output
MERGESORT_OUTPUT = mergesort

# Rules
all: $(MERGESORT_OUTPUT)

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $< 

mergesort: mergesort.o $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ -lcudart

clean:
	rm -f $(OBJS) $(MERGESORT_OUTPUT) merge_error_log.txt mergeoutput.txt

.PHONY: clean
