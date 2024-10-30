# Compiler and flags
CC = gcc
CFLAGS = -c -std=c99 -Wall -Wextra -I.
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -c -rdc=true

# Files
DEPS = util.h
CUDEPS = gpu_util.cuh
OBJS = util.o gpu_util.o

# Output
MERGESORT_OUTPUT = mergesort

# Rules
all: $(MERGESORT_OUTPUT)

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu $(CUDEPS)
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

# mergesort.o: mergesort.cu util.h gpu_util.
# 	$(NVCC) $(NVCCFLAGS) 

mergesort: mergesort.o $(OBJS)
	$(NVCC) $(NVCCFLAGS) -lcudart $^ -o $@

clean:
	rm -f *.o $(MERGESORT_OUTPUT) merge_error_log.txt mergeoutput.txt

.PHONY: clean
