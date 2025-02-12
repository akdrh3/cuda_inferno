#include "gpu_util.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
extern "C"
{
#include "util.h"
}
#include <cmath>
#include <climits>
#include <stdio.h>

#include <parallel/algorithm>
#include <omp.h>
#include <algorithm>

// filename size workload
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s <filename> <arraysize in millions> <workload on gpu>\n");
        return -1;
    }

    const char *file_name = argv[1];
    uint64_t input_size = strtoull(argv[2], NULL, 10) * 1000000;
    float workload_cpu = atof(argv[3]);

    double *unSorted = NULL;
    HANDLE_ERROR(cudaMallocManaged(&unSorted, input_size));
}