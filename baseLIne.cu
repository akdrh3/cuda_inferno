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

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <filename> <arraysize in millions>\n", argv[0]);
        return -1;
    }

    const char *file_name = argv[1];
    uint64_t input_size = strtoull(argv[2], NULL, 10) * 1000000;
    int *host_a = (int *)malloc(sizeof(int) * input_size);
    int *host_b = (int *)malloc(sizeof(int) * input_size);
    if (host_a == NULL || host_b == NULL)
    {
        printf("Failed to allocate memory for host array\n");
        return -1;
    }
    read_from_file_cpu(file_name, host_a, input_size);
    int *d_a;
    HANDLE_ERROR(cudaMalloc((void **)&d_a, sizeof(int) * input_size));

    uint64_t batch_size = 5 * 1000000000;
    uint64_t numChunks = (input_size + batch_size - 1) / batch_size;
    thrust::device_ptr<int> dev_ptr;

    cudaEvent_t event, start, stop, gpu_start, gpu_stop, dtoh_start, dtoh_stop;
    cudaEventCreate(&event);

    cuda_timer_start(&start, &stop);
    cuda_timer_start(&gpu_start, &gpu_stop);
    for (uint64_t i = 0; i < numChunks; i++)
    {
        uint64_t left_size = (i < numChunks - 1) ? batch_size : (input_size % batch_size);
        uint64_t offset = i * batch_size;

        HANDLE_ERROR(cudaMemcpy(d_a + offset, host_a + offset, left_size * sizeof(int), cudaMemcpyHostToDevice));
        dev_ptr = thrust::device_pointer_cast(d_a + offset);
        thrust::sort(dev_ptr, dev_ptr + left_size);
    }
    double gpu_time = cuda_timer_stop(gpu_start, gpu_stop) / 1000.0;
    cuda_timer_start(&dtoh_start, &dtoh_stop);
    HANDLE_ERROR(cudaMemcpy(host_b, d_a, input_size * sizeof(int), cudaMemcpyDeviceToHost));
    double dtoh_time = cuda_timer_stop(dtoh_start, dtoh_stop) / 1000.0;
    // HANDLE_ERROR(cudaMemcpy(host_b, d_a, input_size * sizeof(int), cudaMemcpyDeviceToHost));
    print_array_host(host_b, 10);
    printf("sorted : %d \n", isRangeSorted_cpu(host_b, 0, batch_size - 1));

    double total_time = cuda_timer_stop(start, stop) / 1000.0;
    printf("Total time: %lf, gpu sort: %lf, dtoh : %lf\n", total_time, gpu_time, dtoh_time);

    free(host_a);
    free(host_b);
    cudaFree(d_a);

    return 0;
}