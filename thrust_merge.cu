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

    // Allocate host memory
    int *host_a = (int *)malloc(sizeof(int) * input_size);
    int *host_b = (int *)malloc(sizeof(int) * input_size);
    if (host_a == NULL || host_b == NULL)
    {
        printf("Failed to allocate memory for host array\n");
        return -1;
    }

    read_from_file_cpu(file_name, host_a, input_size);

    size_t pinned_size = 1000000; // Limited by pinned memory size
    size_t numChunks = (input_size + pinned_size - 1) / pinned_size;

    int *d_a;
    HANDLE_ERROR(cudaMalloc((void **)&d_a, sizeof(int) * input_size));

    cudaStream_t stream1, stream2;
    HANDLE_ERROR(cudaStreamCreate(&stream1));
    HANDLE_ERROR(cudaStreamCreate(&stream2));

    // Timers
    cudaEvent_t start, stop, gpu_start, gpu_stop, cpu_start, cpu_stop;
    cuda_timer_start(&start, &stop);
    cuda_timer_start(&gpu_start, &gpu_stop);

// Parallel GPU sort
#pragma omp parallel for
    for (int i = 0; i < numChunks; i++)
    {
        size_t left_size = (i < numChunks - 1) ? pinned_size : (input_size % pinned_size == 0) ? pinned_size
                                                                                               : (input_size % pinned_size);
        cudaStream_t currentStream = (i % 2 == 0) ? stream1 : stream2;
        size_t offset = i * pinned_size;

        // Copy chunk from host to device
        HANDLE_ERROR(cudaMemcpyAsync(d_a + offset, host_a + offset, left_size * sizeof(int), cudaMemcpyHostToDevice, currentStream));

        // Sort chunk on device using Thrust
        thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_a + offset);
        thrust::sort(thrust::cuda::par.on(currentStream), dev_ptr, dev_ptr + left_size);

        // Copy sorted chunk back to host
        HANDLE_ERROR(cudaMemcpyAsync(host_b + offset, d_a + offset, left_size * sizeof(int), cudaMemcpyDeviceToHost, currentStream));
    }

    HANDLE_ERROR(cudaStreamSynchronize(stream1));
    HANDLE_ERROR(cudaStreamSynchronize(stream2));

    double gpu_time = cuda_timer_stop(gpu_start, gpu_stop) / 1000.0;

    // Start CPU final sort in parallel with GPU
    cuda_timer_start(&cpu_start, &cpu_stop);

    std::vector<int> merged_result(input_size);
    uint64_t current_offset = 0;

    for (uint64_t i = 0; i < numChunks - 1; i++)
    {
        uint64_t left_size = std::min(pinned_size, input_size - current_offset);
        uint64_t next_offset = current_offset + left_size;
        uint64_t next_size = std::min(pinned_size, input_size - next_offset);

        std::merge(host_b + current_offset,
                   host_b + current_offset + left_size,
                   host_b + next_offset,
                   host_b + next_offset + next_size,
                   merged_result.begin() + current_offset);
        current_offset += left_size;
    }

    double cpu_time = cuda_timer_stop(cpu_start, cpu_stop) / 1000.0;

    double total_time = cuda_timer_stop(start, stop) / 1000.0;

    printf("Total time: %lf, GPU sort: %lf, CPU sort: %lf\n", total_time, gpu_time, cpu_time);
    printf("sorted : %d \n", isRangeSorted_cpu(host_b, 0, input_size - 1));

    free(host_a);
    free(host_b);
    cudaFree(d_a);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}

// // lets assume that input size is 6 million
// memcpy(h_aPinned, host_a, pinned_size * sizeof(int));

// memcpy(h_bPinned, host_a + pinned_size, pinned_size * sizeof(int));
// HANDLE_ERROR(cudaMemcpyAsync(d_a, h_aPinned, pinned_size * sizeof(int), cudaMemcpyHostToDevice, stream1));
// thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_a);
// thrust::sort(thrust::cuda::par.on(stream1), dev_ptr, dev_ptr + pinned_size);
// HANDLE_ERROR(cudaMemcpyAsync(h_cPinned, d_a, pinned_size * sizeof(int), cudaMemcpyDeviceToHost, stream3));

// memcpy(h_aPinned, host_a + pinned_size * 2, pinned_size * sizeof(int));
// HANDLE_ERROR(cudaMemcpyAsync(d_a + pinned_size, h_bPinned, pinned_size * sizeof(int), cudaMemcpyHostToDevice, stream2));
// dev_ptr = thrust::device_pointer_cast(d_a + pinned_size);
// thrust::sort(thrust::cuda::par.on(stream2), dev_ptr, dev_ptr + pinned_size);
// HANDLE_ERROR(cudaMemcpyAsync(h_dPinned, d_a + pinned_size, pinned_size * sizeof(int), cudaMemcpyDeviceToHost, stream4));
// memcpy(host_b, h_cPinned, pinned_size * sizeof(int));

// memcpy(h_bPinned, host_a + pinned_size * 3, pinned_size * sizeof(int));
// HANDLE_ERROR(cudaMemcpyAsync(d_a + pinned_size * 2, h_aPinned, pinned_size * sizeof(int), cudaMemcpyHostToDevice, stream1));
// dev_ptr = thrust::device_pointer_cast(d_a + pinned_size * 2);
// thrust::sort(thrust::cuda::par.on(stream1), dev_ptr, dev_ptr + pinned_size);
// HANDLE_ERROR(cudaMemcpyAsync(h_cPinned, d_a + pinned_size * 2, pinned_size * sizeof(int), cudaMemcpyDeviceToHost, stream3));
// memcpy(host_b + pinned_size, h_dPinned, pinned_size * sizeof(int));

// memcpy(h_aPinned, host_a + pinned_size * 4, pinned_size * sizeof(int));
// HANDLE_ERROR(cudaMemcpyAsync(d_a + pinned_size * 3, h_bPinned, pinned_size * sizeof(int), cudaMemcpyHostToDevice, stream2));
// dev_ptr = thrust::device_pointer_cast(d_a + pinned_size * 3);
// thrust::sort(thrust::cuda::par.on(stream2), dev_ptr, dev_ptr + pinned_size);
// HANDLE_ERROR(cudaMemcpyAsync(h_dPinned, d_a + pinned_size * 3, pinned_size * sizeof(int), cudaMemcpyDeviceToHost, stream4));
// memcpy(host_b + pinned_size * 2, h_cPinned, pinned_size * sizeof(int));

// memcpy(h_bPinned, host_a + pinned_size * 5, pinned_size * sizeof(int));
// HANDLE_ERROR(cudaMemcpyAsync(d_a + pinned_size * 4, h_aPinned, pinned_size * sizeof(int), cudaMemcpyHostToDevice, stream1));
// dev_ptr = thrust::device_pointer_cast(d_a + pinned_size * 4);
// thrust::sort(thrust::cuda::par.on(stream1), dev_ptr, dev_ptr + pinned_size);
// HANDLE_ERROR(cudaMemcpyAsync(h_cPinned, d_a + pinned_size * 4, pinned_size * sizeof(int), cudaMemcpyDeviceToHost, stream3));
// memcpy(host_b + pinned_size * 3, h_dPinned, pinned_size * sizeof(int));

// HANDLE_ERROR(cudaMemcpyAsync(d_a + pinned_size * 5, h_bPinned, pinned_size * sizeof(int), cudaMemcpyHostToDevice, stream2));
// dev_ptr = thrust::device_pointer_cast(d_a + pinned_size * 5);
// thrust::sort(thrust::cuda::par.on(stream2), dev_ptr, dev_ptr + pinned_size);
// HANDLE_ERROR(cudaMemcpyAsync(h_dPinned, d_a + pinned_size * 5, pinned_size * sizeof(int), cudaMemcpyDeviceToHost, stream4));
// memcpy(host_b + pinned_size * 4, h_cPinned, pinned_size * sizeof(int));

// memcpy(host_b + pinned_size * 5, h_dPinned, pinned_size * sizeof(int));
// printf("%d, %d, %d, %d, %d, %d, %d\n", host_a[0], host_a[pinned_size - 1], host_a[pinned_size * 2 - 1], host_a[pinned_size * 3 - 1], host_a[pinned_size * 4 - 1], host_a[pinned_size * 5 - 1], host_a[pinned_size * 6 - 1]);
// printf("%d, %d, %d, %d, %d, %d, %d\n", host_b[0], host_b[pinned_size - 1], host_b[pinned_size * 2 - 1], host_b[pinned_size * 3 - 1], host_b[pinned_size * 4 - 1], host_b[pinned_size * 5 - 1], host_b[pinned_size * 6 - 1]);
