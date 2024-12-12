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
#include <algorithm>

int main(int argc, char *argv[])
{
    // if (argc < 3)
    // {
    //     printf("Usage: %s <filename> <arraysize in millions>\n", argv[0]);
    //     return -1;
    // }

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

    size_t pinned_size = 1000000; // Limited by pinned memory size
    size_t numChunks = (input_size + pinned_size - 1) / pinned_size;

    int *h_aPinned, *h_bPinned, *h_cPinned, *h_dPinned;
    HANDLE_ERROR(cudaMallocHost((void **)&h_aPinned, sizeof(int) * pinned_size));
    HANDLE_ERROR(cudaMallocHost((void **)&h_bPinned, sizeof(int) * pinned_size));
    HANDLE_ERROR(cudaMallocHost((void **)&h_cPinned, sizeof(int) * pinned_size));
    HANDLE_ERROR(cudaMallocHost((void **)&h_dPinned, sizeof(int) * pinned_size));

    int *d_a;
    HANDLE_ERROR(cudaMalloc((void **)&d_a, sizeof(int) * input_size));

    cudaStream_t stream1, stream2, stream3, stream4;
    HANDLE_ERROR(cudaStreamCreate(&stream1));
    HANDLE_ERROR(cudaStreamCreate(&stream2));
    HANDLE_ERROR(cudaStreamCreate(&stream3));
    HANDLE_ERROR(cudaStreamCreate(&stream4));
    thrust::device_ptr<int> dev_ptr;
    cudaEvent_t event;
    cudaEventCreate(&event);

    for (int i = 0; i < numChunks; i++)
    {
        size_t left_size = (i < numChunks - 1) ? pinned_size : (input_size % pinned_size) + pinned_size;
        cudaStream_t currentStream = (i % 2 == 0) ? stream1 : stream2;
        cudaStream_t dtohStream = (i % 2 == 0) ? stream3 : stream4;
        int *currentPinnedMem = (i % 2 == 0) ? h_aPinned : h_bPinned;
        int *writingBackPinnedMem = (i % 2 == 0) ? h_cPinned : h_dPinned;
        size_t offset = i * pinned_size;

        memcpy(currentPinnedMem, host_a + offset, left_size * sizeof(int));
        HANDLE_ERROR(cudaMemcpy(d_a + offset, currentPinnedMem, left_size * sizeof(int), cudaMemcpyHostToDevice));
        dev_ptr = thrust::device_pointer_cast(d_a + offset);
        thrust::sort(thrust::cuda::par.on(currentStream), dev_ptr, dev_ptr + left_size);
        cudaEventRecord(event, currentStream);
        cudaStreamWaitEvent(dtohStream, event);
        HANDLE_ERROR(cudaMemcpyAsync(writingBackPinnedMem, d_a + offset, left_size * sizeof(int), cudaMemcpyDeviceToHost, dtohStream));
        HANDLE_ERROR(cudaStreamSynchronize(dtohStream));
        memcpy(host_b + offset, writingBackPinnedMem, left_size * sizeof(int));
    }
    // HANDLE_ERROR(cudaMemcpy(host_b, d_a, input_size * sizeof(int), cudaMemcpyDeviceToHost));
    print_array_host(host_b, pinned_size - 1);
    printf("sorted : %d \n", isRangeSorted_cpu(host_b, 0, 9));

    __gnu_parallel::sort(host_b, host_b + input_size);

    // dev_ptr = thrust::device_pointer_cast(d_a);
    // thrust::sort(thrust::cuda::par.on(stream1), dev_ptr, dev_ptr + pinned_size);
    // cudaEventRecord(event, stream1);
    // cudaStreamWaitEvent(stream3, event);
    // HANDLE_ERROR(cudaMemcpyAsync(h_cPinned, d_a, pinned_size * sizeof(int), cudaMemcpyDeviceToHost, stream3));
    // HANDLE_ERROR(cudaStreamSynchronize(stream3));
    // memcpy(host_b, h_cPinned, pinned_size * sizeof(int));

    free(host_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    cudaFreeHost(h_cPinned);
    cudaFreeHost(h_dPinned);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaFree(d_a);

    return 0;

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
}
