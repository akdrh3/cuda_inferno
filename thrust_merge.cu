#include "gpu_util.cuh"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
extern "C"
{
#include "util.h"
}
#include <cmath>
#include <climits>
#include <stdio.h>

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
    if (host_a == NULL)
    {
        printf("Failed to allocate memory for host array\n");
        return -1;
    }
    read_from_file_cpu(file_name, host_a, input_size);

    size_t pinned_size = 1000000; // Limited by pinned memory size
    size_t numChunks = (input_size + pinned_size - 1) / pinned_size;

    int *h_aPinned, *h_bPinned;
    HANDLE_ERROR(cudaMallocHost((void **)&h_aPinned, sizeof(int) * pinned_size));
    HANDLE_ERROR(cudaMallocHost((void **)&h_bPinned, sizeof(int) * pinned_size));

    int *d_a;
    HANDLE_ERROR(cudaMalloc((void **)&d_a, sizeof(int) * input_size));

    cudaStream_t stream1, stream2;
    HANDLE_ERROR(cudaStreamCreate(&stream1));
    HANDLE_ERROR(cudaStreamCreate(&stream2));

    for (size_t i = 0; i < numChunks; i++)
    {
        size_t current_size = (i < numChunks - 1) ? pinned_size : (input_size % pinned_size);
        size_t bytes = current_size * sizeof(int);
        size_t offset = i * pinned_size;

        int *host_a_src = host_a + offset;
        int *host_a_dst = (i % 2 == 0) ? h_aPinned : h_bPinned;
        int *d_dst = d_a + offset;
        cudaStream_t stream_used = (i % 2 == 0) ? stream2 : stream1;

        memcpy(host_a_dst, host_a_src, bytes);
        printf("%d\n", *host_a_dst);

        HANDLE_ERROR(cudaMemcpyAsync(d_dst, host_a_dst, bytes, cudaMemcpyHostToDevice, stream_used));
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    free(host_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_a);

    return 0;
}
