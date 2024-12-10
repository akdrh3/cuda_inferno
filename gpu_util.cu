#include "gpu_util.cuh"

__host__ void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

__device__ void print_array_device(int *array, int64_t array_size)
{
    for (int64_t i = 0; i < array_size; ++i)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

__host__ void cuda_timer_start(cudaEvent_t *start, cudaEvent_t *stop)
{
    HANDLE_ERROR(cudaEventCreate(start));
    HANDLE_ERROR(cudaEventCreate(stop));
    HANDLE_ERROR(cudaEventRecord(*start, 0));
}

__host__ float cuda_timer_stop(cudaEvent_t start, cudaEvent_t stop)
{
    float time_elpased;
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time_elpased, start, stop));
    return time_elpased;
}

__device__ int isRangeSorted(int *arr, size_t start, size_t end)
{
    if (start >= end) // Invalid range
    {
        return 1; // A single element or empty range is always sorted
    }

    for (size_t i = start + 1; i < end; ++i)
    {
        if (arr[i - 1] > arr[i])
        {
            return 0; // Found an element out of order
        }
    }
    return 1; // The range is sorted
}