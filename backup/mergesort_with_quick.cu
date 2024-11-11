#include <cuda_runtime.h>
#include <algorithm> // for std::swap on host
#include <iostream>
#include "gpu_util.cuh"
extern "C"{
#include "util.h"
}

__device__ void quicksort(int* arr, uint64_t low, uint64_t high)
{
    // Device-side quicksort for each thread's portion
    if (low < high)
    {
        uint64_t pivot = arr[high];
        uint64_t i = low - 1;

        for (uint64_t j = low; j < high; j++)
        {
            if (arr[j] < pivot)
            {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        
        uint64_t pivotIndex = i + 1;

        quicksort(arr, low, pivotIndex - 1);
        quicksort(arr, pivotIndex + 1, high);
    }
}

__global__ void quicksortKernel(int* arr, uint64_t chunkSize, uint64_t totalSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = tid * chunkSize;
    uint64_t end = min(start + chunkSize - 1, totalSize - 1);

    quicksort(arr, start, end);
}

void merge(int* arr, uint64_t totalSize, uint64_t chunkSize)
{
    // Host-side merge for sorted chunks
    // Use iterative merging to reduce memory usage
    int* temp = new int[totalSize];

    for (uint64_t size = chunkSize; size < totalSize; size *= 2)
    {
        for (uint64_t left = 0; left < totalSize - size; left += 2 * size)
        {
            uint64_t mid = min(left + size - 1, totalSize - 1);
            uint64_t right = min(left + 2 * size - 1, totalSize - 1);

            // Standard merge between two sorted subarrays
            uint64_t i = left, j = mid + 1, k = left;
            while (i <= mid && j <= right)
            {
                temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
            }
            while (i <= mid) temp[k++] = arr[i++];
            while (j <= right) temp[k++] = arr[j++];
        }
        std::copy(temp, temp + totalSize, arr); // Copy back to arr
    }
    delete[] temp;
}

void parallelSort(int* arr, uint64_t totalSize, uint64_t chunkSize, int number_of_thread)
{
    quicksortKernel<<<1, number_of_thread>>>(arr, chunkSize, totalSize);
    HANDLE_ERROR(cudaDeviceSynchronize());
    merge(arr, totalSize, chunkSize);

}

int main(int argc, char *argv[])
{
    //get param from command; filename , arraysize * 1 million
    const char *file_name = argv[1];
    uint64_t size_of_array = strtoull(argv[2], NULL, 10)*100;
    
    int *gpu_array = NULL;
    HANDLE_ERROR(cudaMallocManaged((void**)&gpu_array, size_of_array * sizeof(int)));
    cudaEvent_t start, stop;

    //different thread number
    // int thread_numbers[5] = {1, 256, 512, 768, 1024};
    int thread_numbers[5] = {1, 2, 3, 4, 5};

    double array_size_in_GB = SIZE_IN_GB(sizeof(int)*size_of_array);
    printf("Data Set Size: %f GB Number of integers : %lu\n", array_size_in_GB, size_of_array);

    for (int i = 0; i < 5; ++i){
        int number_of_thread = thread_numbers[i];
        uint64_t segment_size = (uint64_t)ceil((double)size_of_array / number_of_thread);

        //read from file and store it to gpu_array
        read_from_file(file_name, gpu_array, size_of_array);
        cuda_timer_start(&start, &stop);
        parallelSort(gpu_array, size_of_array, segment_size, number_of_thread);

        // Stop timer
        double gpu_sort_time = cuda_timer_stop(start, stop);
        double gpu_sort_time_sec = gpu_sort_time / 1000.0;

        printf("Time elapsed for merge sort with %d threads: %lf s\n", number_of_thread, gpu_sort_time_sec);
        print_array_host(gpu_array, size_of_array);
        // print_array_host(gpu_tmp, size_of_array);
        printf("-------------------------------------------------\n");
    }
    //free pointers
    HANDLE_ERROR(cudaFree(gpu_array));      

    return 0;
}
