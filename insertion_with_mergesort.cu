#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "gpu_util.cuh"
extern "C"{
#include "util.h"
}
// Device function for insertion sort on a subarray
__device__ void insertionSort(int* arr, int low, int high)
{
    for (int i = low + 1; i <= high; i++)
    {
        int key = arr[i];
        int j = i - 1;

        while (j >= low && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Kernel to call insertion sort on each chunk
__global__ void insertionSortKernel(int* arr, uint64_t chunkSize, uint64_t totalSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = tid * chunkSize;
    uint64_t end = min(start + chunkSize - 1, totalSize - 1);

    if (start < totalSize)
    {
        insertionSort(arr, start, end);
    }
}

// Host function to merge sorted chunks
void merge(int* arr, uint64_t totalSize, uint64_t chunkSize)
{
    int* temp = new int[totalSize];

    for (uint64_t size = chunkSize; size < totalSize; size *= 2)
    {
        for (uint64_t left = 0; left < totalSize - size; left += 2 * size)
        {
            uint64_t mid = min(left + size - 1, totalSize - 1);
            uint64_t right = min(left + 2 * size - 1, totalSize - 1);

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

// Main function to launch insertion sort on CUDA with streams
void parallelSortWithStreams(int* arr, uint64_t totalSize, uint64_t chunkSize, int number_of_streams)
{
    std::vector<cudaStream_t> streams(number_of_streams);

    // Create CUDA streams
    for (int i = 0; i < number_of_streams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Define kernel launch parameters
    int threadsPerBlock = 256;
    int numChunks = (totalSize + chunkSize - 1) / chunkSize;

    for (int i = 0; i < numChunks; i++)
    {
        int streamIdx = i % number_of_streams;
        uint64_t start = i * chunkSize;
        uint64_t end = min(start + chunkSize - 1, totalSize - 1);

        // Launch kernel for each chunk in the corresponding stream
        insertionSortKernel<<<1, threadsPerBlock, 0, streams[streamIdx]>>>(arr, chunkSize, totalSize);
    }

    // Synchronize each stream to ensure sorting is complete
    for (int i = 0; i < number_of_streams; ++i)
    {
        cudaStreamSynchronize(streams[i]);
    }

    // Host-side merge of sorted chunks
    merge(arr, totalSize, chunkSize);

    // Destroy CUDA streams
    for (int i = 0; i < number_of_streams; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }
}

int main(int argc, char *argv[])
{
    // Check command-line arguments for filename and array size
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file_name> <size_of_array_in_millions>" << std::endl;
        return 1;
    }

    const char* file_name = argv[1];
    uint64_t size_of_array = strtoull(argv[2], NULL, 10) * 10;

    // Allocate managed memory for the array on GPU
    int* gpu_array;
    cudaMallocManaged(&gpu_array, size_of_array * sizeof(int));

    // Number of streams to use for concurrent execution
    int number_of_streams = 4;

    // Read data from file (assuming function `read_from_file` is available)
    read_from_file(file_name, gpu_array, size_of_array);

    // Set chunk size based on the number of streams (adjustable)
    uint64_t chunkSize = std::ceil(static_cast<double>(size_of_array) / number_of_streams);

    // Start the sorting process with CUDA streams
    parallelSortWithStreams(gpu_array, size_of_array, chunkSize, number_of_streams);

    // Print a part of the sorted array for verification
    for (int i = 0; i < std::min(size_of_array, uint64_t(10)); i++)
    {
        std::cout << gpu_array[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(gpu_array);

    return 0;
}
