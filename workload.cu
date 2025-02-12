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

struct SortingInfo
{
    double dataSizeGB;
    size_t numElements;
    int threads;
    float workload_cpu;
    double durationSeconds;
    double dataTransferTime;
    bool isSorted;
} SORTINGINFO;

void readFileToUnifiedMemory(const char *filename, double *data, uint64_t numElements);
__device__ void print_array_device(double *array, int64_t array_size)
{
    for (int64_t i = 0; i < array_size; ++i)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}
// filename size workload
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s <filename> <arraysize in millions> <workload on cpu>\n");
        return -1;
    }

    const char *file_name = argv[1];
    uint64_t input_size = strtoull(argv[2], NULL, 10) * 1000000;
    float workload_cpu = atof(argv[3]);

    double *unSorted = NULL;
    HANDLE_ERROR(cudaMallocManaged(&unSorted, input_size * sizeof(double)));

    cudaEvent_t event, data_trans_start, data_trans_stop;
    cudaEventCreate(&event);

    cuda_timer_start(&data_trans_start, &data_trans_stop);
    readFileToUnifiedMemory(file_name, unSorted, input_size);
    double data_trans_time = cuda_timer_stop(data_trans_start, data_trans_stop) / 1000.0;
    // print_array_device(unSorted, 5);

    SORTINGINFO.dataSizeGB = (input_size * sizeof(double)) / (double)(1024 * 1024 * 1024);
    SORTINGINFO.numElements = input_size;
    SORTINGINFO.threads = omp_get_max_threads(); // Assuming OpenMP is used for parallelism
    SORTINGINFO.workload_cpu = workload_cpu;
    SORTINGINFO.durationSeconds = data_trans_time;  // Just for reading, adjust according to actual sort
    SORTINGINFO.dataTransferTime = data_trans_time; // Simplified assumption
    SORTINGINFO.isSorted = false;                   // Update after sorting
    printf("successful");

    HANDLE_ERROR(cudaFree(unSorted));
    return 0;
}

// Function to read data from the file into unified memory
void readFileToUnifiedMemory(const char *filename, double *data, uint64_t size_of_array)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Failed to open file for reading\n");
        exit(EXIT_FAILURE);
    }

    for (uint64_t i = 0; i < size_of_array; i++)
    {
        if (fscanf(file, "%lf", &data[i]) == EOF)
        {
            perror("Error reading from file");
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}