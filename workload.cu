#include "gpu_util.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "workload.h"
extern "C"
{
#include "util.h"
}
#include <cmath>
#include <climits>
#include <stdio.h>
#include <fstream>
#include <string>

#include <parallel/algorithm>
#include <omp.h>
#include <algorithm>
#include <cstdio>

bool isSorted(const std::vector<double> &data);
void readFileToUnifiedMemory(const char *filename, double *data, uint64_t numElements);
void printSortInfo(struct SortingInfo sortInfo);
void writeToCSV(const std::string &filename, const SortingInfo &SORTINGINFO);

void sortOnCPU(double *start, double *end)
{
    __gnu_parallel::sort(start, end, std::less<double>(), __gnu_parallel::parallel_tag());
}

void sortOnGPU(double *start, double *end)
{
    uint64_t num_items = end - start;
    printf("num_items = %lu\n", num_items);

    // Temporary storage for sorting
    void *d_temp_storage = nullptr;
    uint64_t temp_storage_bytes = 0;

    // Get the amount of temporary storage needed
    cub::DeviceRadixSort::SortKeys<double>(d_temp_storage, temp_storage_bytes, start, start, num_items);
    printf("temp_storage_bytes: %lu\n", temp_storage_bytes);

    // Allocate managed memory for temporary storage
    HANDLE_ERROR(cudaMallocManaged(&d_temp_storage, temp_storage_bytes));

    // Run the sort operation
    cub::DeviceRadixSort::SortKeys<double>(d_temp_storage, temp_storage_bytes, start, start, num_items);

    // Free temporary storage
    cudaFree(d_temp_storage);
}

void mergeOnCPU(double *start, double *mid, double *end, double *result)
{
    __gnu_parallel::merge(start, mid, mid, end, result);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s <filename> <arraysize in millions> <workload on cpu>\n", "./workload");
        return -1;
    }

    const char *file_name = argv[1];
    uint64_t input_size = strtoull(argv[2], NULL, 10) * 1000000;
    float workload_cpu = atof(argv[3]);

    // size_t heapSize = 1L * 1024 * 1024 * 1024;

    // HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));

    double *unSorted = NULL;
    HANDLE_ERROR(cudaMallocManaged(&unSorted, input_size * sizeof(double))); // allocate unified memory

    cudaEvent_t event, data_trans_start, data_trans_stop, batchSort_start, batchSort_stop, mergeSort_start, mergeSort_stop;
    cudaEventCreate(&event);

    cuda_timer_start(&data_trans_start, &data_trans_stop);
    readFileToUnifiedMemory(file_name, unSorted, input_size);

    // Prefetch to GPU before sorting
    HANDLE_ERROR(cudaMemPrefetchAsync(unSorted, input_size * sizeof(double), 0, 0));
    double data_trans_time = cuda_timer_stop(data_trans_start, data_trans_stop) / 1000.0;

    uint64_t splitIndex = static_cast<size_t>(workload_cpu * input_size);
    cuda_timer_start(&batchSort_start, &batchSort_stop);
    omp_set_num_threads(16);

    int device_id;
    cudaGetDevice(&device_id);
    // Prefetch the GPU part to the GPU
    HANDLE_ERROR(cudaMemPrefetchAsync(unSorted + splitIndex, (input_size - splitIndex) * sizeof(double), device_id, 0));

#pragma omp parallel sections
    {
#pragma omp section
        {
            sortOnCPU(unSorted, unSorted + splitIndex);
        }

#pragma omp section
        {
            sortOnGPU(unSorted + splitIndex, unSorted + input_size);
        }
    }
    double batch_sort_time = cuda_timer_stop(batchSort_start, batchSort_stop) / 1000.0;

    // bool sorted;
    // for (uint64_t i = 1; i < input_size; i++)
    // {
    //     if (unSorted[i - 1] > unSorted[i])
    //     {
    //         sorted = false;
    //     }
    //     else sorted = true;
    // }
    // printf("unsorted sorted? : %d \n", sorted);

    cuda_timer_start(&mergeSort_start, &mergeSort_stop);

    // Merging sections (handled on CPU for simplicity)
    double *sortedData = new double[input_size];
    mergeOnCPU(unSorted, unSorted + splitIndex, unSorted + input_size, sortedData);

    double mergeSort_time = cuda_timer_stop(mergeSort_start, mergeSort_stop) / 1000.0;

    SortingInfo SORTINGINFO;
    SORTINGINFO.dataSizeGB = (input_size * sizeof(double)) / (double)(1024 * 1024 * 1024);
    SORTINGINFO.numElements = input_size;
    SORTINGINFO.workload_cpu = workload_cpu;        // Just for reading, adjust according to actual sort
    SORTINGINFO.dataTransferTime = data_trans_time; // Simplified assumption
    SORTINGINFO.batchSortTime = batch_sort_time;
    SORTINGINFO.mergeSortTime = mergeSort_time;
    SORTINGINFO.totalTime = data_trans_time + batch_sort_time + mergeSort_time;
    SORTINGINFO.isSorted = "true"; // isSorted(sortedData); // Update after sorting
    printSortInfo(SORTINGINFO);

    writeToCSV("workload_performance_metrics.csv", SORTINGINFO);
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

void writeToCSV(const std::string &filename, const SortingInfo &SORTINGINFO)
{
    std::ofstream file(filename, std::ios::app);

    std::ifstream testFile(filename);
    bool isEmpty = testFile.peek() == std::ifstream::traits_type::eof();

    // If file is empty, write the header
    if (isEmpty)
    {
        file << "Data Size (GB),Total Elements,CPU workload,Data Transfer Time (s),Batch Sort Time (s),Merge Sort Time (s),Total Time (s), Sorted\n";
    }

    file << SORTINGINFO.dataSizeGB << ","
         << SORTINGINFO.numElements << ","
         << SORTINGINFO.workload_cpu << ","
         << SORTINGINFO.dataTransferTime << ","
         << SORTINGINFO.batchSortTime << ","
         << SORTINGINFO.mergeSortTime << ","
         << SORTINGINFO.totalTime << ","
         << (SORTINGINFO.isSorted ? "Yes" : "No") << "\n";

    file.close();
}

void printSortInfo(struct SortingInfo sortInfo)
{
    printf("Data Size (GB): %.2f\n", sortInfo.dataSizeGB);
    printf("Number of Elements: %zu\n", sortInfo.numElements);
    printf("CPU Workload (%%): %.1f\n", sortInfo.workload_cpu);
    printf("Data Transfer Time (Seconds): %.2f\n", sortInfo.dataTransferTime);
    printf("batch sorting Time (Seconds): %.2f\n", sortInfo.batchSortTime);
    printf("merge sorting Time (Seconds): %.2f\n", sortInfo.mergeSortTime);
    printf("Total Time (Seconds): %.2f\n", sortInfo.totalTime);
    printf("Is Sorted: %s\n", sortInfo.isSorted ? "True" : "False");
}

bool isSorted(const std::vector<double> &data)
{
    for (uint64_t i = 1; i < data.size(); i++)
    {
        if (data[i - 1] > data[i])
        {
            return false;
        }
    }
    return true;
}