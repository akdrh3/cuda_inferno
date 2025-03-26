#include "gpu_util.cuh"
#include "workload.h"
#include "sort.h"
#include "utils.h"

extern "C"
{
#include "util.h"
}
#include <cmath>
#include <climits>
#include <stdio.h>
#include <fstream>
#include <string>
#include <cub/cub.cuh>
#include "device_launch_parameters.h"

#include <parallel/algorithm>
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <pthread.h>
#include <vector>
#include <chrono>


bool isSorted(const std::vector<double> &data);
void readFileToUnifiedMemory(const char *filename, double *data, uint64_t numElements);
void printSortInfo(struct SortingInfo sortInfo);
void writeToCSV(const std::string &filename, const SortingInfo &SORTINGINFO);
void cpu_merge(double *unSorted, uint64_t sizeOfArray, int threadNum, SortingInfo *SORTINGINFO);

__global__ void gpu_mergesort(double*, double*, uint64_t, uint64_t, uint64_t, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(double*, double*, uint64_t, uint64_t, uint64_t);
bool verbose = true;
dim3 threadsPerBlock;
dim3 blocksPerGrid;


void gpu_merge(double *start, double *end, SortingInfo *SORTINGINFO, dim3 threadsPerBlock, dim3 blocksPerGrid)
{
    cudaEvent_t event, gpuSortTimeStart, gpuSortTimeStop;
    cudaEventCreate(&event);
    cuda_timer_start(&gpuSortTimeStart, &gpuSortTimeStop);

    uint64_t num_items = end - start;
    printf("num_items = %lu\n", num_items);


    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    double* D_data;
    double* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays

    HANDLE_ERROR(cudaMallocManaged((void**) &D_data, num_items * sizeof(double)));
    HANDLE_ERROR(cudaMallocManaged((void**) &D_swp, num_items * sizeof(double)));
    // if (verbose)
    //     std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    //
    // Copy the thread / block info to the GPU as well
    //
    HANDLE_ERROR(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    HANDLE_ERROR(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

    // if (verbose)
    //     std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    HANDLE_ERROR(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));
    // if (verbose)
    //     std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    double* A = D_data;
    double* B = D_swp;

    uint64_t nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    cuda_timer_start(&gpuSortTimeStart, &gpuSortTimeStop);
    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (uint64_t width = 2; width < (num_items << 1); width <<= 1) {
        uint64_t slices = num_items / ((nThreads) * width) + 1;

        // if (verbose) {
        //     std::cout << "mergeSort - width: " << width 
        //               << ", slices: " << slices 
        //               << ", nThreads: " << nThreads << '\n';
        //     tm();
        // }
 

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, num_items, width, slices, D_threads, D_blocks);

        // if (verbose)
        //     std::cout << "call mergesort kernel: " << tm() << " microseconds\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    //
    // Get the list back from the GPU
    //
    // Free the GPU memory
    HANDLE_ERROR(cudaFree(A));
    HANDLE_ERROR(cudaFree(B));
    // if (verbose)
    //     std::cout << "cudaFree: " << tm() << " microseconds\n";



    double gpuSortTime = cuda_timer_stop(gpuSortTimeStart, gpuSortTimeStop) / 1000.0;
    std::cout << "gpu sorting Time: " << gpuSortTime << std::endl;
    SORTINGINFO->gpuSortTime = gpuSortTime;

}

void merge_on_cpu(double *start, double *mid, double *end, double *result, SortingInfo *SORTINGINFO)
{
    __gnu_parallel::merge(start, mid, mid, end, result);
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        printf("Usage: %s <filename> <arraysize in millions> <workload on cpu> <cpu thread number>\n", "./workload");
        return -1;
    }

    const char *file_name = argv[1];
    uint64_t input_size = strtoull(argv[2], NULL, 10) * 1000000;
    double workload_cpu = strtod(argv[3], NULL);
    int cpu_thread_num = atoi(argv[4]);
    double *unSorted = NULL;
    HANDLE_ERROR(cudaMallocManaged(&unSorted, input_size * sizeof(double))); // allocate unified memory

    SortingInfo SORTINGINFO;

    cudaEvent_t event, data_trans_start, data_trans_stop, dataPrefetchTimeStart, dataPrefetchTimeStop, batchSort_start, batchSort_stop, mergeSort_start, mergeSort_stop;
    cudaEventCreate(&event);

    cuda_timer_start(&data_trans_start, &data_trans_stop);
    readFileToUnifiedMemory(file_name, unSorted, input_size);
    double data_trans_time = cuda_timer_stop(data_trans_start, data_trans_stop) / 1000.0;

    // Prefetch to GPU before sorting
    cuda_timer_start(&dataPrefetchTimeStart, &dataPrefetchTimeStop);
    HANDLE_ERROR(cudaMemPrefetchAsync(unSorted, input_size * sizeof(double), 0, 0));
    double dataPrefetchTime = cuda_timer_stop(dataPrefetchTimeStart, dataPrefetchTimeStop) / 1000.0;
    SORTINGINFO.dataPrefetchTime = dataPrefetchTime;

    uint64_t splitIndex = workload_cpu * input_size;
    cuda_timer_start(&batchSort_start, &batchSort_stop);
    omp_set_num_threads(cpu_thread_num);
    omp_set_nested(1); // Enable nested parallelism
    printf("splitIndex: %lu, inputSize: %lu\n", splitIndex, input_size);



    threadsPerBlock.x = 512;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 2;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    if (workload_cpu != 1 && workload_cpu != 0)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                cpu_merge(unSorted, splitIndex, cpu_thread_num, &SORTINGINFO);
            }

#pragma omp section
            {
                gpu_merge(unSorted + splitIndex, unSorted + input_size, &SORTINGINFO, threadsPerBlock, blocksPerGrid);
            }
        }
    }
    else if (workload_cpu == 1)
    {
        SORTINGINFO.gpuSortTime = 0;
        cpu_merge(unSorted, splitIndex, cpu_thread_num, &SORTINGINFO);
    }
    else
    {
        SORTINGINFO.cpuSortTime = 0;
        gpu_merge(unSorted + splitIndex, unSorted + input_size, &SORTINGINFO, threadsPerBlock, blocksPerGrid);
    }

    double batch_sort_time = cuda_timer_stop(batchSort_start, batchSort_stop) / 1000.0;
    cuda_timer_start(&mergeSort_start, &mergeSort_stop);

    // Merging sections (handled on CPU for simplicity)
    double *sortedData = new double[input_size];
    merge_on_cpu(unSorted, unSorted + splitIndex, unSorted + input_size, sortedData, &SORTINGINFO);

    double mergeSort_time = cuda_timer_stop(mergeSort_start, mergeSort_stop) / 1000.0;

    bool sorted;
    for (uint64_t i = 1; i < input_size; i++)
    {
        if (unSorted[i - 1] > unSorted[i])
        {
            sorted = false;
        }
        else
            sorted = true;
    }

    printf("after merge unsorted sorted? : %d \n", sorted);
    SORTINGINFO.dataSizeGB = (input_size * sizeof(double)) / (double)(1024 * 1024 * 1024);
    SORTINGINFO.numElements = input_size;
    SORTINGINFO.workload_cpu = workload_cpu; // Just for reading, adjust according to actual sort
    SORTINGINFO.cpu_thread_num = cpu_thread_num;
    SORTINGINFO.dataTransferTime = data_trans_time; // Simplified assumption
    SORTINGINFO.batchSortTime = batch_sort_time;
    SORTINGINFO.mergeSortTime = mergeSort_time;
    SORTINGINFO.totalSortTime = SORTINGINFO.gpuSortTime + SORTINGINFO.cpuSortTime + mergeSort_time;
    SORTINGINFO.totalTime = data_trans_time + SORTINGINFO.dataPrefetchTime + batch_sort_time + mergeSort_time;
    SORTINGINFO.isSorted = sorted; // isSorted(sortedData); // Update after sorting
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
        file << "Data Size (GB),Total Elements,CPU workload, cpu_thread_num, Data Transfer Time (s),Data Prefetch Time (s),GPU Sort Time (s), CPU Sort Time(s) ,Batch Sort Time (s),Merge Sort Time (s),Total Sort Time (s), Total Time (s), Sorted\n";
    }

    file << SORTINGINFO.dataSizeGB << ","
         << SORTINGINFO.numElements << ","
         << SORTINGINFO.workload_cpu << ","
         << SORTINGINFO.cpu_thread_num << ","
         << SORTINGINFO.dataTransferTime << ","
         << SORTINGINFO.dataPrefetchTime << ","
         << SORTINGINFO.gpuSortTime << ","
         << SORTINGINFO.cpuSortTime << ","
         << SORTINGINFO.batchSortTime << ","
         << SORTINGINFO.mergeSortTime << ","
         << SORTINGINFO.totalSortTime << ","
         << SORTINGINFO.totalTime << ","
         << (SORTINGINFO.isSorted ? "Yes" : "No") << "\n";

    file.close();
}

void printSortInfo(struct SortingInfo sortInfo)
{
    printf("Data Size (GB): %.2f\n", sortInfo.dataSizeGB);
    printf("Number of Elements: %zu\n", sortInfo.numElements);
    printf("CPU Workload (%%): %lf\n", sortInfo.workload_cpu);
    printf("CPU thread number: %d\n", sortInfo.cpu_thread_num);
    printf("Data Transfer Time (Seconds): %.2f\n", sortInfo.dataTransferTime);
    printf("Data prefetch Time (Seconds): %.2f\n", sortInfo.dataPrefetchTime);
    printf("gpu sorting Time (Seconds): %.2f\n", sortInfo.gpuSortTime);
    printf("cpu sorting Time (Seconds): %.2f\n", sortInfo.cpuSortTime);

    printf("batch sorting Time (Seconds): %.2f\n", sortInfo.batchSortTime);
    printf("merge sorting Time (Seconds): %.2f\n", sortInfo.mergeSortTime);
    printf("Total Sort Time (Seconds): %.2f\n", sortInfo.totalSortTime);
    printf("Total Time (Seconds): %.2f\n", sortInfo.totalTime);
    printf("Is Sorted: %s\n", sortInfo.isSorted ? "True" : "False");
}


// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(double* source, double* dest, uint64_t size, uint64_t width, uint64_t slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    uint64_t start = width*idx*slices, 
         middle, 
         end;

    for (uint64_t slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(double* source, double* dest, uint64_t start, uint64_t middle, uint64_t end) {
    uint64_t i = start;
    uint64_t j = middle;
    for (uint64_t k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
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

using namespace std;

struct ThreadArgs
{
    double *array;
    uint64_t left;
    uint64_t right;
    uint64_t depth;
    uint64_t maxDepth;
};

pthread_mutex_t mutexPart;
uint64_t currentPart = 0;

void merge(double *arr, uint64_t l, uint64_t m, uint64_t r)
{
    uint64_t i, j, k;
    uint64_t n1 = m - l + 1;
    uint64_t n2 = r - m;

    // Create temp arrays
    double *L = new double[n1];
    double *R = new double[n2];

    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temp arrays back into arr[l..r]
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

void *mergeSort(void *args)
{
    ThreadArgs *arg = (ThreadArgs *)args;
    if (arg->left < arg->right)
    {
        if (arg->depth >= arg->maxDepth)
        {
            // Perform normal mergesort if max depth reached
            uint64_t mid = arg->left + (arg->right - arg->left) / 2;
            ThreadArgs leftArgs = {arg->array, arg->left, mid, arg->depth + 1, arg->maxDepth};
            ThreadArgs rightArgs = {arg->array, mid + 1, arg->right, arg->depth + 1, arg->maxDepth};
            mergeSort(&leftArgs);
            mergeSort(&rightArgs);
            merge(arg->array, arg->left, mid, arg->right);
        }
        else
        {
            uint64_t mid = arg->left + (arg->right - arg->left) / 2;

            pthread_t leftThread, rightThread;
            ThreadArgs leftArgs = {arg->array, arg->left, mid, arg->depth + 1, arg->maxDepth};
            ThreadArgs rightArgs = {arg->array, mid + 1, arg->right, arg->depth + 1, arg->maxDepth};

            pthread_create(&leftThread, NULL, mergeSort, &leftArgs);
            pthread_create(&rightThread, NULL, mergeSort, &rightArgs);

            pthread_join(leftThread, NULL);
            pthread_join(rightThread, NULL);

            merge(arg->array, arg->left, mid, arg->right);
        }
    }
    return NULL;
}

void cpu_merge(double *unSorted, uint64_t sizeOfArray, int threadNum, SortingInfo *SORTINGINFO)
{

    // Calculate max depth based on number of threads
    uint64_t maxDepth = 0;
    while ((1 << maxDepth) < threadNum)
        maxDepth++;

    pthread_mutex_init(&mutexPart, NULL);

    ThreadArgs args = {unSorted, 0, sizeOfArray - 1, 0, maxDepth};

    auto start = std::chrono::high_resolution_clock::now();
    mergeSort(&args);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Sorting took " << elapsed.count() << " milliseconds.\n";
    SORTINGINFO->cpuSortTime = elapsed.count();

    pthread_mutex_destroy(&mutexPart);
}
