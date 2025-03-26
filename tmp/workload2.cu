#include "gpu_util.cuh"
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
#include <cub/cub.cuh>

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


template <int ValueSize, bool IsPlus>
struct sm60_tuning {
    static constexpr int threads = 128;
    static constexpr uint64_t items = 16;  // Changed to uint64_t
};

template <>
struct sm60_tuning<8, true> { // Specialization for 8-byte values (e.g., uint64_t)
    static constexpr int threads = 256;
    static constexpr uint64_t items = 20;  // Changed to uint64_t
};

template <int ValueSize>
struct sm60_tuning<ValueSize, true> {
    static constexpr int threads = 128;
    static constexpr uint64_t items = 12;  // Changed to uint64_t
};

template <typename ValueType>
struct SortingPolicy {
    struct Policy600 : cub::ChainedPolicy<600, Policy600, Policy500> {
        using tuning = sm60_tuning<sizeof(ValueType), false>;
        using AlgorithmPolicy = cub::AgentRadixSortPolicy<
            tuning::threads, static_cast<int>(tuning::items), 4, true
        >;
    };
};




//
// Block-sorting CUDA kernel
//
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(double *d_in, double *d_out)
{
    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    using BlockLoadT = cub::BlockLoad<
        int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<
        int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE>;
    using BlockRadixSortT = cub::BlockRadixSort<
        int, BLOCK_THREADS, ITEMS_PER_THREAD>;

    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union
    {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    // Obtain this block's segment of consecutive keys (blocked across threads)
    int thread_keys[ITEMS_PER_THREAD];
    //int thread_offset = blockIdx.x * (ITEMS_PER_THREAD);
    //thread_keys = thread_keys + thread_offset;
    uint64_t block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);

    __syncthreads(); // Barrier for smem reuse

    // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

    __syncthreads(); // Barrier for smem reuse

    // Store the sorted segment
    BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
}



void gpu_merge(double *start, double *end, SortingInfo *SORTINGINFO)
{
    cudaEvent_t event, gpuSortTimeStart, gpuSortTimeStop;
    cudaEventCreate(&event);
    cudaError_t err;

    uint64_t num_items = end - start;
    // printf("num_items = %lu\n", num_items);

    // Temporary storage for sorting
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    using ValueType = uint64_t;  // Sorting uint64_t data
    using MyPolicy = SortingPolicy<ValueType>::Policy600;

    cuda_timer_start(&gpuSortTimeStart, &gpuSortTimeStop);

    
    // double *d_out = NULL;
    // int *thread_keys = NULL; 
    uint64_t num_blocks = 144;
    // //uint64_t num_blocks = (num_items + (512 * 16 - 1)) / (512 * 16);
    uint64_t items_per_thread = num_items + (num_blocks * 512-1)/ (num_blocks * 512);
    
    // HANDLE_ERROR(cudaMallocManaged(&d_out, num_items * sizeof(double))); // allocate unified memory
    // HANDLE_ERROR(cudaMallocManaged(&thread_keys, sizeof(int) * num_blocks * items_per_thread));

    // std::cout << "items per thread: " << items_per_thread << std::endl;
    // BlockSortKernel<512, 16><<<num_blocks, 512>>>(start, d_out, thread_keys);
    // HANDLE_ERROR(cudaDeviceSynchronize());
    // start = d_out;
    
    // Get the amount of temporary storage needed
    err = cub::DeviceRadixSort::SortKeys<MyPolicy>(d_temp_storage, temp_storage_bytes, start, start, num_items);
    if (err != cudaSuccess)
    {
        printf("Error in estimating storage: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("temp_storage_bytes: %zu\n", temp_storage_bytes);

    // Allocate managed memory for temporary storage
    HANDLE_ERROR(cudaMallocManaged(&d_temp_storage, temp_storage_bytes));

    // Run the sort operation
    err = cub::DeviceRadixSort::SortKeys<MyPolicy>(d_temp_storage, temp_storage_bytes, start, start, num_items);
    if (err != cudaSuccess)
    {
        printf("Error in estimating storage: %s\n", cudaGetErrorString(err));
        return;
    }
    double gpuSortTime = cuda_timer_stop(gpuSortTimeStart, gpuSortTimeStop) / 1000.0;
    std::cout << "gpu sorting Time: " << gpuSortTime << std::endl;
    SORTINGINFO->gpuSortTime = gpuSortTime;

    // Free temporary storage
    // cudaFree(d_temp_storage);
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
    uint64_t input_size = strtoull(argv[2], NULL, 10); //* 1000000;
    double workload_cpu = strtod(argv[3], NULL);
    int cpu_thread_num = atoi(argv[4]);

    // size_t heapSize = 1L * 1024 * 1024 * 1024;

    // HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));

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
    std::cout << "data Prefetch Time: " << dataPrefetchTime << std::endl;
    SORTINGINFO.dataPrefetchTime = dataPrefetchTime;

    uint64_t splitIndex = workload_cpu * input_size;
    cuda_timer_start(&batchSort_start, &batchSort_stop);
    omp_set_num_threads(cpu_thread_num);
    omp_set_nested(1); // Enable nested parallelism
    printf("splitIndex: %lu, inputSize: %lu\n", splitIndex, input_size);

    if (workload_cpu != 1 && workload_cpu != 0)
    {
        printf("cpu not 1 or 0\n");
#pragma omp parallel sections
        {
#pragma omp section
            {
                cpu_merge(unSorted, splitIndex, cpu_thread_num, &SORTINGINFO);
            }

#pragma omp section
            {
                gpu_merge(unSorted + splitIndex, unSorted + input_size, &SORTINGINFO);
            }
        }
    }
    else if (workload_cpu == 1)
    {
        printf("cpu 1.0\n");
        SORTINGINFO.gpuSortTime = 0;
        cpu_merge(unSorted, splitIndex, cpu_thread_num, &SORTINGINFO);
    }
    else
    {
        printf("cpu 0\n");
        SORTINGINFO.cpuSortTime = 0;
        gpu_merge(unSorted + splitIndex, unSorted + input_size, &SORTINGINFO);
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

    printf("unsorted sorted? : %d \n", sorted);
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
