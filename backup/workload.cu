#include "gpu_util.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/pair.h>
#include <cstdlib>
#include <map>
#include <cassert>
#include <thrust/iterator/retag.h>

#include <iostream>
#include <thrust/universal_allocator.h>
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


#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/remove.h>

// create a custom execution policy by deriving from the existing cuda::execution_policy
struct my_policy : thrust::cuda::execution_policy<my_policy> {};

// provide an overload of malloc() for my_policy
__host__ __device__ void* malloc(my_policy, size_t n )
{
  printf("hello, world from my special malloc!\n");
  double *result = NULL;
  HANDLE_ERROR(cudaMallocManaged(&result, num_bytes));

  return thrust::raw_pointer_cast(cudaMallocManaged(n));
}

// create a custom allocator which will use our malloc
// we can inherit from cuda::allocator to reuse its existing functionality
template<class T>
struct my_allocator : thrust::cuda::allocator<T>
{
  using super_t = thrust::cuda::allocator<T>;
  using pointer = typename super_t::pointer;

  pointer allocate(size_t n)
  {
    T* raw_ptr = reinterpret_cast<T*>(malloc(my_policy{}, sizeof(T) * n));

    // wrap the raw pointer in the special pointer wrapper for cuda pointers
    return pointer(raw_ptr);
  }
};

template<class T>
using my_vector = thrust::cuda::vector<T, my_allocator<T>>;

bool isSorted(const std::vector<double> &data);
void readFileToUnifiedMemory(const char *filename, double *data, uint64_t numElements);
void printSortInfo(struct SortingInfo sortInfo);
void writeToCSV(const std::string &filename, const SortingInfo &SORTINGINFO);


// // cached_allocator: a simple allocator for caching allocation requests
// struct cached_allocator
// {
//   cached_allocator() {}

//   void *allocate(std::ptrdiff_t num_bytes)
//   {
//     std::cout << "inside myallocate" << std::endl;
//     void *result = 0;

//     // search the cache for a free block
//     free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

//     if(free_block != free_blocks.end())
//     {
//       std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

//       // get the pointer
//       result = free_block->second;

//       // erase from the free_blocks map
//       free_blocks.erase(free_block);
//     }
//     else
//     {
//       // no allocation of the right size exists
//       // create a new one with cuda::malloc
//       // throw if cuda::malloc can't satisfy the request
//       try
//       {
//         std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;

//         // allocate memory and convert cuda::pointer to raw pointer
//         HANDLE_ERROR(cudaMallocManaged(&result, num_bytes));
//       }
//       catch(std::runtime_error &e)
//       {
//         throw;
//       }
//     }

//     // insert the allocated pointer into the allocated_blocks map
//     allocated_blocks.insert(std::make_pair(result, num_bytes));

//     return thrust::device_pointer_cast(result);
//   }

//   void deallocate(void *ptr)
//   {
//     // erase the allocated block from the allocated blocks map
//     allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
//     std::ptrdiff_t num_bytes = iter->second;
//     allocated_blocks.erase(iter);

//     // insert the block into the free blocks map
//     free_blocks.insert(std::make_pair(num_bytes, ptr));
//   }

//   void free_all()
//   {
//     std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

//     // deallocate all outstanding blocks in both lists
//     for(free_blocks_type::iterator i = free_blocks.begin();
//         i != free_blocks.end();
//         ++i)
//     {
//       // transform the pointer to cuda::pointer before calling cuda::free
//       thrust::system::cuda::free(thrust::system::cuda::pointer<void>(i->second));
//     }

//     for(allocated_blocks_type::iterator i = allocated_blocks.begin();
//         i != allocated_blocks.end();
//         ++i)
//     {
//       // transform the pointer to cuda::pointer before calling cuda::free
//       thrust::system::cuda::free(thrust::system::cuda::pointer<void>(i->first));
//     }
//   }

//   typedef std::multimap<std::ptrdiff_t, void*> free_blocks_type;
//   typedef std::map<void *, std::ptrdiff_t>     allocated_blocks_type;

//   free_blocks_type      free_blocks;
//   allocated_blocks_type allocated_blocks;
// };


// // the cache is simply a global variable
// // XXX ideally this variable is declared thread_local
// cached_allocator g_allocator;


// // create a tag derived from system::cuda::tag for distinguishing
// // our overloads of get_temporary_buffer and return_temporary_buffer
// struct my_tag : thrust::system::cuda::tag {};


// // overload get_temporary_buffer on my_tag
// // its job is to forward allocation requests to g_allocator
// template<typename T>
//   thrust::pair<T*, std::ptrdiff_t>
//     get_temporary_buffer(my_tag, std::ptrdiff_t n)
// {
//   // ask the allocator for sizeof(T) * n bytes
//   T* result = reinterpret_cast<T*>(g_allocator.allocate(sizeof(T) * n));

//   // return the pointer and the number of elements allocated
//   return thrust::make_pair(result,n);
// }


// // overload return_temporary_buffer on my_tag
// // its job is to forward deallocations to g_allocator
// // an overloaded return_temporary_buffer should always accompany
// // an overloaded get_temporary_buffer
// template<typename Pointer>
//   void return_temporary_buffer(my_tag, Pointer p)
// {
//   // return the pointer to the allocator
//   g_allocator.deallocate(thrust::raw_pointer_cast(p));
// }



void sortOnCPU(double *start, double *end)
{
    __gnu_parallel::sort(start, end, std::less<double>(), __gnu_parallel::parallel_tag());
}

void sortOnGPU(double *start, double *end) 
{
    my_policy policy;
    thrust::device_ptr<double> univ_ptr(start);
    thrust::device_ptr<double> end_ptr(end);

    thrust::sort(policy,univ_ptr,
                 end_ptr);

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

    printf("splitIndex : %lu, start : %lu, end : %lu\n", splitIndex, splitIndex, input_size);
    cuda_timer_start(&batchSort_start, &batchSort_stop);
    omp_set_num_threads(16);
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
    
    bool sorted;
    for (uint64_t i = 1; i < input_size; i++)
    {
        if (unSorted[i - 1] > unSorted[i])
        {
            sorted = false;
        }
        else sorted = true;
    }
    printf("unsorted sorted ? : %d \n", sorted);

    cuda_timer_start(&mergeSort_start, &mergeSort_stop);

    // Merging sections (handled on CPU for simplicity)
    std::vector<double> sortedData(input_size);
#pragma omp parallel
    {
        __gnu_parallel::merge(unSorted, unSorted + splitIndex,
                              unSorted + splitIndex, unSorted + input_size,
                              sortedData.begin());
    }
    // g_allocator.free_all();

    double mergeSort_time = cuda_timer_stop(mergeSort_start, mergeSort_stop) / 1000.0;

    SortingInfo SORTINGINFO;
    SORTINGINFO.dataSizeGB = (input_size * sizeof(double)) / (double)(1024 * 1024 * 1024);
    SORTINGINFO.numElements = input_size;
    SORTINGINFO.workload_cpu = workload_cpu;        // Just for reading, adjust according to actual sort
    SORTINGINFO.dataTransferTime = data_trans_time; // Simplified assumption
    SORTINGINFO.batchSortTime = batch_sort_time;
    SORTINGINFO.mergeSortTime = mergeSort_time;
    SORTINGINFO.totalTime = data_trans_time + batch_sort_time + mergeSort_time;
    SORTINGINFO.isSorted = isSorted(sortedData); // Update after sorting
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