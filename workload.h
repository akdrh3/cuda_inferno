// workload.h
#include <string>
#ifndef WORKLOAD_H
#define WORKLOAD_H

#include <cstddef>

struct SortingInfo
{
    double dataSizeGB;
    size_t numElements;
    double workload_cpu;
    int cpu_thread_num;
    double dataTransferTime;
    double gpuSortTime;
    double cpuSortTime;
    double batchSortTime double mergeSortTime;
    double totalTime;
    bool isSorted;
};

// Declaration of functions that are implemented in C++ files
void writeToCSV(const std::string &filename, const SortingInfo &SORTINGINFO);

#endif // WORKLOAD_H
