// workload.h
#ifndef WORKLOAD_H
#define WORKLOAD_H

#include <cstddef>

struct SortingInfo
{
    double dataSizeGB;
    size_t numElements;
    float workload_cpu;
    double dataTransferTime;
    double batchSortTime;
    double mergeSortTime;
    double totalTime;
    bool isSorted;
};

// Declaration of functions that are implemented in C++ files
void writeToCSV(const std::string &filename, const SortingInfo &SORTINGINFO);

#endif // WORKLOAD_H
