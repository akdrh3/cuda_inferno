#include <fstream>
#include <iomanip> // Include for std::fixed and std::setprecision
#include <string>
#include <workload.h>

void writeToCSV(const std::string &filename, const SortingInfo &SORTINGINFO)
{
    std::ofstream file(filename, std::ios::app);
    std::ifstream testFile(filename);
    bool isEmpty = testFile.peek() == std::ifstream::traits_type::eof();
    testFile.close();

    if (isEmpty)
    {
        file << "Data Size (GB),Total Elements,CPU workload,cpu_thread_num,Data Transfer Time (s),Data Prefetch Time (s),GPU Sort Time (s), CPU Sort Time(s) ,Batch Sort Time (s),Merge Sort Time (s),Total Time (s), Sorted\n";
    }

    file << std::fixed << std::setprecision(2); // Set fixed-point notation with two decimals
    file << SORTINGINFO.dataSizeGB << ","
         << SORTINGINFO.numElements << ","
         << std::fixed << std::setprecision(1) << SORTINGINFO.workload_cpu << "," SORTINGINFO.cpu_thread_num << ","
         << std::fixed << std::setprecision(2) << SORTINGINFO.dataTransferTime << ","
         << std::fixed << std::setprecision(2) << SORTINGINFO.dataprefetchTime << ","
         << std::fixed << std::setprecision(2) << SORTINGINFO.gpuSortTime << ","
         << std::fixed << std::setprecision(2) << SORTINGINFO.cpuSortTime << ","
         << SORTINGINFO.batchSortTime << ","
         << SORTINGINFO.mergeSortTime << ","
         << SORTINGINFO.totalTime << ","
         << (SORTINGINFO.isSorted ? "Yes" : "No") << "\n";

    file.close();
}
