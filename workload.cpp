#include "workload.h"
#include <fstream>
#include <string>

void writeToCSV(const std::string &filename, const SortingInfo &SORTINGINFO)
{
    std::ofstream file(filename, std::ios::app);
    std::ifstream testFile(filename);
    bool isEmpty = testFile.peek() == std::ifstream::traits_type::eof();
    testFile.close();

    if (isEmpty)
    {
        file << "Data Size (GB),Total Elements,CPU workload,Data Transfer Time (s),Batch Sort Time (s),Merge Sort Time (s),Total Time (s),Sorted\n";
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
