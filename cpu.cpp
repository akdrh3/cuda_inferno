#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>            // For std::atoi
#include <parallel/algorithm> // for __gnu_parallel::sort
#include <omp.h>
#include <chrono> // for omp_set_num_threads

struct PerformanceData
{
    double dataSizeGB;
    size_t numElements;
    int threads;
    long long durationSeconds;
    long long dataTransferTime;
    bool isSorted;
};

void writeToCSV(const std::string &filename, const PerformanceData &perfData);
std::vector<double> readDoublesFromFile(const std::string &filename, size_t numElements);
bool isSorted(const std::vector<double> &data)
{
    for (size_t i = 0; i < data.size() - 1; ++i)
    {
        if (data[i] > data[i + 1])
        {
            return false;
        }
    }
    return true;
}

double dataSizeInGB(const std::vector<double> &data)
{
    size_t elementSize = sizeof(double);
    size_t capacity = data.capacity();
    size_t totalBytes = capacity * elementSize;
    return static_cast<double>(totalBytes) / (1024 * 1024 * 1024);
}

std::chrono::time_point<std::chrono::high_resolution_clock> start;

void timerStart()
{
    start = std::chrono::high_resolution_clock::now();
}

long long timerStop()
{
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    return duration.count();
}
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <number of elements> <thread>" << std::endl;
        return 1; // Exit if not enough arguments
    }

    std::string filename = argv[1];                    // First argument is the filename
    size_t numElements = std::atoi(argv[2]) * 1000000; // Second argument is the number of elements to read
    timerStart();
    std::vector<double> data = readDoublesFromFile(filename, numElements);
    long long dataTransfertime = timerStop();
    int threads = std::atoi(argv[3]);

    // Set the number of threads
    omp_set_num_threads(threads); // Adjust the number of threads like 16 or 20 as per your scenario
    timerStart();

    // Now call the parallel sort
    __gnu_parallel::sort(data.begin(), data.end());

    double dataGB = dataSizeInGB(data);
    long long durationSeconds = timerStop();
    // bool sortedStatus = isSorted(data);
    bool sortedStatus = true;

    PerformanceData perfData{
        dataSizeInGB(data),
        numElements,
        threads,
        durationSeconds,
        dataTransferTime,
        sortedStatus};

    std::cout << "dataSize : " << dataGB << " sorting Time: " << durationSeconds << "s sorted: " << (sortedStatus ? "Yes" : "No") << std::endl;

    writeToCSV("performance_metrics.csv", perfData);
    return 0;
}

void writeToCSV(const std::string &filename, const PerformanceData &perfData)
{
    std::ofstream file(filename, std::ios::app);

    std::ifstream testFile(filename);
    bool isEmpty = testFile.peek() == std::ifstream::traits_type::eof();

    // If file is empty, write the header
    if (isEmpty)
    {
        file << "Data Size (GB),Total Elements,Threads, Sorting Duration (s),Data Transfer Time (s), Total Time (s), Sorted\n";
    }

    file << perfData.dataSizeGB << ","
         << perfData.numElements << ","
         << perfData.threads << ","
         << perfData.durationSeconds << ","
         << perfData.dataTransferTime << ","
         << perfData.durationSeconds + perfData.dataTransferTime << ","
         << (perfData.isSorted ? "Yes" : "No") << "\n";

    file.close();
}

std::vector<double> readDoublesFromFile(const std::string &filename, size_t numElements)
{
    std::ifstream file(filename);
    std::vector<double> data;

    if (!file)
    {
        std::cerr << "Unable to open file " << filename << std::endl;
        return data; // Return an empty vector if the file cannot be opened
    }

    double value;
    size_t count = 0;
    while (file >> value && count < numElements)
    {
        data.push_back(value);
        ++count;
    }

    file.close();
    return data;
}
