#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>            // For std::atoi
#include <parallel/algorithm> // for __gnu_parallel::sort
#include <omp.h>
#include <chrono> // for omp_set_num_threads

void writeToCSV(const std::string &filename, size_t dataSize, size_t numElements, int threads, long long duration, bool isSorted);
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

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <number of elements> <thread>" << std::endl;
        return 1; // Exit if not enough arguments
    }

    std::string filename = argv[1];                    // First argument is the filename
    size_t numElements = std::atoi(argv[2]) * 1000000; // Second argument is the number of elements to read

    std::vector<double> data = readDoublesFromFile(filename, numElements);
    int threads = std::atoi(argv[3]);

    // Set the number of threads
    omp_set_num_threads(threads); // Adjust the number of threads like 16 or 20 as per your scenario
    auto start = std::chrono::high_resolution_clock::now();

    // Now call the parallel sort
    __gnu_parallel::sort(data.begin(), data.end());

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    long long durationSeconds = duration.count();
    double dataGB = dataSizeInGB(data);
    bool sortedStatus = isSorted(data);

    std::cout << "dataSize : " << dataGB << " sorting Time: " << durationSeconds << "s sorted: " << (sortedStatus ? "Yes" : "No") << std::endl;

    writeToCSV("performance_metrics.csv", dataGB, numElements, threads, durationSeconds, sortedStatus);
    return 0;
}

void writeToCSV(const std::string &filename, size_t dataSize, size_t numElements, int threads, long long duration, bool isSorted)
{
    std::ofstream file(filename, std::ios::app);

    std::ifstream testFile(filename);
    bool isEmpty = testFile.peek() == std::ifstream::traits_type::eof();
    // If file is empty, write the header
    if (isEmpty)
    {
        file << "Data Size (GB),Total Elements,Threads,Duration (seconds),Sorted\n";
    }

    file << "Data Size (GB),Total Elements,Threads,Duration (seconds),Sorted\n";

    file << dataSize << "," << numElements << "," << threads << "," << duration << "," << (isSorted ? "Yes" : "No") << "\n";

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
