#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>            // For std::atoi
#include <parallel/algorithm> // for __gnu_parallel::sort
#include <omp.h>              // for omp_set_num_threads

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
    std::cout << "data is " << (isSorted(data) ? "sorted." : "not sorted.") << std::endl;

    // Set the number of threads
    omp_set_num_threads(std::atoi(argv[3])); // Adjust the number of threads like 16 or 20 as per your scenario

    // Now call the parallel sort
    __gnu_parallel::sort(data.begin(), data.end());

    std::cout << "data is " << (isSorted(data) ? "sorted." : "not sorted.") << std::endl;

    return 0;
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
