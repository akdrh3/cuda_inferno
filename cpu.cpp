#include <iostream>
#include <vector>
#include <parallel/algorithm> // Include the header for parallel algorithms
#include <fstream>
#include <string>

std::vector<double> readDoublesFromFile(const std::string &filename)
{
    std::ifstream file(filename);
    std::vector<double> data;

    if (!file)
    {
        std::cerr << "Unable to open file " << filename << std::endl;
        return data; // Return an empty vector if the file cannot be opened
    }

    double value;
    while (file >> value)
    {
        data.push_back(value);
    }

    file.close();
    return data;
}

int sizet()
{
    std::cout << "The size of size_t is " << sizeof(size_t) << " bytes." << std::endl;
    return 0;
}

int main()
{
    // Create a vector with some unsorted data
    std::vector<int> data{5, 3, 8, 6, 2, 7, 4, 1};

    // Call the parallel sort using OpenMP
    __gnu_parallel::sort(data.begin(), data.end());

    // Output the sorted vector
    for (int n : data)
    {
        std::cout << n << " ";
    }

    int result = sizet();

    return 0;
}
