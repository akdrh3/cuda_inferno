#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib> // For std::atoi

std::vector<double> readDoublesFromFile(const std::string &filename, size_t numElements);

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <number of elements>" << std::endl;
        return 1; // Exit if not enough arguments
    }

    std::string filename = argv[1];          // First argument is the filename
    size_t numElements = std::atoi(argv[2]); // Second argument is the number of elements to read

    std::vector<double> data = readDoublesFromFile(filename, numElements);

    // Output the contents of the vector
    for (double num : data)
    {
        std::cout << num << std::endl;
    }

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
