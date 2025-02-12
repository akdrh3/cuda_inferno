#include <iostream>
#include <vector>
#include <parallel/algorithm> // Include the header for parallel algorithms

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

    return 0;
}
