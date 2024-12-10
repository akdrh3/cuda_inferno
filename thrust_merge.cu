#include "gpu_util.cuh"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
extern "C"
{
#include "util.h"
}
#include <cmath>
#include <climits>
#include <stdio.h>

int main(int argc, char *argv[])
{
    // get param from command; filename , arraysize * 1 million
    const char *file_name = argv[1];
    uint64_t input_size = strtoull(argv[2], NULL, 10) * 10;
    int *a = (int *)malloc(sizeof(int) * input_size);
    if (a == NULL)
    {
        return -1;
    }
    read_from_file_cpu(file_name, a, input_size);
    print_array_host(a, input_size);

    free(a);

    return 0;
}
