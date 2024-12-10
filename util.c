#include "util.h"
#include <stdio.h>
#include <stdlib.h>

void print_array_host(int *array, int64_t array_size)
{
    for (int64_t i = 0; i < array_size; ++i)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

void read_from_file_cpu(const char *file_name, int *numbers, uint64_t size_of_array)
{
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (uint64_t i = 0; i < size_of_array; i++)
    {
        if (fscanf(file, "%d", numbers[i]) == EOF)
        {
            perror("Error reading from file");
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

void read_from_file(const char *file_name, int *numbers, uint64_t size_of_array)
{
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (uint64_t i = 0; i < size_of_array; i++)
    {
        if (fscanf(file, "%d", &numbers[i]) == EOF)
        {
            perror("Error reading from file");
            exit(EXIT_FAILURE);
        }
    }
}

uint64_t count_size_of_file(const char *file_name)
{
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    uint64_t count = 0;
    int tmp = 0;
    while (fscanf(file, "%d", &tmp) != EOF)
    {
        count++;
    }

    fclose(file);
    return count;
}
int isRangeSorted_cpu(int *arr, size_t start, size_t end)
{
    if (start >= end) // Invalid range
    {
        return 1; // A single element or empty range is always sorted
    }

    for (size_t i = start + 1; i < end; ++i)
    {
        if (arr[i - 1] > arr[i])
        {
            return 0; // Found an element out of order
        }
    }
    return 1; // The range is sorted
}