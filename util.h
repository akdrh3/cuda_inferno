#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE_IN_GB(x) ((double)x / (1024.0 * 1024.0 * 1024.0))
#define SIZE_IN_MB(x) ((double)x / (1024.0 * 1024.0))

void print_array_host(int *array, int64_t array_size);
void read_from_file_cpu(const char *file_name, int *numbers, uint64_t size_of_array);
void read_from_file(const char *file_name, int *numbers, uint64_t size_of_array);
uint64_t count_size_of_file(const char *file_name);
int isRangeSorted_cpu(int *arr, size_t start, size_t end);

#endif