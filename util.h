#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void print_array_host(int *array, int64_t array_size);
void read_from_file_cpu(char *file_name, int **numbers, uint64_t size_of_array);
void read_from_file(const char *file_name, int *numbers, uint64_t size_of_array);
uint64_t count_size_of_file(char *file_name);

#endif