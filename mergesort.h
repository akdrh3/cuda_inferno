// merge_sort.h
#ifndef MERGE_SORT_H
#define MERGE_SORT_H

#include <cstdint> // For uint64_t

#ifdef __cplusplus
extern "C" {
#endif

void cpu_merge(double *unSorted, uint64_t sizeOfArray, int threadNum);

#ifdef __cplusplus
}
#endif

#endif // MERGE_SORT_H
