#include "cuda_profiler_api.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__device__ void print_array_device(int *array, int64_t array_size);
__host__ void HandleError(cudaError_t err, const char *file, int line);
__host__ void cuda_timer_start(cudaEvent_t *start, cudaEvent_t *stop);
__host__ float cuda_timer_stop(cudaEvent_t start, cudaEvent_t stop);