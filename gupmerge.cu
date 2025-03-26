
// profiling
int tm();
// data[], size, threads, blocks, 
void mergesort(double*, uint64_t, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(double*, double*, uint64_t, uint64_t, uint64_t, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(double*, double*, uint64_t, uint64_t, uint64_t);

bool verbose = true;
dim3 threadsPerBlock;
dim3 blocksPerGrid;

threadsPerBlock.x = 32;
threadsPerBlock.y = 1;
threadsPerBlock.z = 1;

blocksPerGrid.x = 8;
blocksPerGrid.y = 1;
blocksPerGrid.z = 1;
void mergesort(double* data, uint64_t size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    double* D_data;
    double* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
    tm();

    HANDLE_ERROR(cudaMallocManaged((void**) &D_data, size * sizeof(double)));
    HANDLE_ERROR(cudaMallocManaged((void**) &D_swp, size * sizeof(double)));
    if (verbose)
        std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    //
    // Copy the thread / block info to the GPU as well
    //
    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

    if (verbose)
        std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    if (verbose)
        std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    double* A = D_data;
    double* B = D_swp;

    uint64_t nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (uint64_t width = 2; width < (size << 1); width <<= 1) {
        uint64_t slices = size / ((nThreads) * width) + 1;

        if (verbose) {
            std::cout << "mergeSort - width: " << width 
                      << ", slices: " << slices 
                      << ", nThreads: " << nThreads << '\n';
            tm();
        }

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        if (verbose)
            std::cout << "call mergesort kernel: " << tm() << " microseconds\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    //
    // Get the list back from the GPU
    //
    // Free the GPU memory
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
    if (verbose)
        std::cout << "cudaFree: " << tm() << " microseconds\n";
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(double* source, double* dest, uint64_t size, uint64_t width, uint64_t slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    uint64_t start = width*idx*slices, 
         middle, 
         end;

    for (uint64_t slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(double* source, double* dest, uint64_t start, uint64_t middle, uint64_t end) {
    uint64_t i = start;
    uint64_t j = middle;
    for (uint64_t k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}