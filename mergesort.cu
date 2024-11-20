#include "gpu_util.cuh"
extern "C"{
#include "util.h"
}
#include <cmath>
#include <climits>
#include <stdio.h>

int isRangeSorted(int *arr, size_t start, size_t end)
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

void swap_int_pointer(int **arr_A, int **arr_B, bool *flipped){
    //printf("swapping\n");
    int *tmp_pointer=*arr_A;
    *arr_A = *arr_B;
    *arr_B = tmp_pointer;
    *flipped = !(*flipped);
    //printf("swapped pointer \n\n");
}

__global__ void mergesortKernel(int* arr, int* tmp, uint64_t size_of_array, uint64_t segment_size){
    //getting tid, start, mid, and end index
    uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;  

    // based on tide, devide and get the portion of the array that this specific tid has to work on
    // everything is in index number
    uint64_t start = tid * segment_size * 2; //last tid * segment_size = size_of_array - segment_size

    // Ignore out-of-bounds threads
    if (start > size_of_array -1){
        //printf("out of bound thread happening even though it should not in very frist step unless the array size is smaller than 256.");
        return;  
    }

    //mid means array_B starting index
    uint64_t mid = min(start + segment_size, size_of_array -1);
    //end means array_B ending index
    uint64_t end = min(start + segment_size*2 - 1, size_of_array -1);

    if(start < end){
        uint64_t array_a_index = start, array_b_index = mid, temp_index = start;
       //printf("inside merge; index1 : %lu, index2 : %lu, tmp index: %lu, end: %lu\n", start, mid, start, end);
        while (array_a_index < mid && array_b_index <= end){
            if (arr[array_a_index] <= arr[array_b_index]){
                tmp[temp_index++] = arr[array_a_index++];
            } 
            else{
                tmp[temp_index++] = arr[array_b_index++];
            }
        }

        while (array_a_index < mid){
            tmp[temp_index++] = arr[array_a_index++];
        }

        while (array_b_index <= end){
            tmp[temp_index++] = arr[array_b_index++];
        } 
    }


}

__global__ void initial_merge(int* arr, int* tmp, uint64_t size_of_array, uint64_t segment_size)
{
    //getting tid, start, mid, and end index
    uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    // based on tide, devide and get the portion of the array that this specific tid has to work on
    // everything is in index number
    uint64_t block_start = tid * segment_size; //last tid * segment_size = size_of_array - segment_size

    // Ignore out-of-bounds threads
    if (block_start > size_of_array -1){
        //printf("out of bound thread happening even though it should not in very frist step unless the array size is smaller than 256.");
        return;  
    }
    uint64_t block_end = min(block_start + segment_size - 1, size_of_array -1);
    //printf("tid : %lu, segmentSize : %lu, blockStart : %lu, blockEnd : %lu\n ", tid, segment_size, block_start, block_end);

    uint64_t curr_size = 1, left_start = 0;
    //keep doubling the curr_size
    for (curr_size = 1; curr_size < segment_size; curr_size *= 2){
        for(left_start = block_start; left_start <= block_end; left_start += 2 * curr_size){
            uint64_t subarray_middle_index = min(left_start + curr_size, block_end);
            uint64_t right_end = min(left_start + 2 * curr_size -1, block_end);   
            //printf("curr_size: %lu, array_a_index: %lu, array_b_index: %lu, end: %lu\n", curr_size, left_start, subarray_middle_index, right_end);
            if(subarray_middle_index <= right_end){
                uint64_t array_a_index = left_start, array_b_index = subarray_middle_index, temp_index = left_start, end = right_end;
                while (array_a_index < subarray_middle_index && array_b_index <= end){
                    if (arr[array_a_index] <= arr[array_b_index]){
                        tmp[temp_index++] = arr[array_a_index++];
                    } 
                    else{
                        tmp[temp_index++] = arr[array_b_index++];
                    }
                }

                while (array_a_index < subarray_middle_index){
                    tmp[temp_index++] = arr[array_a_index++];
                }

                while (array_b_index <= right_end){
                    tmp[temp_index++] = arr[array_b_index++];
                }  

                // Now copy the sorted elements from tmp back to the original array 'arr'
                for (uint64_t i = left_start; i <= right_end; i++) {
                    arr[i] = tmp[i];
                }
            }
        }
    }
    if (isRangeSorted(&arr, block_start, block_end) == 1){
        printf("tid : %lu, start: %lu, end : %lu, sorted well\n", tid, block_start, block_end);
        return;
    }
    printf("tid : %lu, start: %lu, end : %lu, not sorted well!\n", tid, block_start, block_end);
    return; 
}

void mergesort(int *arr, int *tmp, uint64_t size_of_array, int number_of_thread){
    //calculate the array size that initially goes into each thread
    uint64_t segment_size = (uint64_t)ceil((double)size_of_array / number_of_thread);
    bool flipped = false;
    

    //sort the smallest portion of the sorting
    printf("initial merge segment_size : %lu\n",segment_size);
    initial_merge<<<1, number_of_thread>>>(arr, tmp, size_of_array, segment_size);
    HANDLE_ERROR(cudaDeviceSynchronize());
    //now it is sure that the smallest segments are sorted
    //inter-segments mergesort
    if (number_of_thread == 1){
        return;
    }

    segment_size *= 2;

    while (segment_size < size_of_array*2){
        printf("mergesort kernel segment_size : %lu\n",segment_size);
        mergesortKernel<<<1, number_of_thread>>>(arr, tmp, size_of_array, segment_size);
        HANDLE_ERROR(cudaDeviceSynchronize());
        swap_int_pointer(&arr, &tmp, &flipped);
        segment_size *= 2;
    }

    if (flipped == true){
        swap_int_pointer(&arr, &tmp, &flipped);
    }
}


int main(int argc, char *argv[]){
    //get param from command; filename , arraysize * 1 million
    const char *file_name = argv[1];
    uint64_t size_of_array = strtoull(argv[2], NULL, 10)*1000000;
    int number_of_thread;
    if (sscanf(argv[3], "%d", &number_of_thread) != 1) 
    {
        printf("Invalid number: %s\n", argv[3]);
        return 1; // Exit with an error code
    }
    int *gpu_array = NULL;
    int *gpu_tmp = NULL;
    HANDLE_ERROR(cudaMallocManaged((void**)&gpu_array, size_of_array * sizeof(int)));
    HANDLE_ERROR(cudaMallocManaged((void **)&gpu_tmp, size_of_array * sizeof(int)));
    cudaEvent_t start, stop;

    //different thread number
    //int thread_numbers[5] = {128, 320, 384, 448, 576};
    //int thread_numbers[5] = {1, 2, 3, 4, 5};

    //size of the array
    double array_size_in_GB = SIZE_IN_GB(sizeof(int)*size_of_array);
    printf("Data Set Size: %f GB Number of integers : %lu number of threads : %d\n", array_size_in_GB, size_of_array, number_of_thread);

    //read from file and store it to gpu_array
    read_from_file(file_name, gpu_array, size_of_array);

    //start timer
    cuda_timer_start(&start, &stop);

    //call mergesort function
    mergesort(gpu_array, gpu_tmp, size_of_array, number_of_thread);

    // Stop timer
    double gpu_sort_time = cuda_timer_stop(start, stop);
    double gpu_sort_time_sec = gpu_sort_time / 1000.0;

    printf("Time elapsed for merge sort with %d threads: %lf s\n", number_of_thread, gpu_sort_time_sec);
    // print_array_host(gpu_array, size_of_array);
    // print_array_host(gpu_tmp, size_of_array);
    // printf("-------------------------------------------------\n");
    // //run mergesort 5 times
    // for (int i = 0; i < 5; ++i){
    //     int number_of_thread = thread_numbers[i];

    //     //read from file and store it to gpu_array
    //     read_from_file(file_name, gpu_array, size_of_array);
 
    //     //start timer
    //     cuda_timer_start(&start, &stop);

    //     //call mergesort function
    //     mergesort(gpu_array, gpu_tmp, size_of_array, number_of_thread);

    //     // Stop timer
    //     double gpu_sort_time = cuda_timer_stop(start, stop);
    //     double gpu_sort_time_sec = gpu_sort_time / 1000.0;

    //     printf("Time elapsed for merge sort with %d threads: %lf s\n", number_of_thread, gpu_sort_time_sec);
    //     // print_array_host(gpu_array, size_of_array);
    //     // print_array_host(gpu_tmp, size_of_array);
    //     // printf("-------------------------------------------------\n");
    // }


    //free pointers
    HANDLE_ERROR(cudaFree(gpu_tmp));
    HANDLE_ERROR(cudaFree(gpu_array));      

    return 0;

}

