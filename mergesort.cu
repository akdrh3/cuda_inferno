#include "gpu_util.cuh"
extern "C"{
#include "util.h"
}

__device__ void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

void swap_int_pointer(int **arr_A, int **arr_B, bool flipped){
    //printf("swapping\n");
    int *tmp_pointer=*arr_A;
    *arr_A = *arr_B;
    *arr_B = tmp_pointer;
    flipped = !flipped;
    //printf("swapped pointer \n\n");
}


void mergesortkernel(){

}

void mergesort(){

}

__global__ void initial_merge(int* arr, int* tmp, uint64_t size_of_array, uint64_t segment_size, bool flipped)
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
                // for (uint64_t i = left_start; i <= right_end; i++) {
                //     arr[i] = tmp[i];
                // }
                swap_int_pointer(int **arr_A, int **arr_B, bool flipped);
                printf("flipped: %d", flipped);
            }
        }
    }
    return; 
}


int main(int argc, char *argv[]){
    //get param from command; filename , arraysize * 1 million
    const char *file_name = argv[1];
    uint64_t size_of_array = strtoull(argv[2], NULL, 10);

    int *gpu_array = NULL;
    int *gpu_tmp = NULL;
    HANDLE_ERROR(cudaMallocManaged((void**)&gpu_array, size_of_array * sizeof(int)));
    HANDLE_ERROR(cudaMallocManaged((void **)&gpu_tmp, size_of_array * sizeof(int)));
    bool flipped = false;
    cudaEvent_t start, stop;

    //different thread number
    int thread_numbers[5] = {1, 256, 512, 768, 1024};

    //size of the array
    double array_size_in_GB = SIZE_IN_GB(sizeof(int)*size_of_array);

    //run mergesort 5 times
    for (int i = 0; i < 5; ++i){
        int number_of_thread = thread_numbers[i];

        //read from file and store it to gpu_array
        read_from_file(file_name, gpu_array, size_of_array);
        print_array_host(gpu_array, size_of_array);

        //calculate the array size that initially goes into each thread
        uint64_t segment_size = (uint64_t)ceil((double)size_of_array / number_of_thread);
        
        //start timer
        cuda_timer_start(&start, &stop);

        //sort the smallest portion of the sorting
        initial_merge<<<1, number_of_thread>>>(gpu_array, gpu_tmp, size_of_array, segment_size, flipped);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Stop timer
        double gpu_sort_time = cuda_timer_stop(start, stop);
        double gpu_sort_time_sec = gpu_sort_time / 1000.0;

        printf("Time elapsed for merge sort with %d threads: %lf s\n", number_of_thread, gpu_sort_time_sec);
        print_array_host(gpu_array, size_of_array);
        printf("-------------------------------------------------\n");
    }


    //free pointers
    HANDLE_ERROR(cudaFree(gpu_tmp));
    HANDLE_ERROR(cudaFree(gpu_array));      

    return 0;

}

