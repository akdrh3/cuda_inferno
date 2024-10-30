#include "gpu_util.cu"
extern "C"{
#include "util.h"
}

void mergesort(){

}

void mergesortkernel(){

}

void main(int argc, char *argv[]){
    //read file
    const char file_name = argv[1];
    uint64_t size_of_array = strtoull(argv[2], NULL, 10);

    printf("file name %s, size of the array : %llu", file_name, size_of_array);


    //different thread number

    //launch mergesort kernel
    //measure time

}

