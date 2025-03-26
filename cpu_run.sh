#!/bin/bash

# List of the number of elements to test with
#elements_list=(1342 2684 4026 5368 6710 8054 9396 10738 12080 13400)
elements_list=(5368 8054)

# Filename and number of threads
filename="21000000000_numbers.txt"
workloads=(1)
#thread_nums=(16 24 32 40 48 56 64 72 80)
thread_nums=(1 2 4 8)
> workload_performance_metrics.csv

# Loop through each element in the elements list
for num_elements in "${elements_list[@]}"
do
    for workload in "${workloads[@]}"
    do
        for thread_num in "${thread_nums[@]}"
        do
            echo "Running ./workload with $num_elements elements with $workload cpu workload with $thread_num threads"
            ./workload $filename $num_elements $workload $thread_num
        done
    done
done
