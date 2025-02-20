#!/bin/bash

# List of the number of elements to test with
# elements_list=(100 200 400 800 1600 3200 6400 12800 25600 51200 102400 204800 409600)
elements_list=(204800)

# Filename and number of threads
filename="12800numbers.txt"
workloads=(0 0.2 0.4 0.6 0.8 1)

> workload_performance_metrics.csv

# Loop through each element in the elements list
for num_elements in "${elements_list[@]}"
do
    for workload in "${workloads[@]}"
    do
        echo "Running ./workload with $num_elements elements with $workload cpu workload"
        ./workload $filename $num_elements $workload
    done
done
