#!/bin/bash

# List of the number of elements to test with
elements_list=(100 200 400 800 1600 3200 6400 12800 25600 51200 102400 204800 409600)

# Filename and number of threads
filename="12800numbers.txt"
threads=16

# Loop through each element in the elements list
for num_elements in "${elements_list[@]}"
do
    echo "Running ./cpu with $num_elements elements"
    ./cpu $filename $num_elements $threads
done
