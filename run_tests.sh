#!/bin/bash

numbers=(128 256 512 1024 2048)

> merge_error_log.txt
> output.txt

for number in "${numbers[@]}"
do
    output=$(./mergesort numbers.txt ${number} 2>> merge_error_log.txt)
    echo "$output"
    echo "$output" >> output.txt
    echo "--------------------------------------"
    echo "--------------------------------------" >> output.txt
done