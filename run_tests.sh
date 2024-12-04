#!/bin/bash

# Check if the input file exists
if [[ ! -f "16000numbers.txt" ]]; then
    echo "Error: Input file 16000numbers.txt not found."
    exit 1
fi

# numbers=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
# Initialize variables
numbers=(256 512 1024 2048)
threads=(256)

# Clear log files
> merge_error_log.txt
> output.txt

# Loop through numbers and threads
for number in "${numbers[@]}"
do
    for thread in "${threads[@]}"
    do
        output=$(./mergesort 16000numbers.txt ${number} ${thread} 2>> merge_error_log.txt)
        
        # Check if ./mergesort succeeded
        if [[ $? -ne 0 ]]; then
            echo "Error: ./mergesort failed for number=${number}, thread=${thread}" >> merge_error_log.txt
        fi

        # Log output
        echo "$output"
        echo "$output" >> output.txt
        echo "--------------------------------------"
        echo "--------------------------------------" >> output.txt
    done
done

# Print merge_error_log.txt at the end
echo -e "\nContent of merge_error_log.txt:"
cat merge_error_log.txt