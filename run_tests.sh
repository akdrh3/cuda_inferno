#!/bin/bash
# Save this as runAllTests.sh

# Array of numbers
numbers=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304)

# Clear the output file at the beginning
> quick_error_log.txt
> output.txt

# Iterate over each number in the array
for number in "${numbers[@]}"
do
    # Run numberGenerate.py with the current number
    echo "Running numberGenerate.py with $number"
    echo $number | python3 numberGenerate.py

    # Run testQuickSort with numbers.txt and capture output
    #output=$(echo "numbers.txt" | ./testQuickSort 2>> quick_error_log.txt)

    # Run mergesort with numbers.txt and capture output
     output=$(echo "numbers.txt" | ./mergesort 2>> merge_error_log.txt)

    # Get the file size in bytes
    file_size_bytes=$(stat -c%s "numbers.txt")

    # Convert bytes to gigabytes
    file_size_gb=$(echo "scale=6; $file_size_bytes / (1024*1024*1024)" | bc)

    # Print and save output with file size in GB
    echo "File size: $file_size_gb GB"
    echo "File size: $file_size_gb GB" >> output.txt

    # Print and save output
    echo "$output"
    echo "$output" >> output.txt

    # Separate outputs with a line
    echo "--------------------------------------"
    echo "--------------------------------------" >> output.txt
done