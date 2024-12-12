#!/bin/bash

# Check if the input file exists
if [[ ! -f "16000numbers.txt" ]]; then
    echo "Error: Input file 16000numbers.txt not found."
    exit 1
fi

# numbers=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
# Initialize variables
numbers=(500 1000 2000 3000 4000 5000)

# Clear log files
> thrust_merge_error_log.txt
> output.txt

# Loop through numbers and threads
for number in "${numbers[@]}"
do
    output=$(./thrust_merge 16000numbers.txt ${number} 2>> thrust_merge_error_log.txt)
    
    # Check if ./mergesort succeeded
    if [[ $? -ne 0 ]]; then
        echo "Error: failed for number=${number}" >> thrust_merge_error_log.txt
    fi

    # Log output
    echo "$output"
    echo "$output" >> output.txt
    echo "--------------------------------------"
    echo "--------------------------------------" >> output.txt
done

# Print merge_error_log.txt at the end
echo -e "\nContent of thrust_merge_error_log.txt:"
cat thrust_merge_error_log.txt