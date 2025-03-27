#!/bin/bash

# Loop over directories INS-W_1 to INS-W_4
for i in {1..4}; do
    base_dir="/home/willkewang/Datasets/GLOBEM/INS-W_$i"
    echo "Processing directory: $base_dir"
    
    # Find all files under the current directory. Using -print0 to handle spaces in file names.
    find "$base_dir" -type f -print0 | while IFS= read -r -d '' file; do
        # Read the first line (header) of the file
        header=$(head -n 1 "$file")
        
        # Check if the header contains a column name that starts with f_call or f_blue.
        # The regex (^|,)(f_call|f_blue) checks if either at the beginning of the line or following a comma,
        # a column starting with f_call or f_blue exists.
        if echo "$header" | grep -qE '(^|,)(f_call|f_blue)'; then
            # Compute the file's path relative to the base_dir for a cleaner output.
            rel_path="${file#$base_dir/}"
            echo "Found in INS-W_$i: $rel_path"
        fi
    done
done