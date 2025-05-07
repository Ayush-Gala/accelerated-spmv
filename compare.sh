#!/bin/bash

# Usage: ./compare_files.sh file1.txt file2.txt

file1="$1"
file2="$2"
line_num=1
found_diff=0
max_diff=1

while IFS= read -r line1 && IFS= read -r line2 <&3
do
    if [ "$line1" != "$line2" ]; then
        diff=$(($line1 - $line2))
        if [ "$diff" -gt $max_diff ]; then
            echo "Mismatch at line $line_num:"
            echo "File1: $line1"
            echo "File2: $line2"
            found_diff=1
            exit 1
        fi
        diff=$(($line2 - $line1))
        if [ "$diff" -gt $max_diff ]; then
            echo "Mismatch at line $line_num:"
            echo "File1: $line1"
            echo "File2: $line2"
            found_diff=1
            exit 1
        fi
    fi
    ((line_num++))
done < "$file1" 3< "$file2"

# Check for different file lengths
if [ $found_diff -eq 0 ]; then
    if [ $(wc -l < "$file1") -ne $(wc -l < "$file2") ]; then
        echo "Files have different number of lines but matching lines are identical"
    else
        echo "Files are identical"
    fi
fi
