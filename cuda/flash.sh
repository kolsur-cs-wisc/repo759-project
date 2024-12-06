#!/usr/bin/env zsh

nvcc cuda_flashattention_evaluation.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o flash

for i in {1..11};
do
    n=$((2**$i))
    echo "$n"
    ./flash 64 $n 512 8
done
echo "Flash GPU Finished"
