#!/usr/bin/env zsh

nvcc cuda_attention_evaluation.cu matmul.cu softmax.cu attention.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o attn

for i in {1..11};
do
    n=$((2**$i))
    echo "$n"
    ./attn 64 $n 512 8
done
echo "Attn GPU Finished"
