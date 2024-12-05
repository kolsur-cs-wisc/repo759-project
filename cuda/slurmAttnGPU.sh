#!/usr/bin/env zsh

#SBATCH --partition=instruction
#SBATCH --job-name=attngpu
#SBATCH --output=attngpu.out
#SBATCH --time 00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1

cd $SLURM_SUBMIT_DIR
nvcc cuda_attention_evaluation.cu matmul.cu softmax.cu attention.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o attn

for i in {1..10};
do
    n=$((2**$i))
    echo "2^$i $n"
    ./attn 128 $n 512 8
done
echo "Attn GPU Finished"