#!/usr/bin/env zsh

#SBATCH --partition=instruction
#SBATCH --job-name=flashgpu
#SBATCH --output=flashgpu.out
#SBATCH --time 00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1

cd $SLURM_SUBMIT_DIR
nvcc cuda_flashattention_evaluation.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o flash

for i in {1..10};
do
    n=$((2**$i))
    echo "2^$i $n"
    ./flash 128 $n 512 8
done
echo "Flash GPU Finished"