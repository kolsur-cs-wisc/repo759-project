#!/usr/bin/env zsh

#SBATCH --partition=instruction
#SBATCH --job-name=parallelcpu
#SBATCH --output=parallel.out
#SBATCH --time 00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

cd $SLURM_SUBMIT_DIR
g++ cpu_evaluation.cpp attention.cpp matmul.cpp softmax.cpp -Wall -O3 -std=c++17 -o parallel -fopenmp

for i in {1..10};
do
    n=$((2**$i))
    echo "2^$i $n"
    ./parallel 128 $n 512 8 16
done
echo "Parallel CPU Finished"