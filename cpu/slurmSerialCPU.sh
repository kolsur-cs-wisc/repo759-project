#!/usr/bin/env zsh

#SBATCH --partition=instruction
#SBATCH --job-name=serialcpu
#SBATCH --output=serial.out
#SBATCH --time 00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd $SLURM_SUBMIT_DIR
g++ cpu_evaluation.cpp attention.cpp matmul.cpp softmax.cpp -Wall -O3 -std=c++17 -o serial

for i in {1..10};
do
    n=$((2**$i))
    echo "2^$i $n"
    ./serial 128 $n 512 8
done
echo "Serial CPU Finished"
