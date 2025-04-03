#!/bin/bash

#SBATCH --job-name=RedPatterns
#SBATCH --output=RedPatterns.slurm.%N.%J.%u.%a.out # STDOUT
#SBATCH --error=RedPatterns.slurm.%N.%J.%u.%a.err # STDERR
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=6GB
#SBATCH --partition=a100ai
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH -A nhr-rbc-pattern
#SBATCH --array=1-1:1 #(start on one, end on TOTAL NUMBER NODES, increment 1 each time)

module use /apps/easybuild/current/cuda/modules/all
module load tools/parallel
module load system/CUDA
module load lang/Python
module load compiler/GCC

mkdir parameters_$SLURM_ARRAY_TASK_ID
cp ./red_patterns parameters_$SLURM_ARRAY_TASK_ID/.
cp ./run_parameter_list.py parameters_$SLURM_ARRAY_TASK_ID/.
mv ./parameters_$SLURM_ARRAY_TASK_ID.txt parameters_$SLURM_ARRAY_TASK_ID/.
cd parameters_$SLURM_ARRAY_TASK_ID
python3 run_parameter_list.py parameters_$SLURM_ARRAY_TASK_ID.txt