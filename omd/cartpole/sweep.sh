#!/bin/bash
#SBATCH --partition=devlab
#SBATCH --cpus-per-task=6
#SBATCH --time=6:30:00
#SBATCH --array=0-9
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=metric
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/checkpoint/dishank/slurm-%j.out
#SBATCH --error=/checkpoint/dishank/slurm-%j.err

srun --label -u --cpus-per-task=$SLURM_CPUS_PER_TASK --mem-per-cpu=$SLURM_MEM_PER_CPU --output=/checkpoint/dishank/slurm-%A-%a.out --error=/checkpoint/dishank/slurm-%A-%a.err ./wrapper_tmp.sh activation=normalize,softplus seed=5,6,7,8,9 sizes=6-12 regularization_coeff=0.0
# srun --label -u --cpus-per-task=$SLURM_CPUS_PER_TASK --mem-per-cpu=$SLURM_MEM_PER_CPU --output=/checkpoint/dishank/slurm-%A-%a.out --error=/checkpoint/dishank/slurm-%A-%a.err ./wrapper.sh lr=0.001 dim_distract=512 seed=3,9,7 regularization_coeff=0.0