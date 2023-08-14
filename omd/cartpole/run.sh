#!/bin/bash
#SBATCH --partition=learnfair
#SBATCH --job-name=metric
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=3
#SBATCH --time=8:00:00
#SBATCH --output=/checkpoint/dishank/slurm-%j.out
#SBATCH --error=/checkpoint/dishank/slurm-%j.err
#SBATCH --mem-per-cpu=1G

trap "kill 0" EXIT

# get first and second argument from command line and store it in exp and agent variable
# exp=$1
# agent=$2

exp="upper-bound-final-dim16"
agent="metric"

for i in {0..4}
do
    outdir="exp/$exp/$agent/$i"
    mkdir -p $outdir
    echo "Running $agent with seed $i"
    srun --label -u -n1 --exclusive -c $SLURM_CPUS_PER_TASK --mem-per-cpu $SLURM_MEM_PER_CPU python main.py seed=$i agent_type=$agent exp=$exp warm_opt=True dim_distract=16 > $outdir/log.txt &
done

wait