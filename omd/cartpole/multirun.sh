#!/bin/bash

# Bash script to run sweep of cartpole experiments with arguments to the python script taking multiple values
# Usage: ./multirun.sh scipt.py --arg1=val1,..valt1 --arg2=val1,....valt2 --argn=val1,....,valtn
# This should run python script t1*t2*..tn times corresponding to all combinations of the values of the arguments
# Each run should have output directory with template: exp/$random_string
# Each experiment is ran as python scipy.py --arg1=val1 --arg2=val2 --argn=valn

# Example: ./multirun.sh cartpole.py --num_runs=10 --num_episodes=100,200 --num_steps=100,200
# This should run 4 experiments with the following arguments:
# python cartpole.py --num_runs=10 --num_episodes=100 --num_steps=100
# python cartpole.py --num_runs=10 --num_episodes=100 --num_steps=200
# python cartpole.py --num_runs=10 --num_episodes=200 --num_steps=100
# python cartpole.py --num_runs=10 --num_episodes=200 --num_steps=200

# Parse the command line arguments
script=$1
shift
args=()
for arg in "$@"
do
    args+=("$arg")
done

# Calculate the number of experiments
num_experiments=1
for arg in "${args[@]}"
do
    values=$(echo "$arg" | cut -d'=' -f2)
    num_values=$(echo "$values" | tr ',' '\n' | wc -l)
    num_experiments=$((num_experiments * num_values))
done

# create exp directory as: exp/$hour-$minute-$second
time=$(date +"%H-%M-%S")
output_dir="exp/$time"
mkdir -p $output_dir
echo "Output directory: $output_dir"

trap "kill 0" EXIT
# Loop through all combinations of argument values and run the script

for ((i=0; i<$num_experiments; i++))
do
    # Create the output directory
    output_dir="exp/$time/$i"
    mkdir -p $output_dir
    # Create the command to run the script
    command="python $script --out_dir=$output_dir"
    for arg in "${args[@]}"
    do
        name=$(echo "$arg" | cut -d'=' -f1)
        values=$(echo "$arg" | cut -d'=' -f2)
        value=$(echo "$values" | tr ',' '\n' | head -n $((i % $(echo "$values" | tr ',' '\n' | wc -l) + 1)) | tail -n 1)
        command="$command $name=$value"
    done
    # Run the command
    echo "Running: $command"
    $command > $output_dir/log.txt 2>&1 &
done

# Wait for all experiments to finish
wait