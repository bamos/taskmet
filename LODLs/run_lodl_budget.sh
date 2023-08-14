#!/bin/bash

# This script is used to run the LODL algorithm for 10 times for 10 seeds each and store the results in a file

# make experiment directory with format "year.month.date/hourminute"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

exp_dir=exp/$(date +%Y.%m.%d/%H%M%S)
mkdir -p $exp_dir

for i in {8..9}
do
    for j in {0..4}
    do
        dir=$exp_dir/lodl_$i/$j
        mkdir -p $dir
        python main_old.py --problem=budgetalloc --loss=quad --seed=$i --randomness=$j --directory=$dir --instances=100 --testinstances=500 \
        --valfrac=0.2 --numitems=5 --budget=2 --sampling=random --numsamples=5000 --losslr=0.1 --serial=False --batchsize=500 --quadalpha=10 > $dir/logs.txt &
    done
done
wait