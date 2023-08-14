#!/bin/bash

# This script is used to run the LODL algorithm for 10 times for 10 seeds each and store the results in a file

# make experiment directory with format "year.month.date/hourminute"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

exp_dir=exp/$(date +%Y.%m.%d/%H%M%S)
mkdir -p $exp_dir

for i in {0..7}
do
    for j in 4
    do
        dir=$exp_dir/lodl_$i/$j
        mkdir -p $dir
        python main_old.py --problem=portfolio --loss=quad --seed=$i --randomness=$j --directory=$dir --instances=400 --testinstances=400 \
        --valfrac=0.5 --stocks=50 --stockalpha=0.1 --lr=0.01 --sampling=random --samplingstd=0.1 --numsamples=5000 \
        --losslr=0.001 --serial=False --batchsize=500 --quadalpha=10 > $dir/logs.txt &
    done
done
wait