#!/bin/bash

# This script is used to run the LODL algorithm for 10 times for 10 seeds each and store the results in a file

# make experiment directory with format "year.month.date/hourminute"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

exp_dir=exp/$(date +%Y.%m.%d/%H%M%S)
mkdir -p $exp_dir

for i in {0..3}
do
    for j in 4
    do
        dir=$exp_dir/lodl_$i/$j
        mkdir -p $dir
        python main_old.py --problem=cubic --loss=quad --seed=$i --randomness=$j --directory=$dir --instances=400 --testinstances=400 --valfrac=0.5 \
        --numitems=50 --budget=1 --lr=0.1 --layers=1 --sampling=random --samplingstd=1 --numsamples=5000 --batchsize=500 \
        --losslr=0.01 --serial=False --quadalpha=10 > $dir/logs.txt &
    done
done
wait