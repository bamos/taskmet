#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

exp_dir=exp/$(date +%Y.%m.%d/%H%M)
mkdir -p $exp_dir

for i in {4..7}
do
    for j in {0..4}
    do
        dir=$exp_dir/.logs
        mkdir -p $dir
        python main.py dataset_seed=$i seed=$j method=dfl problem=cubic loss_kwargs.dflalpha=10.0 > $dir/${i}_$j.txt &
    done
done
wait