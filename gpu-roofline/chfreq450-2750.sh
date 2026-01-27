#!/usr/bin/bash

conda activate /home/ub/data/miniconda3/envs/torch_env

sleep 600

for ((clock=450; clock<=2750; clock+=150))
do
    sudo nvidia-smi -i 0 -lgc $clock,$clock
    sed -i "s/rtxpro4500-[^.]*\.txt/rtxpro4500-${clock}.txt/g" series.sh
    make
    ./series.sh cu-roof
    sleep 600
done
