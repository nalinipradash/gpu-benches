#!/bin/bash



conda activate /home/devshree/data/miniconda3/envs/torch_env


for ((clock=450; clock<=2600; clock+=150))
do
    sudo pkill -u jothi
    nvidia-smi --query-gpu=name,clocks.current.graphics,clocks.current.memory --format=csv
    sudo nvidia-smi -i 0 -lgc $clock,$clock
    sed -i "s/l40s-[^.]*\.txt/l40s-${clock}.txt/g" series.sh
    make
    ./series.sh cu-roof
    sleep 600
done
