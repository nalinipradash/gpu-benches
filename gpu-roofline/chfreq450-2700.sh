#!/bin/bash
echo "little nap"
sleep 600
echo "nap ended"

conda activate /home/nalini/data/miniconda3/envs/diffusion


for ((clock=450; clock<=2700; clock+=150))
do
    nvidia-smi --query-gpu=name,clocks.current.graphics,clocks.current.memory --format=csv
    sudo nvidia-smi -i 0 -lgc $clock,$clock
    sed -i "s/rtxpro4000-[^.]*\.txt/rtxpro4000-${clock}.txt/g" series.sh
    make
    ./series.sh cu-roof
    echo "little nap"
    sleep 600
    echo "nap ended"
done
