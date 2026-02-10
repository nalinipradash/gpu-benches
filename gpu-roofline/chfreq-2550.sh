#!/bin/bash



conda activate /home/nalini/data/miniconda3/envs/diffusion

# nvidia-smi --query-gpu=name,clocks.current.graphics,clocks.current.memory --format=csv

# sudo nvidia-smi -i 0 -rgc
sleep 240

sudo nvidia-smi -i 0 -lgc 2550,2550

sed -i 's/rtxpro4000-[^.]*\.txt/rtxpro4000-2550.txt/g' series.sh

make

./series.sh cu-roof
