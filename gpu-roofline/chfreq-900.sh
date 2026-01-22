#!/bin/bash


sudo pkill -u jothi

conda activate /home/devshree/data/miniconda3/envs/torch_env

nvidia-smi --query-gpu=name,clocks.current.graphics,clocks.current.memory --format=csv

# sudo nvidia-smi -i 0 -rgc

sudo nvidia-smi -i 0 -lgc 900,900

sed -i 's/l40s-[^.]*\.txt/l40s-900.txt/g' series.sh

make

./series.sh cu-roof
