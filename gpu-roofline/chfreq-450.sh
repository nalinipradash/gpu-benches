#!/usr/bin/bash


conda activate /home/ub/data/miniconda3/envs/torch_env

# sudo nvidia-smi -i 0 -rgc

sudo nvidia-smi -i 0 -lgc 450,450

sed -i 's/rtxpro4500-[^.]*\.txt/rtxpro4500-450.txt/g' series.sh

make

./series.sh cu-roof
