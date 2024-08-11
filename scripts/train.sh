#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# custom config

DATA=/home/dataset

datasets=("oxford_flowers" "stanford_cars" "oxford_pets" "food101" "sun397" "eurosat" "ucf101")
targets=("rn18" "eff" "regnet")
SEED=1

for t in "${targets[@]}"; do
    for DATASET in "${datasets[@]}"; do
        python main.py \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/rn50.yaml \
        --flag train_scratch \
        --num_epoch 300 \
        --target ${t} \
        --dataset ${DATASET} \
        --lr 0.1 \
        --bs 128 \
        > logs/train/${t}.log 2>&1
    done
done