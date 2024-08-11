#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
# custom config

DATA=/home/datasets/

datasets=("oxford_flowers" "stanford_cars" "oxford_pets" "food101" "sun397" "eurosat" "ucf101")
attacks=('our')
#targets=("rn18" "eff" "regnet")
SEED=1

for at in "${attacks[@]}"; do
    for DATASET in "${datasets[@]}"; do
        python main.py \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/rn50.yaml \
        --flag eval_adv \
        --dataset ${DATASET} \
        --attack ${at} \
        > logs/eval/${DATASET}.log 2>&1
    done
done