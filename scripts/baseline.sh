#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
# custom config

DATA=/home/datasets/

datasets=("oxford_pets")
# datasets=("oxford_flowers" "stanford_cars" "oxford_pets" "food101" "sun397" "eurosat" "ucf101")
attacks=('l2t' 'ags' 'sia' 'decowa' 'bsr' 'ilpd' 'mta' 'cwa')
SEED=1

for at in "${attacks[@]}"; do
    for DATASET in "${datasets[@]}"; do
        python baseline.py \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp/rn50.yaml \
        --dataset ${DATASET} \
        --attack ${at} \
        --bs 10 \
        --model resnet50 \
        >> logs/baseline/${at}.log 2>&1
    done
done
#
# mta : resnet_MTA
# cwa : resnet50,resnet34
