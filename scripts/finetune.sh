#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
# custom config

#数据集路径
DATA=/home/datasets/
TRAINER=ZeroshotCLIP

datasets=("oxford_flowers" "stanford_cars" "oxford_pets" "food101" "sun397" "eurosat" "ucf101")
SEED=1

for DATASET in "${datasets[@]}"; do
    python main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/rn50.yaml \
    --flag finetune \
    --num_epoch 300 \
    --dataset ${DATASET} \
    --lr 0.01 \
    --bs 64 \
    >> logs/finetune/${DATASET}.log 2>&1
done
