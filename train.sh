#!/bin/sh
PARTITION=Segmentation

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
mkdir -p ${model_dir} ${result_dir}
config=config/${dataset}/${dataset}_${exp_name}.yaml

nohup python -u train.py --config=${config} error.log 2>&1