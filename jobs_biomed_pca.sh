#!/bin/bash 

conda activate reconstruction 

cd /cluster/home/abizeul/mae


NUM_WORKERS=8
TIME=4:00:00
RATIO=0.99
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_"$RATIO""
JOB="python ./notebooks/pc_filtering_top.py  --ratio "$RATIO"  --split "val" --root /cluster/work/vogtlab/Group/abizeul/data/ --datadir /cluster/work/vogtlab/Group/abizeul/data/tiny-imagenet-200 --dataset tinyimagenet --existing_eigen /cluster/work/vogtlab/Group/abizeul/data/tinyimagenet_"$RATIO"_percent_top/pc_train/pc_train.joblib --existing_mean /cluster/work/vogtlab/Group/abizeul/data/tinyimagenet_"$RATIO"_percent_top/pc_train/mean.npy --existing_std /cluster/work/vogtlab/Group/abizeul/data/tinyimagenet_"$RATIO"_percent_top/pc_train/std.npy"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" --wrap="nvidia-smi;$JOB"