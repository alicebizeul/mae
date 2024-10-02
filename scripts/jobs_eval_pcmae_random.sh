#!/bin/bash 

cd ~/projects/mae

NUM_WORKERS=8
TIME=1:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=12G

##### MAE Baselines patch 8
DATASETs=(pcmae_cifar10_pc pcmae_cifar10_pcsampling)
EPOCHs=(800)

for DATASET in "${DATASETs[@]}"
do
for EPOCH in "${EPOCHs[@]}"
do
RUN_TAG=""$DATASET"_eval_"$EPOCH"_lin"
NAME="~/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=myusername_mymachine experiment=$DATASET trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

RUN_TAG=""$DATASET"_eval_"$EPOCH"_knn"
NAME="~/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=myusername_mymachine experiment=$DATASET trainer=eval_knn checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;
