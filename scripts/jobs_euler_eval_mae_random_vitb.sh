#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

cd /cluster/home/callen/projects/mae

NUM_WORKERS=8
TIME=1:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=12G

# DATASETs=(mae_sampling_tiny mae_sampling_blood mae_sampling_path mae_sampling_derma mae_sampling_cifar10)
DATASETs=(mae_sampling_tiny mae_sampling_path)

MODEL="vit-b"
# EPOCHs=(100 200 300 400 500 600 700 800)
EPOCHs=(800)

for DATASET in "${DATASETs[@]}"
do
for EPOCH in "${EPOCHs[@]}"
do
RUN_TAG=""$DATASET"_eval_"$EPOCH"_model_"$MODEL"_lin"
NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=callen_euler experiment=$DATASET trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH model=$MODEL run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

RUN_TAG=""$DATASET"_eval_"$EPOCH"_model_"$MODEL"_knn"
NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=callen_euler experiment=$DATASET trainer=eval_knn checkpoint=pretrained checkpoint.epoch=$EPOCH model=$MODEL run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;
