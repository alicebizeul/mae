#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

cd /cluster/home/callen/projects/mae

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G

# Be carefull at epochs, workers and batch size

##### MAE Baselines patch 16
# DATASETs=(mae_sampling_tiny mae_sampling_blood mae_sampling_path mae_sampling_derma mae_sampling_cifar10)
DATASETs=(mae_sampling_clevr)

MODEL="vit-b"
for DATASET in "${DATASETs[@]}"
do
RUN_TAG=""$DATASET"_model_"$MODEL""
NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=callen_euler experiment=$DATASET run_tag=$RUN_TAG model=$MODEL"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done;
