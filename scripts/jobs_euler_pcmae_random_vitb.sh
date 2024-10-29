#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

cd /cluster/home/callen/projects/mae_branches/mae_lossC

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G

DATASETs=(pcmae_tiny_pcsampling pcmae_cifar10_pcsampling pcmae_path_pcsampling pcmae_blood_pcsampling pcmae_derma_pcsampling)
# DATASETs=(pcmae_clevr_pcsampling)

MODEL="vit-b"
for DATASET in "${DATASETs[@]}"
do
RUN_TAG=""$DATASET"_model_"$MODEL"_lossC"
NAME="/cluster/home/callen/projects/mae_branches/mae_lossC/output_log/$RUN_TAG"
JOB="python main.py user=callen_euler run_tag=$RUN_TAG experiment=$DATASET model=$MODEL"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done;
