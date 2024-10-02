#!/bin/bash 

cd ~/projects/mae

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G

DATASETs=(pcmae_cifar10_pc pcmae_cifar10_pcsampling )

for DATASET in "${DATASETs[@]}"
do
RUN_TAG="$DATASET"
NAME="~/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=myusername_mymachine experiment=$DATASET run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done;
