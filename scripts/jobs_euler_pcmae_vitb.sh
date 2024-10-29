#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

cd /cluster/home/callen/projects/mae_branches/mae_lossC

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G


# DATASETs=(pcmae_cifar10_tvb pcmae_tiny_bvt pcmae_blood_bvt pcmae_path_bvt pcmae_derma_bvt)
DATASETs=(pcmae_cifar10_pc pcmae_tiny_pc pcmae_blood_pc pcmae_path_pc pcmae_derma_pc)
MASKs=(0.05 0.1 0.2 0.3 0.4 0.5)

MODEL="vit-b"
for DATASET in "${DATASETs[@]}"
do
for MASK in "${MASKs[@]}"
do

RUN_TAG=""$DATASET"_pc_tvb_"$MASK"_model_"$MODEL"_lossC"
NAME="/cluster/home/callen/projects/mae_branches/mae_lossC/output_log/$RUN_TAG"
JOB="python main.py user=callen_euler experiment=$DATASET masking.pc_ratio=$MASK run_tag=$RUN_TAG model=$MODEL"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;

