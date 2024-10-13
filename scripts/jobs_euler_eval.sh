#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

cd /cluster/home/callen/projects/mae

NUM_WORKERS=8
TIME=4:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=12G

##### MAE Baselines patch 8
DATASETs=(mae_cifar10 mae_blood mae_derma mae_path mae_tiny)
# DATASETs=(mae_clevr)

# MASKs=(0.5 0.6 0.75 0.8 0.9)
# MASKs=(0.75)
# EPOCHs=(100 200 300 400 500 600 700)
EPOCHs=(800)

for DATASET in "${DATASETs[@]}"
do
# for MASK in "${MASKs[@]}"
# do
for EPOCH in "${EPOCHs[@]}"
do

# if [ "$DATASET" == "mae_cifar10" ]; then
#     MASK=0.8
# elif [ "$DATASET" == "mae_tiny" ]; then
#     MASK=0.9
# elif [ "$DATASET" == "mae_blood" ]; then
#     MASK=0.6
# elif [ "$DATASET" == "mae_derma" ]; then
#     MASK=0.8
# elif [ "$DATASET" == "mae_path" ]; then
#     MASK=0.9
# fi

MASK=0.75

# RUN_TAG=""$DATASET"_pixel_"$MASK"_eval_"$EPOCH"_lin"
# NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
# JOB="python main.py user=callen_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

RUN_TAG=""$DATASET"_pixel_"$MASK"_eval_"$EPOCH"_mlp"
NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=callen_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_mlp checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"


# RUN_TAG=""$DATASET"_pixel_"$MASK"_eval_"$EPOCH"_knn"
# NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
# JOB="python main.py user=callen_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_knn checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done; 

# ## MAE with patch 16
# DATASETs=(mae_tiny)
# RATIO=0.0
# PATCH_SIZEs=(16)
# for DATASET in "${DATASETs[@]}"
# do
# for PATCH_SIZE in "${PATCH_SIZEs[@]}"
# do

# RUN_TAG=""$DATASET"_pc_"$RATIO"_pixel_0.75_eval_patch_16"
# NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
# JOB="python main.py user=callen_euler experiment=$DATASET data.patch_size=$PATCH_SIZE trainer=eval run_tag=$RUN_TAG"
# echo $JOB
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:24g --wrap="nvidia-smi;$JOB"
# done; done;