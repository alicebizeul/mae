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
# DATASETs=(mae_cifar10 mae_tiny mae_blood mae_path mae_derma)
DATASETs=(mae_tiny mae_path)

MODEL="vit-b"
for DATASET in "${DATASETs[@]}"
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
#     MASK=0.8
# fi

MASK=0.75

RUN_TAG=""$DATASET"_pixel_"$MASK"_model_"$MODEL""
NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=callen_euler experiment=$DATASET run_tag=$RUN_TAG masking.pixel_ratio=$MASK model=$MODEL"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done;
