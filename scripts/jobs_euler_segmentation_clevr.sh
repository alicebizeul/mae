#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

cd /cluster/home/callen/projects/mae

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G

DATASET=mae_clevr
STRATEGIES=(segmentation_complete segmentation_random segmentation_partial)
for STRATEGY in "${STRATEGIES[@]}"
do
    RUN_TAG="${DATASET}_${STRATEGY}"
    NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
    JOB="python main.py user=callen_euler experiment=$DATASET masking=$STRATEGY run_tag=$RUN_TAG"
    echo $JOB
    sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done;
