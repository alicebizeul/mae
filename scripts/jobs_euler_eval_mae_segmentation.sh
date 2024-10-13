#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

cd /cluster/home/callen/projects/mae

NUM_WORKERS=8
TIME=24:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=12G

DATASET=mae_clevr
STRATEGIES=(segmentation_complete)
MASK=0.0
EPOCHs=(100 200 300 400 500 600)
for STRATEGY in "${STRATEGIES[@]}"
do
for EPOCH in "${EPOCHs[@]}"
do
    RUN_TAG="${DATASET}_${STRATEGY}_pixel_${MASK}_eval_"$EPOCH"_lin"
    NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
    JOB="python main.py user=callen_euler experiment=$DATASET masking=$STRATEGY masking.pixel_ratio=$MASK trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
    sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

    # RUN_TAG="${DATASET}_${STRATEGY}_pixel_${MASK}_eval_"$EPOCH"_knn"
    # NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
    # JOB="python main.py user=callen_euler experiment=$DATASET masking=$STRATEGY masking.pixel_ratio=$MASK trainer=eval_knn checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
    # sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;


STRATEGIES=(segmentation_random segmentation_partial)
MASKs=(0.5 0.6 0.7 0.8 0.9)
EPOCHs=(100 200)
for STRATEGY in "${STRATEGIES[@]}"
do
for MASK in "${MASKs[@]}"
do
for EPOCH in "${EPOCHs[@]}"
do
    RUN_TAG="${DATASET}_${STRATEGY}_pixel_${MASK}_eval_"$EPOCH"_lin"
    NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
    JOB="python main.py user=callen_euler experiment=$DATASET masking=$STRATEGY masking.pixel_ratio=$MASK trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
    sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

    # RUN_TAG="${DATASET}_${STRATEGY}_pixel_${MASK}_eval_"$EPOCH"_knn"
    # NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
    # JOB="python main.py user=callen_euler experiment=$DATASET masking=$STRATEGY masking.pixel_ratio=$MASK trainer=eval_knn checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
    # sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done; done;