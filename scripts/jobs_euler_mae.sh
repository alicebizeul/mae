#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

cd /cluster/home/callen/projects/mae

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G

DATASETs=(mae_clevr)
MASKs=(0.5 0.6 0.7 0.8 0.9 0.75)
for DATASET in "${DATASETs[@]}"
do
for MASK in "${MASKs[@]}"
do
RUN_TAG=""$DATASET"_pixel_"$MASK""
NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
JOB="python main.py user=callen_euler experiment=$DATASET masking.pixel_ratio=$MASK run_tag=$RUN_TAG"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;
