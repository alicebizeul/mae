#!/bin/bash 

conda activate reconstruction 

cd /cluster/home/abizeul/mae


NUM_WORKERS=8
TIME=120:00:00
SAVEDIR="/cluster/work/vogtlab/Group/abizeul/mae/outputs"

RATIO=0.99
MASK=0.0
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_inverted"
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_bottom_top --epochs 600 --seed 0 --root "/cluster/work/vogtlab/Group/abizeul/data" --dataset "tinyimagenet_"$RATIO"_percent_bottom" --dataset2 "tinyimagenet_"$RATIO"_percent_top" --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 128"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# #############

# RATIO=0.99
# MASK=0.0
# NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK""
# JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "/cluster/work/vogtlab/Group/abizeul/data" --dataset "tinyimagenet_"$RATIO"_percent_top" --dataset2 "tinyimagenet_"$RATIO"_percent_bottom" --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 128"
# echo $JOB
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
# RATIO=0.98
# MASK=0.0
# NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK""
# JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "/cluster/work/vogtlab/Group/abizeul/data" --dataset "tinyimagenet_"$RATIO"_percent_top" --dataset2 "tinyimagenet_"$RATIO"_percent_bottom" --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 128"
# echo $JOB
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
# RATIO=0.95
# MASK=0.0
# NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK""
# JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "/cluster/work/vogtlab/Group/abizeul/data" --dataset "tinyimagenet_"$RATIO"_percent_top" --dataset2 "tinyimagenet_"$RATIO"_percent_bottom" --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 128"
# echo $JOB
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
# RATIO=0.99
# MASK=0.75
# NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK""
# JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "/cluster/work/vogtlab/Group/abizeul/data" --dataset "tinyimagenet_"$RATIO"_percent_top" --dataset2 "tinyimagenet_"$RATIO"_percent_bottom" --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 128"
# echo $JOB
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
# RATIO=0.99
# MASK=0.7
# NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK""
# JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "/cluster/work/vogtlab/Group/abizeul/data" --dataset "tinyimagenet_"$RATIO"_percent_top" --dataset2 "tinyimagenet_"$RATIO"_percent_bottom" --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 128"
# echo $JOB
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
# RATIO=0.99
# MASK=0.8
# NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK""
# JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "/cluster/work/vogtlab/Group/abizeul/data" --dataset "tinyimagenet_"$RATIO"_percent_top" --dataset2 "tinyimagenet_"$RATIO"_percent_bottom" --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 128"
# echo $JOB
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
# RATIO=0.99
# MASK=0.5
# NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK""
# JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "/cluster/work/vogtlab/Group/abizeul/data" --dataset "tinyimagenet_"$RATIO"_percent_top" --dataset2 "tinyimagenet_"$RATIO"_percent_bottom" --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 128"
# echo $JOB
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

