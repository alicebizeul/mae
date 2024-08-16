#!/bin/bash 

conda activate reconstruction 

cd /cluster/home/abizeul/mae

NUM_WORKERS=8
TIME=120:00:00
SAVEDIR="/cluster/work/vogtlab/Group/abizeul/mae/outputs"
SMALL_SCALE="--small_scale"

RATIO=0.99
MASK=0.0
DATASET1="tinyimagenet_"$RATIO"_percent_bottom"
DATASET2="tinyimagenet_"$RATIO"_percent_top"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_inverted_"$SMALL_SCALE""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_bottom_top  --epochs 600 --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# #############
RATIO=0.99
MASK=0.0
DATASET1="tinyimagenet_"$RATIO"_percent_top"
DATASET2="tinyimagenet_"$RATIO"_percent_bottom"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_"$SMALL_SCALE""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom  --epochs 600 --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# #######ViT Small - 0.99 - 0.0######
RATIO=0.99
MASK=0.0
DATASET1="tinyimagenet_"$RATIO"_percent_top"
DATASET2="tinyimagenet_"$RATIO"_percent_bottom"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
ARCH="vit_s"
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_arch_"$ARCH"_"$SMALL_SCALE""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_"$ARCH"_top_bottom  --epochs 600 --arch $ARCH --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"


# ################
RATIO=0.98
MASK=0.0
DATASET1="tinyimagenet_"$RATIO"_percent_top"
DATASET2="tinyimagenet_"$RATIO"_percent_bottom"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_"$SMALL_SCALE""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom  --epochs 600 --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
RATIO=0.95
MASK=0.0
DATASET1="tinyimagenet_"$RATIO"_percent_top"
DATASET2="tinyimagenet_"$RATIO"_percent_bottom"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_"$SMALL_SCALE""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom  --epochs 600 --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
RATIO=0.99
MASK=0.75
DATASET1="tinyimagenet_"$RATIO"_percent_top"
DATASET2="tinyimagenet_"$RATIO"_percent_bottom"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_"$SMALL_SCALE""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
RATIO=0.99
MASK=0.7
DATASET1="tinyimagenet_"$RATIO"_percent_top"
DATASET2="tinyimagenet_"$RATIO"_percent_bottom"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_"$SMALL_SCALE""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
RATIO=0.99
MASK=0.8
DATASET1="tinyimagenet_"$RATIO"_percent_top"
DATASET2="tinyimagenet_"$RATIO"_percent_bottom"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK"_"$SMALL_SCALE""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

# ################
RATIO=0.99
MASK=0.5
DATASET1="tinyimagenet_"$RATIO"_percent_top"
DATASET2="tinyimagenet_"$RATIO"_percent_bottom"
if [ ! -d \"\$TMP/"$DATASET1"\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/"$DATASET1".zip \$TMP/; \
        unzip -q \$TMP/"$DATASET1".zip -d \$TMP/; \
    fi; \
if [ ! -d \"\$TMP/$DATASET2\" ]; then \
        echo \"Copying data to \${TMP}\";
        cp -r /cluster/work/vogtlab/Group/abizeul/data/$DATASET2.zip \$TMP/; \
        unzip -q \$TMP/$DATASET2.zip -d \$TMP/; \
    fi; \
NAME="/cluster/home/abizeul/mae/output_log/tiny_train_ratio_"$RATIO"_mask_"$MASK""
JOB="python main.py --save_dir $SAVEDIR --run_id pca_tiny_"$RATIO"_"$MASK"_top_bottom --epochs 600 --seed 0 --root "$TMP" --dataset $DATASET1 --dataset2 $DATASET2 --eval_freq 100 --mask-ratio "$MASK" --eval_epochs 100  --batch_size 512 $SMALL_SCALE"
echo $JOB
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu 25G --time="$TIME" -p gpu --gres=gpu:rtx4090:1 --wrap="nvidia-smi;$JOB"

