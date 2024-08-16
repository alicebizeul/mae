
cd /local/home/abizeul/reconstruction

# python main.py --type ae --run_id ae --epochs 500 --seed 0 
# python main.py --type ae --run_id ae --epochs 800 --seed 0 --arch resnet34

# python main.py --type ae_mask --run_id ae_mask --epochs 800 --seed 0

# python main.py --type pca_ae --run_id pca_ae --epochs 800 --seed 0 --arch resnet18
# python main.py --type pca_ae --run_id pca_ae --epochs 800 --seed 0 --arch resnet34



# python main.py --type pca_ae --run_id pca_ae --epochs 500 --seed 0
# python main.py --type ae --run_id ae --epochs 800 --seed 0

# python main.py --type pca_ae --run_id pca_ae --epochs 800 --seed 0
# python main.py --type ae --run_id ae --epochs 400 --arch resnet18 --seed 0
python main.py --type ae --run_id pca_tiny_nomask_0.99_top_bottom --epochs 600 --seed 0 --dataset "tinyimagenet_0.99_percent_top" --dataset2 "tinyimagenet_0.99_percent_bottom" --eval_freq 100 --mask-ratio 0.0 --eval_epochs 100  --batch_size 128
python main.py --type ae --run_id pca_tiny_nomask_0.99_top_bottom --epochs 600 --seed 0 --dataset "tinyimagenet_0.99_percent_top" --dataset2 "tinyimagenet_0.99_percent_bottom" --eval_freq 100 --mask-ratio 0.75 --eval_epochs 100  --batch_size 128
python main.py --type ae --run_id pca_tiny_nomask_0.99_top_bottom --epochs 600 --seed 0 --dataset "tinyimagenet_0.99_percent_top" --dataset2 "tinyimagenet_0.99_percent_bottom" --eval_freq 100 --mask-ratio 0.50 --eval_epochs 100  --batch_size 128
python main.py --type ae --run_id pca_tiny_nomask_0.99_top_bottom --epochs 600 --seed 0 --dataset "tinyimagenet_0.99_percent_top" --dataset2 "tinyimagenet_0.99_percent_bottom" --eval_freq 100 --mask-ratio 0.90 --eval_epochs 100  --batch_size 128
python main.py --type ae --run_id pca_tiny_nomask_0.99_top_bottom --epochs 600 --seed 0 --dataset "tinyimagenet_0.99_percent_top" --dataset2 "tinyimagenet_0.98_percent_bottom" --eval_freq 100 --mask-ratio 0.0 --eval_epochs 100  --batch_size 128
python main.py --type ae --run_id pca_tiny_nomask_0.99_top_bottom --epochs 600 --seed 0 --dataset "tinyimagenet_0.99_percent_top" --dataset2 "tinyimagenet_0.95_percent_bottom" --eval_freq 100 --mask-ratio 0.0 --eval_epochs 100  --batch_size 128

# python main.py --type ae_mask --run_id ae_mask --epochs 600 --seed 0 --dataset "tiny-imagenet-200" --eval_freq 200 --mask-ratio 0.75 --eval_epochs 100
# python main.py --type ae_mask --run_id ae_mask --epochs 1000 --seed 0 --dataset "tiny-imagenet-200" --eval_freq 200 --mask-ratio 0.75 --eval_epochs 100
# python main.py --type ae_mask --run_id ae_mask --epochs 1000 --seed 0 --dataset "tiny-imagenet-200" --eval_freq 200 --mask-ratio 0.50 --eval_epochs 100

# python main.py --type ae --run_id ae_mask --epochs 1 --seed 0 --dataset "tinyimagenet_0.99_percent" --eval_freq 600 --mask-ratio 0.0 --eval_epochs 100 --eval --pretrained "/local/home/abizeul/reconstruction/outputs/ae_mask/ViT/0/0.0/networks/model_600_epochs.pth"
