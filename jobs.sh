
cd /local/home/abizeul/mae
conda activate reconstruction


python main.py pl_module.eval_freq=10 trainer.max_epochs=100 evaluator.max_epochs=100 pl_module.warmup=1