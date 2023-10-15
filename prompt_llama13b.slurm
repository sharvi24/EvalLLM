#!/bin/bash
#SBATCH --job-name=llama2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:2


eval "$(conda shell.bash hook)"
conda activate nlp
cd /scratch/sb7787/sharvi/EvalLLM
torchrun --nproc_per_node 2 run_llama2.py \
--ckpt_dir /vast/work/public/ml-datasets/llama-2/llama-2-13b \
--tokenizer_path /vast/work/public/ml-datasets/llama-2/tokenizer.model \

