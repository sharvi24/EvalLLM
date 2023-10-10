#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1


eval "$(conda shell.bash hook)"
conda activate nlp
cd /scratch/sb7787/sharvi/EvalLLM
torchrun run_llama2.py
