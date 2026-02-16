#!/bin/bash

#SBATCH --job-name=dpo_train_aff_tpsa
#SBATCH -A research
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=dpo_train_aff_tpsa.out
#SBATCH --error=dpo_train_aff_tpsa.err
#SBATCH --time=4-00:00:00

echo "Starting ..."
python3 dpo_training_generation.py --model_properties affinity tpsas --preference_properties affinity

