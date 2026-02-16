#!/bin/bash

#SBATCH --job-name=dpo_aff_logp_generate_molecules
#SBATCH -A research
#SBATCH -c 30
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2G
#SBATCH --output=dpo_train_affinity.out
#SBATCH --error=dpo_train_affinity.err
#SBATCH --time=4-00:00:00
#SBATCH -w gnode058

echo "Starting ..."
python3 dpo_training.py --properties affinity logps